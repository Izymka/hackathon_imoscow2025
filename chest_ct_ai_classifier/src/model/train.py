import os
import time
import logging
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from config import ModelConfig
from datasets.medical_tensors import MedicalTensorDataset
from model_generator import generate_model


def setup_logging(log_file='logs/train.log'):
    """Создание логгера."""
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Консоль
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файл
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def build_optimizer(parameters, cfg: ModelConfig):
    """
    Создает оптимизатор с разными learning rate
    для новых и базовых слоев.

    Args:
        parameters: dict {'base_parameters': [...], 'new_parameters': [...]}
                    или iterable параметров
        cfg: ModelConfig объект конфигурации
    """
    # Если parameters - словарь (fine-tuning режим)
    if isinstance(parameters, dict):
        base_lr_mult = cfg.base_lr_multiplier
        new_lr_mult = cfg.new_layers_lr_multiplier

        base_lr = cfg.learning_rate * base_lr_mult
        new_lr = cfg.learning_rate * new_lr_mult

        print(f"🎯 Fine-tuning режим с differential LR:")
        print(f"   Base layers (layer3+layer4): {base_lr:.2e} (x{base_lr_mult})")
        print(f"   New layers (FC): {new_lr:.2e} (x{new_lr_mult})")

        params = [
            {
                'params': parameters['base_parameters'],
                'lr': base_lr,
                'weight_decay': cfg.weight_decay
            },
            {
                'params': parameters['new_parameters'],
                'lr': new_lr,
                'weight_decay': cfg.weight_decay * 0.1  # меньше для нового слоя
            }
        ]
    else:
        # Обучение с нуля - единый LR
        print(f"🎯 Обучение с нуля с единым LR: {cfg.learning_rate:.2e}")
        params = parameters

    # Выбор оптимизатора
    optimizer_name = getattr(cfg, 'optimizer', 'adam').lower()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=cfg.learning_rate if not isinstance(parameters, dict) else params[0]['lr'],
            weight_decay=cfg.weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=cfg.learning_rate if not isinstance(parameters, dict) else params[0]['lr'],
            momentum=getattr(cfg, 'momentum', 0.9),
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")

    return optimizer


def train(data_loader, model, optimizer, scheduler,
          total_epochs, save_interval, save_folder, cfg: ModelConfig):
    logger = setup_logging()
    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.no_cuda) else "cpu")
    logger.info(f"🖥️  Используем устройство: {device}")

    # Критерий потерь
    if cfg.use_focal_loss:
        # Focal Loss для несбалансированных данных
        from torch.nn import functional as F

        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        loss_cls = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma).to(device)
        logger.info(f"📊 Функция потерь: FocalLoss (α={cfg.focal_alpha}, γ={cfg.focal_gamma})")
    elif cfg.n_seg_classes == 2:
        loss_cls = nn.CrossEntropyLoss().to(device)
        logger.info("📊 Функция потерь: CrossEntropyLoss (binary)")
    else:
        loss_cls = nn.CrossEntropyLoss().to(device)
        logger.info(f"📊 Функция потерь: CrossEntropyLoss ({cfg.n_seg_classes} классов)")

    model.to(device)
    model.train()

    batches_per_epoch = len(data_loader)
    train_time_sp = time.time()

    # Статистика
    best_loss = float('inf')
    total_batches = 0

    for epoch in range(total_epochs):
        logger.info('=' * 60)
        logger.info('Эпоха %d/%d', epoch + 1, total_epochs)
        logger.info('=' * 60)

        # ExponentialLR – step() в начале эпохи
        if isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
            scheduler.step()

        # Текущий lr (может быть несколько групп)
        lrs = scheduler.get_last_lr()
        if len(lrs) > 1:
            logger.info('Learning rates: base=%.6f, new=%.6f', lrs[0], lrs[1])
        else:
            logger.info('Learning rate: %.6f', lrs[0])

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_id, (volumes, labels) in enumerate(data_loader):
            total_batches += 1

            volumes, labels = volumes.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(volumes)
            loss = loss_cls(outputs, labels)
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Метрики
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100.0 * correct / labels.size(0)

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += labels.size(0)

            avg_batch_time = (time.time() - train_time_sp) / total_batches

            # Логирование каждые log_every_n_steps батчей
            if (batch_id + 1) % cfg.log_every_n_steps == 0 or (batch_id + 1) == batches_per_epoch:
                logger.info(
                    'Batch[%d/%d] loss=%.4f acc=%.2f%% avg_time=%.3fs',
                    batch_id + 1, batches_per_epoch,
                    loss.item(), accuracy, avg_batch_time
                )

            # Сохранение чекпоинта по интервалу
            if (not cfg.ci_test and total_batches % save_interval == 0):
                model_save_path = os.path.join(
                    save_folder, f'checkpoint_epoch{epoch + 1}_batch{batch_id + 1}.pth.tar'
                )
                os.makedirs(save_folder, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'batch_id': batch_id + 1,
                    'total_batches': total_batches,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss.item()
                }, model_save_path)
                logger.info('💾 Checkpoint saved: %s', model_save_path)

        # Статистика за эпоху
        avg_epoch_loss = epoch_loss / batches_per_epoch
        avg_epoch_acc = 100.0 * epoch_correct / epoch_total

        logger.info('─' * 60)
        logger.info('Эпоха %d завершена: avg_loss=%.4f avg_acc=%.2f%%',
                    epoch + 1, avg_epoch_loss, avg_epoch_acc)
        logger.info('─' * 60)

        # Сохранение лучшей модели
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(save_folder, 'best_model.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
                'accuracy': avg_epoch_acc
            }, best_model_path)
            logger.info('⭐ Новая лучшая модель сохранена! Loss: %.4f', best_loss)

        # ReduceLROnPlateau – step() после эпохи
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)

    logger.info('=' * 60)
    logger.info('✅ Обучение завершено! Лучший loss: %.4f', best_loss)
    logger.info('=' * 60)


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)

    # Загрузка конфигурации
    cfg = ModelConfig()

    # Вывод конфигурации
    print("=" * 60)
    print("⚙️  КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
    print("=" * 60)
    print(cfg.get_training_summary())
    print(cfg.get_augmentation_summary())

    # Валидация конфига
    warnings = cfg.validate_config()
    if warnings:
        print("⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for w in warnings:
            print(f"   • {w}")
        print()

    if cfg.ci_test:
        cfg.img_list = '../toy_data/test_ci.txt'
        cfg.n_epochs = 1
        cfg.no_cuda = True
        cfg.data_root = '../toy_data'
        cfg.pretrain_path = ''
        cfg.num_workers = 0
        cfg.model_depth = 10
        cfg.resnet_shortcut = 'A'
        cfg.input_D, cfg.input_H, cfg.input_W = 14, 28, 28

    # Установка seed для воспроизводимости
    torch.manual_seed(cfg.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.manual_seed)

    # Детерминизм (может замедлить обучение)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Генерация модели
    print("=" * 60)
    print("🔧 Создание модели...")


    # Преобразуем ModelConfig в объект с нужными атрибутами для generate_model
    class OptNamespace:
        pass


    opt = OptNamespace()
    for key, value in cfg.__dict__.items():
        setattr(opt, key, value)
    opt.phase = 'train'
    opt.gpu_id = [0] if not cfg.no_cuda and torch.cuda.is_available() else []

    model, parameters = generate_model(opt)
    print("=" * 60)

    # Создание оптимизатора с использованием функции
    optimizer = build_optimizer(parameters, cfg)

    # Планировщик LR
    if cfg.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=getattr(cfg, 'lr_gamma', 0.99)
        )
        print(f"📉 Scheduler: ExponentialLR (gamma={getattr(cfg, 'lr_gamma', 0.99)})")
    elif cfg.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
            min_lr=cfg.lr_scheduler_min_lr,
            verbose=True
        )
        print(
            f"📉 Scheduler: ReduceLROnPlateau (factor={cfg.lr_scheduler_factor}, patience={cfg.lr_scheduler_patience})")
    elif cfg.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs,
            eta_min=cfg.lr_scheduler_min_lr
        )
        print(f"📉 Scheduler: CosineAnnealingLR (T_max={cfg.n_epochs})")
    else:
        # Dummy scheduler (не меняет LR)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        print("📉 Scheduler: None (constant LR)")

    # Загрузка чекпоинта (если есть)
    resume_path = getattr(cfg, 'resume_path', None)
    if resume_path and os.path.isfile(resume_path):
        print(f"📥 Загрузка checkpoint: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location='cpu' if cfg.no_cuda else None,
            weights_only=False
        )

        # Загрузка state_dict с обработкой DataParallel
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict

        new_state_dict = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
        model.load_state_dict(new_state_dict, strict=False)

        # Загрузка оптимизатора и scheduler
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"✅ Checkpoint загружен (эпоха {checkpoint.get('epoch', '?')})")

    # Подготовка данных
    print("=" * 60)
    print("📊 Загрузка данных...")
    opt.phase = 'train'
    opt.pin_memory = cfg.pin_memory
    dataset = MedicalTensorDataset(cfg.data_root, cfg.img_list, opt)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    print(f"   Датасет: {len(dataset)} образцов")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Батчей в эпохе: {len(loader)}")
    print("=" * 60)

    # Запуск обучения
    train(
        loader, model, optimizer, scheduler,
        total_epochs=cfg.n_epochs,
        save_interval=cfg.save_intervals,
        save_folder=cfg.save_folder,
        cfg=cfg
    )