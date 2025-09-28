import os
import time
import logging
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from setting import parse_opts
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
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def build_optimizer(model, cfg):
    """
    Создает оптимизатор с разными learning rate
    для новых и базовых слоев согласно конфигу.
    """
    if not cfg.use_differential_lr:
        # обычный путь
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

    # ---------- группировка параметров ----------
    # новые слои (размороженные для fine-tuning)
    new_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(layer in name for layer in cfg.new_layer_names):
            new_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.Adam(
        [
            {"params": base_params,
             "lr": cfg.learning_rate * cfg.base_lr_multiplier},
            {"params": new_params,
             "lr": cfg.learning_rate * cfg.new_layers_lr_multiplier}
        ],
        weight_decay=cfg.weight_decay
    )
    return optimizer


def train(data_loader, model, optimizer, scheduler,
          total_epochs, save_interval, save_folder, sets):

    logger = setup_logging()
    device = torch.device("cuda" if (torch.cuda.is_available() and not sets.no_cuda) else "cpu")
    print(f"🖥️  Используем устройство [x]: {device}")

    # === FIX: добавлен критерий потерь ===
    loss_cls = nn.CrossEntropyLoss().to(device)

    model.to(device)
    model.train()

    # === FIX: количество батчей в эпохе ===
    batches_per_epoch = len(data_loader)
    train_time_sp = time.time()

    for epoch in range(total_epochs):
        logger.info('Start epoch %d', epoch + 1)

        # ExponentialLR – step() в начале эпохи
        if isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
            scheduler.step()

        # Текущий lr
        current_lr = scheduler.get_last_lr()[0]
        logger.info('Learning rate: %.6f', current_lr)

        for batch_id, (volumes, labels) in enumerate(data_loader):
            batch_id_sp = epoch * batches_per_epoch + batch_id

            volumes, labels = volumes.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(volumes)
            loss = loss_cls(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100.0 * correct / labels.size(0)

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            logger.info(
                'Epoch[%d/%d] Batch[%d/%d] loss=%.4f acc=%.2f%% avg_time=%.3fs',
                epoch + 1, total_epochs, batch_id + 1, batches_per_epoch,
                loss.item(), accuracy, avg_batch_time
            )

            # Сохранение чекпоинта
            if (not sets.ci_test and batch_id == 0
                and batch_id_sp != 0 and batch_id_sp % save_interval == 0):
                model_save_path = os.path.join(
                    save_folder, f'epoch_{epoch+1}_batch_{batch_id+1}.pth.tar'
                )
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'batch_id': batch_id + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)
                logger.info('Checkpoint saved: %s', model_save_path)

        # ReduceLROnPlateau – step() после эпохи
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.item())

    logger.info('Training finished')


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    sets = parse_opts()

    if sets.ci_test:
        sets.img_list = '../toy_data/test_ci.txt'
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = '../toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D, sets.input_H, sets.input_W = 14, 28, 28

    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)

    # === Настройка оптимизатора ===
    # generate_model теперь может вернуть:
    #   1) обычный iterable параметров (старый случай)
    #   2) dict {'base_parameters': ..., 'new_parameters': ...}
    #      если вы заранее пометили новые слои (fc/layer3/layer4)
    if isinstance(parameters, dict):
        # --- DIFF-LR ---
        # базовые слои учим медленно
        base_lr = sets.learning_rate * getattr(sets, 'base_lr_multiplier', 0.1)
        # новые слои (fc, layer3/4) учим быстрее
        new_lr = sets.learning_rate * getattr(sets, 'new_layers_lr_multiplier', 1.0)
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
        ]
        print(f"💡 Differential LR: base={base_lr:.2e}, new={new_lr:.2e}")
    else:
        # если ничего не размораживали
        params = [{'params': list(parameters), 'lr': sets.learning_rate}]
        print(f"💡 Single LR: {sets.learning_rate:.2e}")

    # оптимизатор
    optimizer = optim.SGD(
        params,
        momentum=0.9,
        weight_decay=getattr(sets, 'weight_decay', 1e-3)
    )
    # планировщик (пример – экспоненциальный)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=getattr(sets, 'lr_gamma', 0.99)
    )

    # === Загрузка чекпоинта (если есть) ===
    if sets.resume_path and os.path.isfile(sets.resume_path):
        checkpoint = torch.load(
            sets.resume_path,
            map_location='cpu' if sets.no_cuda else None
        )
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
        model.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))

    # === Данные ===
    sets.phase = 'train'
    sets.pin_memory = not sets.no_cuda
    dataset = MedicalTensorDataset(sets.data_root, sets.img_list, sets)
    loader = DataLoader(dataset, batch_size=sets.batch_size,
                        shuffle=True, num_workers=sets.num_workers,
                        pin_memory=sets.pin_memory)

    # === Обучение ===
    train(loader, model, optimizer, scheduler,
          total_epochs=sets.n_epochs,
          save_interval=sets.save_intervals,
          save_folder=sets.save_folder,
          sets=sets)
