# main.py
from datasets.medical_tensors import MedicalTensorDataset
from model import generate_model
from lightning_module import MedicalClassificationModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from config import ModelConfig
from omegaconf import OmegaConf
import torch
import argparse

# === Импортируем MONAI аугментации ===
from monai.transforms import Compose, RandFlip, RandRotate90, RandGaussianNoise


def get_train_transforms():
    """Безопасные аугментации для тренировочной выборки."""
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate90(prob=0.5, max_k=1, spatial_axes=(1, 2)),
        RandGaussianNoise(prob=0.25, std=0.01),
    ])


def get_val_transforms():
    """Аугментации для валидации (обычно нет)."""
    return None


def main():
    # Создаем конфиг по умолчанию
    cfg = ModelConfig()
    # Преобразуем в OmegaConf
    cfg = OmegaConf.structured(cfg)

    # Переопределяем параметры из CLI
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Обработка ci_test как раньше
    if cfg.ci_test:
        cfg.img_list = '../toy_data/test_ci.txt'
        cfg.n_epochs = 1
        cfg.no_cuda = True
        cfg.data_root = '../toy_data'
        cfg.pretrain_path = ''
        cfg.num_workers = 4
        cfg.model_depth = 10
        cfg.resnet_shortcut = 'A'
        cfg.input_D = 14
        cfg.input_H = 28
        cfg.input_W = 28
        cfg.batch_size = 2

    # Установка seed
    torch.manual_seed(cfg.manual_seed)

    # Создаем namespace объект для совместимости
    cfg_dict = OmegaConf.to_container(cfg)

    # Добавляем недостающие атрибуты
    cfg_dict['gpu_id'] = [] if cfg.no_cuda else [0]
    cfg_dict['phase'] = 'train'
    cfg_dict['pin_memory'] = not cfg.no_cuda

    # Используем argparse.Namespace
    cfg_namespace = argparse.Namespace(**cfg_dict)

    # УСТАНОВИТЬ НЕДЕТЕРМИНИРОВАННОСТЬ ПЕРЕД СОЗДАНИЕМ МОДЕЛИ
    torch.use_deterministic_algorithms(False)

    model, parameters = generate_model(cfg_namespace)
    lightning_model = MedicalClassificationModel(
        model,
        learning_rate=cfg.learning_rate,
        #num_classes=cfg.n_seg_classes
    )

    # === Создаем аугментации ===
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # === Создаем датасеты с аугментациями ===
    train_dataset = MedicalTensorDataset(
        cfg.data_root,
        cfg.img_list,
        cfg_namespace,
        transform=train_transforms  # <-- добавляем аугментации
    )

    # Уменьшаем num_workers для избежания проблем с pickle
    num_workers = 0 if cfg.ci_test else 4  # временно 0 для стабильности

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg_namespace.pin_memory,
        persistent_workers=num_workers > 0
    )

    val_dataset = MedicalTensorDataset(
        cfg.val_data_root,
        cfg.val_list,
        cfg_namespace,
        transform=val_transforms  # <-- без аугментаций
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg_namespace.pin_memory,
        persistent_workers=num_workers > 0
    )

    # Logger & Checkpointing
    tb_logger = TensorBoardLogger("tb_logs", name="medical_classification")
    csv_logger = CSVLogger("logs", name="medical_classification")

    # Callback для сохранения лучших весов
    best_model_checkpoint = ModelCheckpoint(
        dirpath=cfg.save_folder,
        filename="best-checkpoint-{epoch:02d}-{val_acc:.3f}-{val_f1:.3f}",
        save_top_k=cfg.save_top_k,
        monitor=cfg.monitor_metric,
        mode=cfg.checkpoint_mode,
        save_weights_only=True,
        verbose=True
    )

    # Callback для сохранения последнего чекпоинта
    last_model_checkpoint = ModelCheckpoint(
        dirpath=cfg.save_folder,
        filename="last-checkpoint-{epoch:02d}-{val_acc:.3f}-{val_f1:.3f}",
        save_top_k=1,
        monitor=None,
        save_weights_only=True,
        verbose=True
    )

    # Callback для ранней остановки
    early_stopping = EarlyStopping(
        monitor=cfg.early_stopping_metric,
        min_delta=cfg.early_stopping_min_delta,
        patience=cfg.early_stopping_patience,
        verbose=True,
        mode=cfg.checkpoint_mode
    )

    # Callback для мониторинга learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer - правильно настраиваем accelerator и devices
    if cfg.no_cuda or not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "gpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        logger=[tb_logger, csv_logger],
        callbacks=[best_model_checkpoint, last_model_checkpoint, early_stopping, lr_monitor],
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=cfg.ci_test,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        # deterministic=cfg.deterministic,  # ПОЛНОСТЬЮ УБРАТЬ ЭТУ СТРОКУ
        gradient_clip_val=cfg.gradient_clip_val,
    )

    # Train
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # После обучения загружаем лучшую модель и сохраняем только веса
    best_checkpoint_path = best_model_checkpoint.best_model_path
    if best_checkpoint_path:
        print(f"Загружаем лучшую модель из: {best_checkpoint_path}")

        # Загружаем модель с лучшими весами
        best_model = MedicalClassificationModel.load_from_checkpoint(
            best_checkpoint_path,
            model=model,
            learning_rate=cfg.learning_rate,
            num_classes=cfg.n_seg_classes
        )

        # Сохраняем только веса модели (state_dict)
        torch.save(best_model.model.state_dict(), f"{cfg.save_folder}/best_weights.pth")
        print(f"Лучшие веса сохранены в: {cfg.save_folder}/best_weights.pth")


if __name__ == '__main__':
    main()