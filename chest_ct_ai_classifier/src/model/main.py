# main.py
from datasets.medical_tensors import MedicalTensorDataset
from model import generate_model
from lightning_module import MedicalClassificationModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from config import ModelConfig
from omegaconf import OmegaConf
import torch
import argparse


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

    model, parameters = generate_model(cfg_namespace)
    lightning_model = MedicalClassificationModel(model, cfg.learning_rate)

    # Создаем датасет
    train_dataset = MedicalTensorDataset(cfg.data_root, cfg.img_list, cfg_namespace)

    # Уменьшаем num_workers для избежания проблем с pickle
    num_workers = 0 if cfg.ci_test else 4  # временно 0 для стабильности

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg_namespace.pin_memory
    )
    val_dataset = MedicalTensorDataset(cfg.val_data_root, cfg.val_list, cfg_namespace)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # Валидация не требует перемешивания
        num_workers=num_workers,
        pin_memory=cfg_namespace.pin_memory
    )

    # Logger & Checkpointing
    logger = TensorBoardLogger("tb_logs", name="medical_classification")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_folder,
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min"
    )

    # Trainer - правильно настраиваем accelerator и devices
    if cfg.no_cuda or not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1  # для CPU devices должно быть int > 0
    else:
        accelerator = "gpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=devices,  # исправлено
        fast_dev_run=cfg.ci_test,
        log_every_n_steps=1
    )

    # Train
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader  #
    )


if __name__ == '__main__':
    main()