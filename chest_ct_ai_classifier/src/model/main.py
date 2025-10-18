# main.py
import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# === Локальные импорты ===
from datasets.medical_tensors import MedicalTensorDataset
from model_generator import generate_model
from lightning_module import MedicalClassificationModel
from config import ModelConfig

# === MONAI аугментации ===
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandGaussianNoise,
    RandShiftIntensity,
    RandAdjustContrast,
    RandScaleIntensity,
    CropForeground,
    SpatialPad,
    EnsureChannelFirst,
    Orientation,
    ToTensor
)

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


def get_safe_train_transforms(input_size: Tuple[int, int, int]) -> Compose:
    """
    Безопасные аугментации для медицинских 3D изображений.
    Сохраняют медицинскую информативность данных.
    """
    return Compose([
        #EnsureChannelFirst(),

        # Пространственные аугментации (консервативные)
        RandFlip(prob=0.3, spatial_axis=0),  # только по одной оси
        RandRotate90(prob=0.2, max_k=1, spatial_axes=(1, 2)),  # минимальное вращение

        # Интенсивностные аугментации (слабые)
        RandGaussianNoise(prob=0.15, std=0.005),  # очень слабый шум
        RandShiftIntensity(offsets=0.05, prob=0.2),  # сдвиг интенсивности
        RandAdjustContrast(gamma=(0.9, 1.1), prob=0.2),  # минимальная коррекция контраста
        RandScaleIntensity(factors=(-0.05, 0.05), prob=0.2),  # масштабирование

        #ToTensor(),
    ])


def get_val_transforms() -> Compose:
    """Трансформации для валидации (только нормализация)."""
    return Compose([
        #EnsureChannelFirst(),
        #ToTensor(),
    ])


class CrossValidationTrainer:
    """Класс для проведения кросс-валидации."""

    def __init__(self, cfg: OmegaConf, cfg_namespace: argparse.Namespace):
        self.cfg = cfg
        self.cfg_namespace = cfg_namespace
        self.results = []

    def load_data_labels(self) -> Tuple[List[str], List[int]]:
        """Загружает пути к файлам и метки для стратификации."""
        labels_df = pd.read_csv(self.cfg.img_list)

        # Предполагаем, что структура CSV: filename, label
        if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
            raise ValueError("CSV должен содержать колонки 'filename' и 'label'")

        filenames = labels_df['filename'].tolist()
        labels = labels_df['label'].tolist()

        return filenames, labels

    def create_fold_datasets(self, train_indices: List[int], val_indices: List[int],
                             filenames: List[str], labels: List[int]) -> Tuple[DataLoader, DataLoader]:
        """Создает датасеты и загрузчики для фолда."""

        # Создаем временные CSV файлы для фолда
        train_data = [(filenames[i], labels[i]) for i in train_indices]
        val_data = [(filenames[i], labels[i]) for i in val_indices]

        # Сохраняем во временные файлы
        train_df = pd.DataFrame(train_data, columns=['filename', 'label'])
        val_df = pd.DataFrame(val_data, columns=['filename', 'label'])

        fold_train_path = f"temp_train_fold.csv"
        fold_val_path = f"temp_val_fold.csv"

        train_df.to_csv(fold_train_path, index=False)
        val_df.to_csv(fold_val_path, index=False)

        # Создаем датасеты
        train_dataset = MedicalTensorDataset(
            self.cfg.data_root,
            fold_train_path,
            self.cfg_namespace,
            transform=get_safe_train_transforms((self.cfg.input_D, self.cfg.input_H, self.cfg.input_W))
        )

        val_dataset = MedicalTensorDataset(
            self.cfg.data_root,
            fold_val_path,
            self.cfg_namespace,
            transform=get_val_transforms()
        )

        # Параметры DataLoader
        num_workers = 0 if self.cfg.ci_test else min(0, os.cpu_count())

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.cfg_namespace.pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.cfg_namespace.pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

        # Удаляем временные файлы
        os.remove(fold_train_path)
        os.remove(fold_val_path)

        return train_loader, val_loader

    def train_fold(self, fold: int, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Обучает модель на одном фолде."""

        rprint(f"\n📄 [bold blue]Обучение фолда {fold + 1}/{self.cfg.n_splits}[/bold blue]")

        # Создаем модель для фолда
        model, parameters = generate_model(self.cfg_namespace)

        # НЕ перемещаем модель вручную - Lightning сделает это автоматически
        device_name = 'GPU' if torch.cuda.is_available() and not self.cfg.no_cuda else 'CPU'
        print(f"✅ Устройство для обучения: {device_name}")

        # ИСПРАВЛЕНИЕ: Вычисляем веса классов на CPU
        class_weights = self.calculate_class_weights(train_loader)

        lightning_model = MedicalClassificationModel(
            model,
            learning_rate=self.cfg.learning_rate,
            num_classes=self.cfg.n_seg_classes,
            use_weighted_loss=True,
            class_weights=class_weights,
        )
        if device_name == 'GPU':
            lightning_model = lightning_model.to('cuda')
            print("Switched device: ", lightning_model.device)

        # Логгеры для фолда
        fold_name = f"medical_classification_fold_{fold + 1}"
        tb_logger = TensorBoardLogger("tb_logs", name=fold_name, version=f"fold_{fold + 1}")
        csv_logger = CSVLogger("logs", name=fold_name, version=f"fold_{fold + 1}")

        # Коллбэки
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.cfg.save_folder}/fold_{fold + 1}",
            filename="best-{epoch:02d}-{val_f1:.4f}-{val_recall:.4f}-{val_specificity:.4f}--{val_auroc:.4f}",
            save_top_k=-1,
            every_n_epochs=1,
            monitor=self.cfg.monitor_metric,
            mode=self.cfg.checkpoint_mode,
            save_weights_only=False,
            verbose=False
        )

        early_stopping = EarlyStopping(
            monitor=self.cfg.early_stopping_metric,
            min_delta=self.cfg.early_stopping_min_delta,
            patience=self.cfg.early_stopping_patience,
            verbose=False,
            mode=self.cfg.checkpoint_mode
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Настройка accelerator
        if self.cfg.no_cuda or not torch.cuda.is_available():
            accelerator = "cpu"
            devices = 1
        else:
            accelerator = "gpu"
            devices = 1

        # Создаем trainer
        trainer = pl.Trainer(
            max_epochs=self.cfg.n_epochs,
            logger=[tb_logger, csv_logger],
            callbacks=[
                checkpoint_callback,
                early_stopping,
                lr_monitor,
                RichProgressBar(),
            ],
            accelerator=accelerator,
            devices=devices,
            accumulate_grad_batches=getattr(self.cfg, 'accumulate_grad_batches', 1),  # Gradient accumulation
            fast_dev_run=self.cfg.ci_test,
            log_every_n_steps=min(10, len(train_loader) // 4),
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=self.cfg.gradient_clip_val,
            precision=32,
        )

        # Обучение
        trainer.fit(lightning_model, train_loader, val_loader)

        # Получение результатов
        best_metrics = checkpoint_callback.best_model_score.item()

        # Загружаем лучшую модель для финального тестирования
        best_model = MedicalClassificationModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=model,
            learning_rate=self.cfg.learning_rate,
            num_classes=self.cfg.n_seg_classes
        )

        # Валидация на лучшей модели
        trainer.validate(best_model, val_loader, verbose=False)

        fold_results = {
            'fold': fold + 1,
            'best_val_score': best_metrics,
            'best_checkpoint': checkpoint_callback.best_model_path,
            'final_epoch': trainer.current_epoch,
        }

        # Добавляем все валидационные метрики
        if hasattr(best_model, 'validation_metrics'):
            fold_results.update(best_model.validation_metrics)

        return fold_results

    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Вычисляет веса классов для сбалансированной функции потерь."""
        class_counts = torch.zeros(self.cfg.n_seg_classes, dtype=torch.float32)

        for batch in train_loader:
            _, labels = batch
            # Принудительно перемещаем на CPU и извлекаем значения
            labels = labels.cpu() if isinstance(labels, torch.Tensor) else labels
            for label in labels:
                label_val = int(label.item()) if hasattr(label, 'item') else int(label)
                class_counts[label_val] += 1

        # Инвертированные частоты
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.cfg.n_seg_classes * class_counts)

        # Нормализация
        class_weights = class_weights / class_weights.sum() * self.cfg.n_seg_classes

        # Возвращаем веса на CPU
        return class_weights.cpu()

    def run_cross_validation(self) -> Dict:
        """Запускает полную кросс-валидацию (или одиночный фолд при n_splits=1)."""

        console.print(Panel.fit(
            "[bold green]🏥 МЕДИЦИНСКАЯ КЛАССИФИКАЦИЯ - ОБУЧЕНИЕ 🏥[/bold green]",
            border_style="green"
        ))

        # Загружаем данные
        filenames, labels = self.load_data_labels()
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Информация о данных
        data_table = Table(title="📊 Информация о данных")
        data_table.add_column("Параметр", style="cyan")
        data_table.add_column("Значение", style="yellow")
        data_table.add_row("Всего образцов", str(len(filenames)))
        for label, count in zip(unique_labels, counts):
            data_table.add_row(f"Класс {label}", f"{count} ({count / len(labels) * 100:.1f}%)")
        console.print(data_table)

        all_results = []

        if self.cfg.n_splits == 1:
            # Один фолд = один train/val split
            train_idx, val_idx = train_test_split(
                np.arange(len(filenames)),
                test_size=self.cfg.val_size if hasattr(self.cfg, "val_size") else 0.2,
                stratify=labels,
                random_state=self.cfg.cv_random_state
            )
            folds = [(train_idx, val_idx)]
        else:
            skf = StratifiedKFold(
                n_splits=self.cfg.n_splits,
                shuffle=True,
                random_state=self.cfg.cv_random_state
            )
            folds = skf.split(filenames, labels)

        # Обучение по фолдам
        for fold, (train_indices, val_indices) in enumerate(folds):
            train_loader, val_loader = self.create_fold_datasets(
                train_indices, val_indices, filenames, labels
            )
            fold_results = self.train_fold(fold, train_loader, val_loader)
            all_results.append(fold_results)

            rprint(f"✅ [bold green]Фолд {fold + 1} завершен![/bold green]")
            rprint(f"   📈 Лучший результат: {fold_results['best_val_score']:.4f}")

        # Сводка (если фолдов несколько)
        if len(all_results) > 1:
            self.print_cv_summary(all_results)

        return {
            'fold_results': all_results,
            'cv_summary': self.calculate_cv_summary(all_results) if len(all_results) > 1 else all_results[0]
        }

    def calculate_cv_summary(self, results: List[Dict]) -> Dict:
        """Вычисляет сводную статистику по кросс-валидации."""
        scores = [r['best_val_score'] for r in results]

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores)
        }

    def print_cv_summary(self, results: List[Dict]):
        """Выводит красивую сводку результатов кросс-валидации."""

        summary_table = Table(title="🎯 РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ")
        summary_table.add_column("Фолд", style="cyan", justify="center")
        summary_table.add_column("Лучший результат", style="yellow", justify="center")
        summary_table.add_column("Финальная эпоха", style="blue", justify="center")

        scores = []
        for result in results:
            summary_table.add_row(
                str(result['fold']),
                f"{result['best_val_score']:.4f}",
                str(result['final_epoch'])
            )
            scores.append(result['best_val_score'])

        # Статистики
        summary_table.add_row("---", "---", "---", style="dim")
        summary_table.add_row(
            "СРЕДНЕЕ",
            f"{np.mean(scores):.4f} ± {np.std(scores):.4f}",
            "",
            style="bold green"
        )

        console.print(summary_table)

        # Панель с итоговой информацией
        summary_text = f"""
[bold green]📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:[/bold green]
• Средний результат: {np.mean(scores):.4f} ± {np.std(scores):.4f}
• Лучший фолд: {np.max(scores):.4f}
• Худший фолд: {np.min(scores):.4f}
• Медиана: {np.median(scores):.4f}
• Коэффициент вариации: {(np.std(scores) / np.mean(scores) * 100):.2f}%
        """

        console.print(Panel(summary_text, title="🏆 Финальный отчет", border_style="green"))


def setup_environment(cfg: OmegaConf) -> argparse.Namespace:
    """Настраивает окружение для обучения."""

    # Установка seed для воспроизводимости
    torch.manual_seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)

    # Создание директорий
    Path(cfg.save_folder).mkdir(parents=True, exist_ok=True)
    Path("tb_logs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Преобразование в namespace
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['gpu_id'] = [] if cfg.no_cuda else [0]
    cfg_dict['phase'] = 'train'
    cfg_dict['pin_memory'] = not cfg.no_cuda and torch.cuda.is_available()

    # Отключаем детерминизм для производительности
    torch.use_deterministic_algorithms(False)

    return argparse.Namespace(**cfg_dict)


def main():
    """Основная функция для запуска обучения."""

    try:
        # Загрузка конфига
        cfg = ModelConfig()
        cfg = OmegaConf.structured(cfg)

        # CLI параметры
        cli_cfg = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cli_cfg)

        # Настройки для CI тестирования
        if cfg.ci_test:
            cfg.img_list = '../toy_data/test_ci.txt'
            cfg.n_epochs = 2
            cfg.no_cuda = True
            cfg.data_root = '../toy_data'
            cfg.pretrain_path = ''
            cfg.num_workers = 0
            cfg.batch_size = 2
            cfg.n_splits = 2

        # Настройка окружения
        cfg_namespace = setup_environment(cfg)

        # Информация о системе
        device_info = "CPU" if cfg.no_cuda else f"GPU ({torch.cuda.get_device_name()})" if torch.cuda.is_available() else "CPU (CUDA недоступна)"

        system_table = Table(title="💻 Информация о системе")
        system_table.add_column("Параметр", style="cyan")
        system_table.add_column("Значение", style="yellow")

        system_table.add_row("Устройство", device_info)
        system_table.add_row("PyTorch версия", torch.__version__)
        system_table.add_row("CUDA доступна", str(torch.cuda.is_available()))
        system_table.add_row("Размер батча", str(cfg.batch_size))
        system_table.add_row("Learning rate", str(cfg.learning_rate))
        system_table.add_row("Количество фолдов", str(cfg.n_splits))
        system_table.add_row("Максимум эпох", str(cfg.n_epochs))

        console.print(system_table)

        # Запуск кросс-валидации
        cv_trainer = CrossValidationTrainer(cfg, cfg_namespace)
        results = cv_trainer.run_cross_validation()

        rprint("\n🎉 [bold green]Обучение успешно завершено![/bold green]")

        return results

    except Exception as e:
        rprint(f"\n❌ [bold red]Ошибка во время обучения:[/bold red] {str(e)}")
        console.print_exception(show_locals=True)
        return None

if __name__ == '__main__':
    main()
        