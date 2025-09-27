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
from sklearn.model_selection import StratifiedKFold
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
from inference import MedicalModelInference

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
            class_weights=class_weights  # Lightning автоматически переместит на правильное устройство
        )

        # Логгеры для фолда
        fold_name = f"medical_classification_fold_{fold + 1}"
        tb_logger = TensorBoardLogger("tb_logs", name=fold_name, version=f"fold_{fold + 1}")
        csv_logger = CSVLogger("logs", name=fold_name, version=f"fold_{fold + 1}")

        # Коллбэки
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.cfg.save_folder}/fold_{fold + 1}",
            filename="best-{epoch:02d}-{val_f1:.4f}-{val_auroc:.4f}",
            save_top_k=2,
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
            fast_dev_run=self.cfg.ci_test,
            log_every_n_steps=min(10, len(train_loader) // 4),
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=self.cfg.gradient_clip_val,
            precision=16 if accelerator == "gpu" else 32,  # mixed precision
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
        class_counts = torch.zeros(self.cfg.n_seg_classes)

        for batch in train_loader:
            _, labels = batch
            for label in labels:
                class_counts[int(label.item())] += 1

        # Инвертированные частоты
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.cfg.n_seg_classes * class_counts)

        # Нормализация
        class_weights = class_weights / class_weights.sum() * self.cfg.n_seg_classes

        return class_weights

    def run_cross_validation(self) -> Dict:
        """Запускает полную кросс-валидацию."""

        # Красивый заголовок
        console.print(Panel.fit(
            "[bold green]🏥 МЕДИЦИНСКАЯ КЛАССИФИКАЦИЯ - КРОСС-ВАЛИДАЦИЯ 🏥[/bold green]",
            border_style="green"
        ))

        # Загружаем данные
        filenames, labels = self.load_data_labels()

        # Информация о данных
        unique_labels, counts = np.unique(labels, return_counts=True)

        data_table = Table(title="📊 Информация о данных")
        data_table.add_column("Параметр", style="cyan")
        data_table.add_column("Значение", style="yellow")

        data_table.add_row("Всего образцов", str(len(filenames)))
        for label, count in zip(unique_labels, counts):
            data_table.add_row(f"Класс {label}", f"{count} ({count / len(labels) * 100:.1f}%)")

        console.print(data_table)

        # Стратифицированная кросс-валидация
        skf = StratifiedKFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.cv_random_state
        )

        all_results = []

        # Обучение по фолдам
        for fold, (train_indices, val_indices) in enumerate(skf.split(filenames, labels)):
            # Создаем датасеты для фолда
            train_loader, val_loader = self.create_fold_datasets(
                train_indices, val_indices, filenames, labels
            )

            # Информация о фолде
            train_labels = [labels[i] for i in train_indices]
            val_labels = [labels[i] for i in val_indices]

            fold_table = Table(title=f"📋 Фолд {fold + 1}")
            fold_table.add_column("Набор", style="cyan")
            fold_table.add_column("Размер", style="yellow")
            fold_table.add_column("Класс 0", style="red")
            fold_table.add_column("Класс 1", style="green")

            train_counts = np.bincount(train_labels)
            val_counts = np.bincount(val_labels)

            fold_table.add_row(
                "Обучение",
                str(len(train_indices)),
                f"{train_counts[0]} ({train_counts[0] / len(train_indices) * 100:.1f}%)",
                f"{train_counts[1]} ({train_counts[1] / len(train_indices) * 100:.1f}%)"
            )
            fold_table.add_row(
                "Валидация",
                str(len(val_indices)),
                f"{val_counts[0]} ({val_counts[0] / len(val_indices) * 100:.1f}%)",
                f"{val_counts[1]} ({val_counts[1] / len(val_indices) * 100:.1f}%)"
            )

            console.print(fold_table)

            # Обучение фолда
            fold_results = self.train_fold(fold, train_loader, val_loader)
            all_results.append(fold_results)

            # Промежуточные результаты
            rprint(f"✅ [bold green]Фолд {fold + 1} завершен![/bold green]")
            rprint(f"   📈 Лучший результат: {fold_results['best_val_score']:.4f}")

        # Сводка по всем фолдам
        self.print_cv_summary(all_results)

        return {
            'fold_results': all_results,
            'cv_summary': self.calculate_cv_summary(all_results)
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


def adapt_model_for_input_size(model, input_size, model_depth, n_seg_classes):
    """
    Адаптирует модель для нового размера входа путем замены последнего слоя.
    
    Args:
        model: Загруженная модель
        input_size: Новый размер входа (W, H, D)
        model_depth: Глубина модели ResNet
        n_seg_classes: Количество классов
    
    Returns:
        model: Адаптированная модель
        trainable_parameters: Параметры для обучения
    """
    import torch
    import torch.nn as nn
    
    print(f"Адаптация модели для входа размером {input_size}...")
    
    # --- Заморозка всех параметров ---
    print("Замораживание всех параметров...")
    for param in model.parameters():
        param.requires_grad = False
    
    # --- Вычисление нового размера полносвязного слоя ---
    # Создаем фиктивный тензор для вычисления размера после сверток
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, input_size[2], input_size[1], input_size[0])
        
        # Извлекаем только сверточную часть модели
        if hasattr(model, 'module'):
            # Если модель обернута в DataParallel
            conv_features = nn.Sequential(
                model.module.conv1,
                model.module.bn1,
                model.module.relu,
                model.module.maxpool,
                model.module.layer1,
                model.module.layer2,
                model.module.layer3,
                model.module.layer4,
                model.module.avgpool
            )
        else:
            conv_features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool
            )
        
        # Вычисляем размер после сверточных слоев
        conv_output = conv_features(dummy_input)
        flattened_size = conv_output.view(conv_output.size(0), -1).size(1)
    
    print(f"Новый размер входа FC слоя: {flattened_size}")
    
    # --- Замена полносвязного слоя ---
    if hasattr(model, 'module'):
        # DataParallel случай
        old_fc = model.module.fc
        model.module.fc = nn.Linear(flattened_size, n_seg_classes)
        new_fc = model.module.fc
    else:
        old_fc = model.fc
        model.fc = nn.Linear(flattened_size, n_seg_classes)
        new_fc = model.fc
    
    print(f"Заменен FC слой: {old_fc.in_features} → {flattened_size} входов, {n_seg_classes} выходов")
    
    # --- Инициализация нового слоя ---
    if isinstance(new_fc, nn.Linear):
        # Xavier инициализация
        nn.init.xavier_uniform_(new_fc.weight)
        if new_fc.bias is not None:
            nn.init.zeros_(new_fc.bias)
    
    # --- Размораживание только FC слоя ---
    print("Размораживание FC слоя для обучения...")
    for param in new_fc.parameters():
        param.requires_grad = True
    
    # --- Возврат обучаемых параметров ---
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    return model, trainable_parameters


def test_inference_example():
    """Пример тестирования inference."""
    rprint("\n🔬 [bold blue]Тестирование inference модуля...[/bold blue]")

    try:
        # Тестовый тензор
        test_tensor = torch.randn(1, 1, 128, 128, 128)
        rprint(f"📊 Тестовый тензор: {test_tensor.shape}")

        # Создание inference объекта
        inference = MedicalModelInference(
            weights_path="model/outputs/checkpoints/best_weights.pth",
            model_config=ModelConfig()
        )

        # Предсказание
        prediction = inference.predict(test_tensor)
        rprint(f"🎯 Результат предсказания: {prediction}")

        # Батчевое предсказание
        batch_tensor = torch.randn(3, 1, 128, 128, 128)
        batch_predictions = inference.predict_batch(batch_tensor)
        rprint(f"📦 Пакетные предсказания: {batch_predictions}")

        rprint("✅ [bold green]Inference тестирование завершено успешно![/bold green]")

    except Exception as e:
        rprint(f"❌ [bold red]Ошибка inference тестирования:[/bold red] {str(e)}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--test-inference":
        test_inference_example()
    else:
        main()