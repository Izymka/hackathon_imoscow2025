# config.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    # Model architecture
    model: str = "resnet"
    model_depth: int = 34
    resnet_shortcut: str = "A"

    # Input dimensions
    input_W: int = 128
    input_H: int = 128
    input_D: int = 128

    # Classes
    n_seg_classes: int = 2  # для бинарной классификации остается 2

    # Training parameters
    batch_size: int = 8
    learning_rate: float = 0.0001
    n_epochs: int = 150
    save_intervals: int = 100
    num_workers: int = 8

    # Early stopping parameters
    early_stopping_patience: int = 30  # эпох без улучшения до остановки
    early_stopping_min_delta: float = 0.0005  # минимальное улучшение для продолжения
    early_stopping_metric: str = "val_f1"  # метрика для мониторинга - f1_macro

    # Learning rate scheduler
    lr_scheduler_patience: int = 5  # эпох без улучшения до уменьшения LR
    lr_scheduler_factor: float = 0.5  # множитель уменьшения LR
    lr_scheduler_min_lr: float = 1e-7  # минимальный learning rate

    # Checkpoint parameters
    save_top_k: int = 5  # количество лучших чекпоинтов для сохранения
    monitor_metric: str = "val_f1"  # метрика для мониторинга чекпоинтов - f1_macro
    checkpoint_mode: str = "max"  # "max" для метрик, которые нужно максимизировать

    # Paths
    data_root: str = "data/train"
    img_list: str = "data/train/labels.csv"
    pretrain_path: str = "model/pretrain/resnet_34_23dataset.pth"
    save_folder: str = "model/outputs/checkpoints"
    val_list: str = "data/test/labels.csv"
    val_data_root: str = "data/test"  # может совпадать с data_root

    # Model layers
    new_layer_names: List[str] = None
    
    # Fine-tuning parameters
    base_lr_multiplier: float = 1.0  # множитель для базового learning rate
    new_layers_lr_multiplier: float = 10.0  # множитель для новых/размороженных слоев

    # System settings
    no_cuda: bool = False
    manual_seed: int = 1
    ci_test: bool = False
    pin_memory: bool = True

    # Training stability
    gradient_clip_val: float = 1.0  # значение для градиентного обрезания
    deterministic: bool = False  # для воспроизводимости

    # Cross-validation parameters
    n_splits: int = 5  # количество фолдов для кросс-валидации
    cv_random_state: int = 42  # для воспроизводимости в кросс-валидации

    # Binary classification specific parameters
    binary_classification: bool = True  # флаг для бинарной классификации
    use_f1_macro: bool = True  # использовать f1_macro вместо f1_micro
    additional_metrics: List[str] = None  # дополнительные метрики для логгирования

    # Validation metrics for binary classification
    primary_metric: str = "val_f1"  # основная метрика для мониторинга
    secondary_metrics: List[str] = None  # дополнительные метрики для отслеживания

    def __post_init__(self):
        if self.new_layer_names is None:
            # Размораживаем последний блок ResNet (layer4) и полносвязный слой
            self.new_layer_names = ["fc", "layer4"]

        # ... existing code ...