# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ModelConfig:
    # ========== MODEL ARCHITECTURE ==========
    model: str = "resnet"
    model_depth: int = 34
    resnet_shortcut: str = "A"

    # Spatial Attention
    use_spatial_attention: bool = False  # включение/выключение spatial attention encoder

    # ========== INPUT DIMENSIONS ==========
    input_W: int = 256
    input_H: int = 256
    input_D: int = 256

    # ========== CLASSIFICATION PARAMETERS ==========
    n_seg_classes: int = 2  # бинарная классификация
    binary_classification: bool = True

    # ========== TRAINING HYPERPARAMETERS ==========
    batch_size: int = 3  # Физический batch size для 256³ тензоров на 8GB VRAM
    accumulate_grad_batches: int = 4  # Эффективный batch size = 3 * 4 = 12
    learning_rate: float = 5e-5
    n_epochs: int = 75  #
    num_workers: int = 6
    optimizer: str = "adamw"
    momentum: float = 0.9

    # ========== ADVANCED TRAINING PARAMETERS ==========
    # Оптимизация
    weight_decay: float = 5e-4
    gradient_clip_val: float = 1.0

    # Learning Rate Scheduler
    lr_scheduler: str = "plateau"  # "plateau", "cosine", "none"
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.3
    lr_scheduler_min_lr: float = 1e-7

    # ========== LOSS FUNCTION PARAMETERS ==========
    # Выбор функции потерь
    use_focal_loss: bool = True  # True для сложных несбалансированных случаев
    focal_alpha: float = 0.7
    focal_gamma: float = 3.0

    # Weighted Loss для несбалансированных классов
    use_weighted_loss: bool = True
    auto_class_weights: bool = True  # автоматический расчет весов

    # ========== EARLY STOPPING ==========
    early_stopping_patience: int = 15  # увеличено для медицинских данных
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "val_auroc"

    # ========== CHECKPOINTING ==========
    save_intervals: int = 10
    save_top_k: int = 3
    monitor_metric: str = "val_auroc"
    checkpoint_mode: str = "max"
    save_weights_only: bool = False  # сохраняем полные чекпоинты для возобновления
    resume_path: Optional[str] = None

    # ========== CROSS-VALIDATION ==========
    use_cross_validation: bool = True
    n_splits: int = 3
    cv_random_state: int = 42
    stratified_cv: bool = True  # стратифицированная кросс-валидация

    # ========== DATA PATHS ==========
    data_root: str = "data/train"
    img_list: str = "data/train/labels.csv"
    val_data_root: str = "data/test"
    val_list: str = "data/test/labels.csv"

    # Предобученная модель
    pretrain_path: str = "model/pretrain/best-epoch=17-val_f1=0.8400.pth"
    use_pretrained: bool = True

    # Выходные директории
    save_folder: str = "model/outputs/checkpoints"
    log_folder: str = "logs"
    tb_log_folder: str = "tb_logs"

    # ========== DATA AUGMENTATION ==========
    # Пространственные аугментации
    aug_flip_prob: float = 0.3
    aug_rotate_prob: float = 0.2

    # Интенсивностные аугментации
    aug_noise_prob: float = 0.15
    aug_noise_std: float = 0.005
    aug_intensity_shift_prob: float = 0.2
    aug_intensity_shift_offset: float = 0.05
    aug_contrast_prob: float = 0.2
    aug_contrast_gamma: Tuple[float, float] = (0.9, 1.1)
    aug_scale_intensity_prob: float = 0.2
    aug_scale_intensity_factors: Tuple[float, float] = (-0.05, 0.05)

    # ========== MODEL FINE-TUNING ==========
    # Слои для размораживания/обучения
    new_layer_names: List[str] = field(default_factory=lambda: ["fc", "layer4", "layer3"])

    # Дифференцированные learning rates
    use_differential_lr: bool = True
    base_lr_multiplier: float = 0.1  # для предобученных слоев
    new_layers_lr_multiplier: float = 1.0  # для новых слоев

    # ========== METRICS AND LOGGING ==========
    # Основные метрики
    primary_metric: str = "val_auroc"
    log_every_n_steps: int = 10

    # Дополнительные метрики для отслеживания
    track_additional_metrics: bool = True
    additional_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "specificity"
    ])

    # Частота детального логирования
    detailed_logging_frequency: int = 10  # каждые N эпох

    # ========== MEDICAL-SPECIFIC PARAMETERS ==========
    # Медицинские метрики
    calculate_sensitivity: bool = True  # recall для медицины
    calculate_ppv: bool = True  # положительная прогностическая ценность
    calculate_npv: bool = True  # отрицательная прогностическая ценность

    # Пороги для классификации
    classification_threshold: float = 0.5
    optimize_threshold: bool = True  # оптимизация порога по валидации

    # ========== SYSTEM SETTINGS ==========
    no_cuda: bool = False
    manual_seed: int = 42
    deterministic: bool = False
    pin_memory: bool = True

    # Производительность
    mixed_precision: bool = True  # 16-bit training
    compile_model: bool = os.getenv('MODEL_HP_COMPILE_MODEL') or False  # PyTorch 2.0 compilation

    # ========== TESTING AND DEBUGGING ==========
    ci_test: bool = False
    fast_dev_run: bool = False
    overfit_batches: float = 0.0  # для отладки

    # Профилирование
    profiler: Optional[str] = None  # "simple", "advanced", "pytorch"

    # ========== INFERENCE PARAMETERS ==========
    # Test-Time Augmentation
    use_tta: bool = False
    tta_transforms: int = 8

    # Ensemble параметры
    use_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Постобработка конфига после инициализации."""

        # Проверка совместимости параметров
        if self.n_seg_classes != 2 and self.binary_classification:
            self.binary_classification = False
            print("⚠️  Предупреждение: binary_classification установлен в False из-за n_seg_classes != 2")

        # Предупреждение о размере входа
        if (self.input_W, self.input_H, self.input_D) != (128, 128, 128):
            print(f"⚠️  Внимание: Размер входа изменен на {self.input_W}×{self.input_H}×{self.input_D}")
            print("    Может потребоваться пересчет размера полносвязного слоя")

        # Автоматическая настройка метрик для бинарной классификации
        if self.binary_classification:
            if "auroc" not in self.additional_metrics:
                self.additional_metrics.append("auroc")

        # Проверка путей
        from pathlib import Path

        # Создание необходимых директорий
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.log_folder).mkdir(exist_ok=True)
        Path(self.tb_log_folder).mkdir(exist_ok=True)

        # Настройки для CI тестирования
        if self.ci_test:
            self.n_epochs = 2
            self.batch_size = 2
            self.num_workers = 0
            self.n_splits = 2
            self.early_stopping_patience = 3
            self.save_top_k = 1
            self.fast_dev_run = True

        # Проверка корректности scheduler
        valid_schedulers = ["plateau", "cosine", "none"]
        if self.lr_scheduler not in valid_schedulers:
            print(f"⚠️  Предупреждение: Неизвестный scheduler '{self.lr_scheduler}'. Используется 'plateau'.")
            self.lr_scheduler = "plateau"

        # Настройка focal loss для медицинских данных
        if self.use_focal_loss and self.binary_classification:
            # Для медицинских данных часто gamma=2, alpha зависит от дисбаланса
            if not hasattr(self, '_focal_configured'):
                print("ℹ️  Используется Focal Loss для борьбы с дисбалансом классов")
                self._focal_configured = True

    def get_augmentation_summary(self) -> str:
        """Возвращает сводку по аугментациям."""
        aug_summary = f"""
        🔄 Конфигурация аугментаций:
        • Flip: {self.aug_flip_prob:.2f}
        • Rotate: {self.aug_rotate_prob:.2f}  
        • Noise: {self.aug_noise_prob:.2f} (std={self.aug_noise_std:.3f})
        • Intensity shift: {self.aug_intensity_shift_prob:.2f}
        • Contrast: {self.aug_contrast_prob:.2f}
        • Scale intensity: {self.aug_scale_intensity_prob:.2f}
        """
        return aug_summary

    def get_training_summary(self) -> str:
        """Возвращает сводку по обучению."""
        loss_type = "Focal Loss" if self.use_focal_loss else "Cross Entropy"
        if self.use_weighted_loss:
            loss_type += " (Weighted)"

        # Вычисляем эффективный batch size
        effective_batch_size = self.batch_size * getattr(self, 'accumulate_grad_batches', 1)

        training_summary = f"""
        🎯 Конфигурация обучения:
        • Batch size: {self.batch_size} (физический)
        • Accumulate grad batches: {getattr(self, 'accumulate_grad_batches', 1)}
        • Effective batch size: {effective_batch_size}
        • Learning rate: {self.learning_rate}
        • Max epochs: {self.n_epochs}
        • Loss function: {loss_type}
        • LR Scheduler: {self.lr_scheduler}
        • Early stopping: {self.early_stopping_patience} epochs
        • Cross-validation: {self.n_splits} folds
        • Primary metric: {self.primary_metric}
        """
        return training_summary

    def validate_config(self) -> List[str]:
        """Проверяет конфигурацию и возвращает список предупреждений."""
        warnings = []

        # Проверка размеров с учетом gradient accumulation
        effective_batch_size = self.batch_size * getattr(self, 'accumulate_grad_batches', 1)
        if effective_batch_size < 3:
            warnings.append(f"Слишком маленький эффективный batch_size ({effective_batch_size}) может негативно влиять на BatchNorm")

        # Проверка learning rate
        if self.learning_rate > 1e-2:
            warnings.append("Высокий learning_rate может привести к нестабильности")

        if self.learning_rate < 1e-6:
            warnings.append("Слишком низкий learning_rate может замедлить обучение")

        # Проверка early stopping
        if self.early_stopping_patience < 5:
            warnings.append("Низкий early_stopping_patience может привести к преждевременной остановке")

        # Проверка файлов
        from pathlib import Path
        if not Path(self.img_list).exists():
            warnings.append(f"Файл меток не найден: {self.img_list}")

        if self.use_pretrained and self.pretrain_path and not Path(self.pretrain_path).exists():
            warnings.append(f"Предобученная модель не найдена: {self.pretrain_path}")

        return warnings
