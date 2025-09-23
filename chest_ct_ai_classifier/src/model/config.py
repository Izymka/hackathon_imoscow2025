# config.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    # Model architecture
    model: str = "resnet"
    model_depth: int = 10
    resnet_shortcut: str = "B"

    # Input dimensions
    input_W: int = 128
    input_H: int = 128
    input_D: int = 128

    # Classes
    n_seg_classes: int = 2

    # Training parameters
    batch_size: int = 2
    learning_rate: float = 0.0001
    n_epochs: int = 50
    save_intervals: int = 100
    num_workers: int = 4

    # Paths
    data_root: str = "data/train/tensors"
    img_list: str = "data/train/labels.csv"
    pretrain_path: str = "model/pretrain/resnet_10_23dataset.pth"
    save_folder: str = "model/outputs/checkpoints"
    val_list: str = "data/train/val_labels.csv"
    val_data_root: str = "data/train/tensors"  # может совпадать с data_root

    # Model layers
    new_layer_names: List[str] = None

    # System settings
    no_cuda: bool = False
    manual_seed: int = 1
    ci_test: bool = False
    pin_memory: bool = True

    def __post_init__(self):
        if self.new_layer_names is None:
            self.new_layer_names = ["fc"]