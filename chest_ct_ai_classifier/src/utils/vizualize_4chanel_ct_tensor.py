import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Настройки
CHANNEL_NAMES = ['Lung', 'Mediastinal', 'Bone', 'Soft Tissue']
TENSOR_PATH = "path/to/your/patient.pt"  # замените на свой путь
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Загрузка тензора
tensor = torch.load(TENSOR_PATH)  # форма: [1, 4, 128, 128, 128]
assert tensor.shape == (1, 4, 128, 128, 128), f"Unexpected shape: {tensor.shape}"

# Преобразуем [-1, 1] → [0, 1] для визуализации
img_4d = (tensor.squeeze(0).cpu().numpy() + 1) / 2.0  # [4, 128, 128, 128]

# Выберем центральные срезы по каждой оси
center_d = img_4d.shape[1] // 2
center_h = img_4d.shape[2] // 2
center_w = img_4d.shape[3] // 2

# Визуализация: по одному срезу (например, по оси D — аксиальные срезы)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ch in range(4):
    # Аксиальный срез (D — глубина, H×W — изображение)
    slice_img = img_4d[ch, center_d, :, :]
    axes[ch].imshow(slice_img, cmap='gray')
    axes[ch].set_title(f"{CHANNEL_NAMES[ch]} (axial, D={center_d})")
    axes[ch].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{Path(TENSOR_PATH).stem}_axial.png", dpi=150)
plt.close()

# Опционально: сагиттальные и корональные срезы
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for ch in range(4):
    slice_img = img_4d[ch, :, center_h, :]  # сагиттальный
    axes[ch].imshow(slice_img, cmap='gray')
    axes[ch].set_title(f"{CHANNEL_NAMES[ch]} (sagittal, H={center_h})")
    axes[ch].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{Path(TENSOR_PATH).stem}_sagittal.png", dpi=150)
plt.close()