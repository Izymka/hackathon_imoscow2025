#!/usr/bin/env python3
"""
compare_before_after.py - Сравнение оригинального и обработанного изображения
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from monai.data import MetaTensor


def load_original_dicom(dicom_folder):
    """Загрузка оригинального DICOM без обработки"""
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
    series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
    reader.SetFileNames(series_file_names)
    sitk_image = reader.Execute()
    image_array = sitk.GetArrayFromImage(sitk_image)
    return image_array


def compare_volumes(original_path, processed_path, slice_idx=None):
    """
    Сравнение оригинального и обработанного объема

    Args:
        original_path: путь к папке с оригинальными DICOM
        processed_path: путь к .pt файлу
        slice_idx: индекс среза для сравнения (если None - середина)
    """

    # Загружаем оригинальный объем
    original_volume = load_original_dicom(original_path)

    # Загружаем обработанный тензор
    processed_tensor = torch.load(processed_path)
    if processed_tensor.dim() == 5:
        processed_tensor = processed_tensor.squeeze(0).squeeze(0)
    processed_volume = processed_tensor.numpy()

    print(f"Original shape: {original_volume.shape}")
    print(f"Processed shape: {processed_volume.shape}")

    # Выбираем срез
    if slice_idx is None:
        slice_idx_orig = original_volume.shape[0] // 2
        slice_idx_proc = processed_volume.shape[0] // 2
    else:
        slice_idx_orig = min(slice_idx, original_volume.shape[0] - 1)
        slice_idx_proc = min(slice_idx, processed_volume.shape[0] - 1)

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Оригинальный срез
    orig_slice = original_volume[slice_idx_orig]
    axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title(f'Original DICOM\nSlice {slice_idx_orig}\nShape: {orig_slice.shape}')
    axes[0].axis('off')

    # Обработанный срез
    proc_slice = processed_volume[slice_idx_proc]
    axes[1].imshow(proc_slice, cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title(f'Processed Tensor\nSlice {slice_idx_proc}\nShape: {proc_slice.shape}')
    axes[1].axis('off')

    plt.suptitle('Comparison: Original vs Processed')
    plt.tight_layout()
    plt.show()

# Использование
# compare_volumes("path/to/original/dicom/folder", "path/to/processed.pt")