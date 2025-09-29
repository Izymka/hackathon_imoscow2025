#!/usr/bin/env python3
"""
prepare_ct_nii_multichannel.py

Подготовка КТ-данных из NIfTI (.nii / .nii.gz) для MedicalNet с 4-канальным оконным представлением:
- Лёгочное окно
- Медиастинальное окно
- Костное окно
- Окно мягких тканей

Возвращает тензор [1, 4, 128, 128, 128], где каждый канал — результат оконной фильтрации.
Без ресемплинга — используется адаптивный пуллинг.
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from monai.data import MetaTensor
from monai.transforms import (
    EnsureChannelFirst,
    CropForeground,
    ToTensor,
    Compose,
    Lambda
)
import traceback
from tqdm import tqdm
import json
import torch.nn.functional as F
import gc

# -------------------------------
# Конфигурация
# -------------------------------
TARGET_OUTPUT_SHAPE = (128, 128, 128)

WINDOW_PRESETS = {
    'lung': {'center': -600, 'width': 1200},
    'mediastinal': {'center': 50, 'width': 400},
    'bone': {'center': 300, 'width': 1500},
    'soft_tissue': {'center': 40, 'width': 80}
}


# -------------------------------
# Оконная фильтрация
# -------------------------------
def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Применяет оконную фильтрацию к изображению в HU.
    Возвращает нормализованное изображение в диапазоне [0, 1].
    """
    min_val = center - width // 2
    max_val = center + width // 2
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val + 1e-8)
    return windowed.astype(np.float32)


def apply_all_windows(image: np.ndarray) -> np.ndarray:
    """
    Применяет 4 предустановленных окна и возвращает тензор [4, D, H, W]
    """
    channels = []
    for preset in WINDOW_PRESETS.values():
        ch = apply_window(image, preset['center'], preset['width'])
        channels.append(ch)
    return np.stack(channels, axis=0)  # [4, D, H, W]


# -------------------------------
# Трансформации MONAI
# -------------------------------
def apply_windows_and_pool(x):
    # Преобразование к numpy если нужно
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Убедиться, что это [D, H, W]
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    # Применить окна
    x_windows = apply_all_windows(x)  # [4, D, H, W]

    # Преобразовать в тензор и добавить batch dimension
    x_tensor = torch.from_numpy(x_windows).unsqueeze(0)  # [1, 4, D, H, W]

    # Адаптивный пулинг
    x_tensor = F.adaptive_avg_pool3d(x_tensor, TARGET_OUTPUT_SHAPE)  # [1, 4, 128, 128, 128]

    # Нормализация [-1, 1]
    x_tensor = x_tensor * 2.0 - 1.0

    return x_tensor


def get_transforms(target_output_shape=TARGET_OUTPUT_SHAPE):
    """Трансформации без процентильной нормализации — только оконная фильтрация и пулинг"""
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),  # [D, H, W] -> [1, D, H, W]
        CropForeground(select_fn=lambda x: x > -1000, channel_indices=0, margin=5),
        Lambda(apply_windows_and_pool),
        ToTensor()
    ])


# -------------------------------
# Валидация размеров
# -------------------------------
def validate_tensor_size(tensor, min_size=16):
    if isinstance(tensor, MetaTensor):
        shape = tensor.shape
    else:
        shape = tensor.shape

    if len(shape) == 3:
        d, h, w = shape
    elif len(shape) == 4 and shape[0] == 1:
        d, h, w = shape[1], shape[2], shape[3]
    else:
        raise ValueError(f"Unexpected tensor shape: {shape}")

    if min(d, h, w) < min_size:
        raise ValueError(f"Tensor too small: {(d, h, w)}, minimum size: {min_size}")
    return tensor


# -------------------------------
# Загрузка NIfTI с корректной ориентацией
# -------------------------------
def load_nii_file(nii_path: Path) -> MetaTensor:
    """
    Загружает NIfTI-файл и возвращает MetaTensor с метаданными.
    Коррекция: пациент лежит на потолке → переворачиваем каждый срез по вертикали.
    """
    try:
        nii_img = nib.load(str(nii_path))
        data = nii_img.get_fdata().astype(np.float32)
        affine = nii_img.affine

        # Определяем ориентацию и корректируем оси
        img_oriented = nib.as_closest_canonical(nii_img)
        data_oriented = img_oriented.get_fdata().astype(np.float32)
        affine_oriented = img_oriented.affine

        if data_oriented.ndim == 3:
            # (x, y, z) -> (z, y, x) для MedicalNet (D, H, W)
            data_oriented = np.transpose(data_oriented, (2, 1, 0))  # (z, y, x)

            # ИСПРАВЛЕНИЕ: Пациент "лежит на потолке" → переворачиваем каждый срез по вертикали
            # Ось Y в MedicalNet - это вторая ось (индекс 1) в (D, H, W)
            # После транспонирования (z, y, x), ось Y - это вторая ось (axis=1)
            data_oriented = np.flip(data_oriented, axis=1)  # отражаем по оси H (высота)

        elif data_oriented.ndim == 4:
            if data_oriented.shape[-1] in (1, 3):
                data_oriented = data_oriented[..., 0]
                data_oriented = np.transpose(data_oriented, (2, 1, 0))
                data_oriented = np.flip(data_oriented, axis=1)
            else:
                data_oriented = data_oriented[:, :, :, 0]
                data_oriented = np.transpose(data_oriented, (2, 1, 0))
                data_oriented = np.flip(data_oriented, axis=1)
        else:
            raise ValueError(f"Unsupported dimension: {data_oriented.ndim}")

        # ИСПРАВЛЕНИЕ: Создаем копию массива, чтобы избежать отрицательных стридов
        data_oriented = np.ascontiguousarray(data_oriented.copy())

        spacing = np.sqrt(np.sum(affine_oriented[:3, :3] ** 2, axis=0))

        meta_dict = {
            'spacing': spacing.tolist(),
            'original_shape': data.shape,
            'oriented_shape': data_oriented.shape,
            'original_affine': affine,
            'oriented_affine': affine_oriented,
            'filename_or_obj': str(nii_path),
            'source': 'nifti',
            'hu_range': (float(data_oriented.min()), float(data_oriented.max())),
        }
        return MetaTensor(data_oriented, meta=meta_dict)

    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI {nii_path}: {e}")


# -------------------------------
# Настройка логирования
# -------------------------------
def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# -------------------------------
# Обработка одного пациента
# -------------------------------
def process_single_nii(nii_path: Path, output_dir: Path, verbose: bool = False) -> dict:
    result = {
        'file_name': nii_path.name,
        'success': False,
        'error': None,
        'output_path': None,
        'original_shape': None,
        'final_shape': None,
        'original_spacing': None,
        'hu_range_input': None,
        'value_range_output': None,
        'mean': None,
        'std': None
    }

    try:
        # Очистка памяти перед обработкой
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        ct_meta_tensor = load_nii_file(nii_path)
        original_shape = ct_meta_tensor.shape
        result.update({
            'original_shape': original_shape,
            'original_spacing': ct_meta_tensor.meta.get('original_spacing'),
            'hu_range_input': ct_meta_tensor.meta.get('hu_range')
        })

        if verbose:
            logging.info(f"Loaded {nii_path.name}: {original_shape}")

        validated_tensor = validate_tensor_size(ct_meta_tensor)
        transform = get_transforms()
        final_tensor = transform(validated_tensor)  # [1, 4, 128, 128, 128]

        expected_shape = (1, 4) + TARGET_OUTPUT_SHAPE
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        if torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any():
            raise ValueError("Invalid values (NaN/Inf) in final tensor")

        output_path = output_dir / f"{nii_path.stem}.pt"
        torch.save(final_tensor, output_path)

        result.update({
            'success': True,
            'output_path': str(output_path),
            'final_shape': tuple(final_tensor.shape),
            'value_range_output': (float(final_tensor.min()), float(final_tensor.max())),
            'mean': float(final_tensor.mean()),
            'std': float(final_tensor.std())
        })

        if verbose:
            logging.info(f"✓ {nii_path.name}: "
                         f"range=[{result['value_range_output'][0]:.3f}, {result['value_range_output'][1]:.3f}], "
                         f"mean={result['mean']:.3f}, std={result['std']:.3f}")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            logging.error(f"❌ {nii_path.name}: {e}")
            logging.debug(traceback.format_exc())

    finally:
        # Очистка памяти после обработки
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return result


# -------------------------------
# Главная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CT NIfTI volumes for MedicalNet with 4-channel windowing")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with .nii or .nii.gz files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_nii_multichannel.log",
                        help="Log file path")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file)
    logger = setup_logging(log_file)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Поиск NIfTI-файлов
    nii_files = [
        f for f in input_dir.rglob("*")
        if f.is_file() and (f.suffix == '.nii' or f.suffixes[-1] == '.gz' and f.stem.endswith('.nii'))
    ]

    if not nii_files:
        logger.error(f"No .nii or .nii.gz files found in {input_dir}")
        return

    logger.info(f"Found {len(nii_files)} NIfTI file(s)")
    logger.info(f"Target output tensor shape: {(1, 4) + TARGET_OUTPUT_SHAPE}")
    logger.info("✅ Using 4-channel windowing: lung, mediastinal, bone, soft_tissue")
    logger.info("✅ Orientation corrected (canonical + flipped)")
    logger.info("✅ Output intensity range: [-1, 1]")
    logger.info("✅ No resampling — adaptive pooling used")

    results_summary = {
        'total': len(nii_files),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'resampling': 'none',
            'window_presets': WINDOW_PRESETS,
            'output_intensity_range': [-1.0, 1.0]
        }
    }

    logger.info("Starting processing...")
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files"):
        result = process_single_nii(nii_file, output_dir, args.verbose)
        if result['success']:
            results_summary['successful'] += 1
        else:
            results_summary['failed'] += 1
            results_summary['errors'].append({
                'file': result['file_name'],
                'error': result['error']
            })

    report_file = output_dir / "nii_processing_report_multichannel.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("PROCESSING COMPLETED")
    logger.info(f"Total files: {results_summary['total']}")
    logger.info(f"Successfully processed: {results_summary['successful']}")
    logger.info(f"Failed: {results_summary['failed']}")
    logger.info(f"Success rate: {100 * results_summary['successful'] / results_summary['total']:.1f}%")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"{'=' * 60}")

    if results_summary['failed'] > 0:
        logger.warning(f"⚠️  {results_summary['failed']} files failed. Check the report for details.")

    logger.info("Done! ✅")


if __name__ == "__main__":
    main()