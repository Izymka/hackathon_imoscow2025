#!/usr/bin/env python3
"""
prepare_ct_nii_for_medicalnet.py

Подготовка КТ-данных из NIfTI (.nii / .nii.gz) для модели MedicalNet.
Сохраняет тензоры [1, 1, 256, 256, 256] с адаптивной нормализацией по ху с использованием процентилей.
Без ресемплинга — используется адаптивный пуллинг вместо изменения разрешения.
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
    ScaleIntensityRange,
    CropForeground,
    NormalizeIntensity,
    ToTensor,
    Compose,
    Lambda
)
import traceback
from tqdm import tqdm
import json
import torch.nn.functional as F

# -------------------------------
# Конфигурация
# -------------------------------
TARGET_OUTPUT_SHAPE = (256, 256, 256)
HU_MIN = -1000
HU_MAX = 1000
PERCENTILE_MIN = 1  # 1-й процентиль
PERCENTILE_MAX = 99  # 99-й процентиль


# -------------------------------
# Логирование
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
# Адаптивная нормализация по процентилям
# -------------------------------
def adaptive_hu_normalization_by_percentiles(data, percentile_min=PERCENTILE_MIN, percentile_max=PERCENTILE_MAX):
    """
    Адаптивная нормализация данных КТ по ху с использованием процентилей.

    Args:
        data: numpy array с данными КТ
        percentile_min: нижний процентиль (например, 1 для 1-го процентиля)
        percentile_max: верхний процентиль (например, 99 для 99-го процентиля)

    Returns:
        нормализованные данные в диапазоне [0, 1]
    """
    # Вычисляем процентили
    p_min = np.percentile(data, percentile_min)
    p_max = np.percentile(data, percentile_max)

    # Обрабатываем случай, когда p_min >= p_max (все значения одинаковы)
    if p_max <= p_min:
        # Если все значения одинаковы, возвращаем константный тензор
        if p_min == 0:
            return np.zeros_like(data, dtype=np.float32)
        else:
            # Используем безопасное деление
            normalized = np.clip((data - p_min) / (abs(p_min) + 1e-8), 0, 1)
            return np.clip(normalized, 0, 1).astype(np.float32)

    # Нормализуем данные к диапазону [0, 1]
    normalized = (data - p_min) / (p_max - p_min)

    # Ограничиваем значения в диапазоне [0, 1]
    normalized = np.clip(normalized, 0, 1)

    return normalized.astype(np.float32)


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
# Трансформации с адаптивной нормализацией по процентилям
# -------------------------------
def get_transforms(target_output_shape=TARGET_OUTPUT_SHAPE, percentile_min=PERCENTILE_MIN,
                   percentile_max=PERCENTILE_MAX):
    def adaptive_normalize_by_percentiles(x):
        # x is a MetaTensor or numpy array
        if isinstance(x, MetaTensor):
            data = x.numpy()
        else:
            data = x

        # Применяем адаптивную нормализацию по процентилям
        normalized_data = adaptive_hu_normalization_by_percentiles(
            data,
            percentile_min=percentile_min,
            percentile_max=percentile_max
        )

        # Преобразуем обратно в тензор
        if isinstance(x, MetaTensor):
            return MetaTensor(normalized_data, meta=x.meta)
        else:
            return torch.tensor(normalized_data)

    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        CropForeground(select_fn=lambda x: x > -1000, channel_indices=0, margin=5),
        Lambda(lambda x: adaptive_normalize_by_percentiles(x)),  # Адаптивная нормализация по процентилям
        Lambda(lambda x: F.adaptive_avg_pool3d(x, target_output_shape)),
        NormalizeIntensity(subtrahend=0.5, divisor=0.5),  # [0,1] → [-1,1]
        ToTensor()
    ])


# -------------------------------
# Валидация размера
# -------------------------------
def validate_tensor_size(tensor, min_size=16):
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
# Обработка одного NIfTI-файла
# -------------------------------
def process_single_nii(nii_path: Path, output_dir: Path, percentile_min=PERCENTILE_MIN,
                       percentile_max=PERCENTILE_MAX, verbose: bool = False) -> dict:
    result = {
        'file_name': nii_path.name,
        'success': False,
        'error': None,
        'output_path': None,
        'original_shape': None,
        'oriented_shape': None,
        'final_shape': None,
        'spacing': None,
        'hu_range_input': None,
        'value_range_output': None,
        'mean': None,
        'std': None,
        'percentile_min_used': percentile_min,
        'percentile_max_used': percentile_max
    }

    try:
        # Используем функцию с корректной ориентацией
        ct_meta_tensor = load_nii_file(nii_path)
        original_shape = nii_path.stat().st_size  # размер файла
        oriented_shape = ct_meta_tensor.shape
        result.update({
            'original_shape': ct_meta_tensor.meta.get('original_shape'),
            'oriented_shape': oriented_shape,
            'spacing': ct_meta_tensor.meta.get('spacing'),
            'hu_range_input': ct_meta_tensor.meta.get('hu_range')
        })

        if verbose:
            logging.info(f"Loaded {nii_path.name}: oriented shape {oriented_shape}")

        validated_tensor = validate_tensor_size(ct_meta_tensor)
        transform = get_transforms(
            target_output_shape=TARGET_OUTPUT_SHAPE,
            percentile_min=percentile_min,
            percentile_max=percentile_max
        )
        transformed_tensor = transform(validated_tensor)
        final_tensor = transformed_tensor.unsqueeze(0)  # [D, H, W] → [1, D, H, W]

        expected_shape = (1, 1) + TARGET_OUTPUT_SHAPE
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

    return result


# -------------------------------
# Главная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CT NIfTI volumes for MedicalNet 3D ResNet models with adaptive HU normalization using percentiles"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with .nii or .nii.gz files (or subdirs)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for .nii/.nii.gz files recursively")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_nii_tensors.log",
                        help="Log file path")
    parser.add_argument("--method", type=str, choices=["canonical", "manual"],
                        default="canonical",
                        help="Method for axis correction: 'canonical' (recommended) or 'manual'")
    parser.add_argument("--percentile-min", type=float, default=PERCENTILE_MIN,
                        help="Lower percentile for adaptive normalization (default: 1)")
    parser.add_argument("--percentile-max", type=float, default=PERCENTILE_MAX,
                        help="Upper percentile for adaptive normalization (default: 99)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(Path(args.log_file))

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Поиск NIfTI-файлов
    pattern = "**/*.nii*" #if args.recursive else "*.nii*"
    nii_files = [f for f in input_dir.glob(pattern) if f.is_file() and f.suffix in ('.nii', '.gz')]
    # Фильтруем .nii.gz правильно
    nii_files = [f for f in nii_files if f.name.endswith(('.nii', '.nii.gz'))]

    if not nii_files:
        logger.error(f"No .nii or .nii.gz files found in {input_dir} (recursive={args.recursive})")
        return

    logger.info(f"Found {len(nii_files)} NIfTI file(s)")
    logger.info(f"Target output tensor shape: {(1, 1) + TARGET_OUTPUT_SHAPE}")
    logger.info(f"Using method: {args.method}")
    logger.info(f"Percentile normalization: {args.percentile_min}-{args.percentile_max}%")
    logger.info("✅ Using adaptive pooling — no resampling")

    results_summary = {
        'total': len(nii_files),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'resampling': 'none (adaptive pooling)',
            'percentile_normalization': [args.percentile_min, args.percentile_max],
            'output_intensity_range': [-1.0, 1.0],
            'correction_method': args.method
        }
    }

    logger.info("Starting processing...")
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files"):
        result = process_single_nii(
            nii_file,
            output_dir,
            percentile_min=args.percentile_min,
            percentile_max=args.percentile_max,
            verbose=args.verbose
        )
        if result['success']:
            results_summary['successful'] += 1
        else:
            results_summary['failed'] += 1
            results_summary['errors'].append({
                'file': result['file_name'],
                'error': result['error']
            })

    report_file = output_dir / "nii_processing_report.json"
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