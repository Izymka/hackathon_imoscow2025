#!/usr/bin/env python3
"""
prepare_ct_nii_for_medicalnet.py

Подготовка КТ-данных из NIfTI (.nii / .nii.gz) для модели MedicalNet.
Сохраняет тензоры [1, 1, 256, 256, 256] с нормализацией [-1, 1].
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
# Загрузка NIfTI
# -------------------------------
def load_nii_file(nii_path: Path) -> MetaTensor:
    """
    Загружает NIfTI-файл и возвращает MetaTensor с метаданными.
    Предполагается, что данные — КТ в HU.
    """
    try:
        nii_img = nib.load(str(nii_path))
        data = nii_img.get_fdata().astype(np.float32)

        # Получаем affine и spacing
        affine = nii_img.affine
        # Spacing — абсолютные значения диагонали (в мм)
        spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        # Обычно nibabel возвращает (H, W, D), MONAI ожидает (D, H, W)
        # Но порядок зависит от ориентации! Для простоты предположим стандартный RAS.
        # Если данные в другом порядке — может потребоваться reorientation.
        if data.ndim == 3:
            # Преобразуем в (D, H, W)
            data = np.transpose(data, (2, 0, 1))
        elif data.ndim == 4:
            if data.shape[-1] in (1, 3):
                data = data[..., 0] if data.shape[-1] == 1 else data[..., 0]
                data = np.transpose(data, (2, 0, 1))
            else:
                raise ValueError(f"Unsupported 4D shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported dimension: {data.ndim}")

        meta_dict = {
            'spacing': spacing.tolist(),
            'original_shape': data.shape,
            'original_affine': affine,
            'filename_or_obj': str(nii_path),
            'source': 'nifti',
            'hu_range': (float(data.min()), float(data.max())),
        }
        return MetaTensor(data, meta=meta_dict)

    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI {nii_path}: {e}")


# -------------------------------
# Трансформации (аналогично DICOM-скрипту)
# -------------------------------
def get_transforms(target_output_shape=TARGET_OUTPUT_SHAPE):
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        CropForeground(select_fn=lambda x: x > -1000, channel_indices=0, margin=5),
        ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
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
def process_single_nii(nii_path: Path, output_dir: Path, verbose: bool = False) -> dict:
    result = {
        'file_name': nii_path.name,
        'success': False,
        'error': None,
        'output_path': None,
        'original_shape': None,
        'final_shape': None,
        'spacing': None,
        'hu_range_input': None,
        'value_range_output': None,
        'mean': None,
        'std': None
    }

    try:
        ct_meta_tensor = load_nii_file(nii_path)
        original_shape = ct_meta_tensor.shape
        result.update({
            'original_shape': original_shape,
            'spacing': ct_meta_tensor.meta.get('spacing'),
            'hu_range_input': ct_meta_tensor.meta.get('hu_range')
        })

        if verbose:
            logging.info(f"Loaded {nii_path.name}: {original_shape}")

        validated_tensor = validate_tensor_size(ct_meta_tensor)
        transform = get_transforms()
        transformed_tensor = transform(validated_tensor)
        final_tensor = transformed_tensor.unsqueeze(0)  # [1, D, H, W] → [1, 1, D, H, W]

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
        description="Prepare CT NIfTI volumes for MedicalNet 3D ResNet models"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with .nii or .nii.gz files (or subdirs)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for .nii/.nii.gz files recursively")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_nii_medicalnet.log",
                        help="Log file path")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(Path(args.log_file))

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Поиск NIfTI-файлов
    pattern = "**/*.nii*" if args.recursive else "*.nii*"
    nii_files = [f for f in input_dir.glob(pattern) if f.is_file() and f.suffix in ('.nii', '.gz')]
    # Фильтруем .nii.gz правильно
    nii_files = [f for f in nii_files if f.name.endswith(('.nii', '.nii.gz'))]

    if not nii_files:
        logger.error(f"No .nii or .nii.gz files found in {input_dir} (recursive={args.recursive})")
        return

    logger.info(f"Found {len(nii_files)} NIfTI file(s)")
    logger.info(f"Target output tensor shape: {(1, 1) + TARGET_OUTPUT_SHAPE}")
    logger.info("✅ Using adaptive pooling — no resampling")

    results_summary = {
        'total': len(nii_files),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'resampling': 'none (adaptive pooling)',
            'hu_clipping_range': [HU_MIN, HU_MAX],
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