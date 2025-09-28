#!/usr/bin/env python3
"""
Подготовка КТ-данных из NIfTI (.nii / .nii.gz) для MedicalNet с 4-канальным оконным представлением:
- Лёгочное окно
- Медиастинальное окно
- Костное окно
- Окно мягких тканей

Возвращает тензор [1, 4, 128, 128, 128], где каждый канал — результат оконной фильтрации.
С применением изотропного ресемплинга перед адаптивным пулингом для сохранения анатомии.
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
    Lambda,
    Spacing,
    Orientation,
    ResizeWithPadOrCrop
)
import traceback
from tqdm import tqdm
import json
import torch.nn.functional as F

# -------------------------------
# Конфигурация
# -------------------------------
TARGET_OUTPUT_SHAPE = (128, 128, 128)
TARGET_SPACING_MM = 1.0  # Изотропный воксель в мм

WINDOW_PRESETS = {
    'lung': {'center': -600, 'width': 1200},      # Легочное окно
    'mediastinal': {'center': 50, 'width': 400},  # Медиастинальное окно
    'bone': {'center': 300, 'width': 1500},       # Костное окно
    'soft_tissue': {'center': 40, 'width': 80}    # Мягкие ткани (узкое окно)
}


# -------------------------------
# Оконная фильтрация
# -------------------------------
def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Применяет оконную фильтрацию и нормализует в [0, 1]"""
    min_val = center - width // 2
    max_val = center + width // 2
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val + 1e-8)
    return windowed.astype(np.float32)


def apply_all_windows(image: np.ndarray) -> np.ndarray:
    """Применяет 4 окна → [4, D, H, W]"""
    channels = []
    for preset in WINDOW_PRESETS.values():
        ch = apply_window(image, preset['center'], preset['width'])
        channels.append(ch)
    return np.stack(channels, axis=0)  # [4, D, H, W]


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

        # Приводим к канонической ориентации
        img_oriented = nib.as_closest_canonical(nii_img)
        data_oriented = img_oriented.get_fdata().astype(np.float32)
        affine_oriented = img_oriented.affine

        if data_oriented.ndim == 3:
            # (x, y, z) → (z, y, x) → (D, H, W)
            data_oriented = np.transpose(data_oriented, (2, 1, 0))
            # Исправление: отражение по оси H (вторая ось)
            data_oriented = np.flip(data_oriented, axis=1)
        elif data_oriented.ndim == 4:
            if data_oriented.shape[-1] in (1, 3):
                data_oriented = data_oriented[..., 0]
            else:
                data_oriented = data_oriented[:, :, :, 0]
            data_oriented = np.transpose(data_oriented, (2, 1, 0))
            data_oriented = np.flip(data_oriented, axis=1)
        else:
            raise ValueError(f"Unsupported dimension: {data_oriented.ndim}")

        data_oriented = np.ascontiguousarray(data_oriented.copy())

        # Извлекаем спейсинг из аффина
        spacing = np.sqrt(np.sum(affine_oriented[:3, :3] ** 2, axis=0))
        # [X, Y, Z] -> [Z, Y, X] для [D, H, W]
        spacing_dhw = [spacing[2], spacing[1], spacing[0]]

        meta_dict = {
            'spacing': spacing_dhw,
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
# Трансформации MONAI с изотропией
# -------------------------------
def get_transforms(target_output_shape=TARGET_OUTPUT_SHAPE, target_spacing=TARGET_SPACING_MM):
    """Трансформации с изотропным ресемплингом"""

    def apply_isotropic_resampling_and_windows(x):
        # x: MetaTensor [D, H, W] с метаданными
        # Извлекаем спейсинг
        original_spacing = x.meta.get('spacing', [1.0, 1.0, 1.0])
        # MONAI ожидает [H_spacing, W_spacing, D_spacing], но у нас [D, H, W] => [Z, Y, X]
        spacing_xyz = [original_spacing[2], original_spacing[1], original_spacing[0]]

        # 1. Применяем ресемплинг к изотропному разрешению
        resampler = Spacing(pixdim=target_spacing, mode="trilinear", padding_mode="border")
        x_resampled = resampler(x, mode="trilinear", padding_mode="border")

        # 2. Приводим к RAS ориентации (опционально, но полезно)
        orienter = Orientation(axcodes="RAS")
        x_oriented = orienter(x_resampled)

        # 3. Обрезаем/дополняем до нужного размера
        resizer = ResizeWithPadOrCrop(spatial_size=target_output_shape)
        x_resized = resizer(x_oriented)

        # 4. Применяем окна
        image_np = x_resized.numpy()  # [D, H, W]

        # Убедиться, что это 3D
        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]

        windows_np = apply_all_windows(image_np)  # [4, D, H, W]

        # 5. Преобразуем в тензор, добавляем batch и нормализуем
        x_tensor = torch.from_numpy(windows_np).unsqueeze(0)  # [1, 4, D, H, W]
        x_tensor = x_tensor * 2.0 - 1.0  # нормализация [-1, 1]

        return x_tensor

    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),  # [D, H, W] → [1, D, H, W]
        CropForeground(select_fn=lambda x: x > -1000, channel_indices=0, margin=5),
        Lambda(apply_isotropic_resampling_and_windows),
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
        'oriented_shape': None,
        'final_shape': None,
        'spacing': None,
        'hu_range_input': None,
        'value_range_output': None,
        'mean': None,
        'std': None
    }

    try:
        ct_meta_tensor = load_nii_file(nii_path)
        result.update({
            'original_shape': ct_meta_tensor.meta.get('original_shape'),
            'oriented_shape': ct_meta_tensor.shape,
            'spacing': ct_meta_tensor.meta.get('spacing'),
            'hu_range_input': ct_meta_tensor.meta.get('hu_range')
        })

        if verbose:
            logging.info(f"Loaded {nii_path.name}: oriented shape {ct_meta_tensor.shape}")

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

    return result


# -------------------------------
# Главная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CT NIfTI volumes for MedicalNet with 4-channel windowing (isotropic resampling)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with .nii or .nii.gz files (or subdirs)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for .nii/.nii.gz files recursively")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_nii_multichannel_isotropic.log",
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
    all_files = input_dir.glob(pattern)
    nii_files = [
        f for f in all_files
        if f.is_file() and (f.suffix == '.nii' or f.suffixes[-2:] == ['.nii', '.gz'])
    ]

    if not nii_files:
        logger.error(f"No .nii or .nii.gz files found in {input_dir} (recursive={args.recursive})")
        return

    logger.info(f"Found {len(nii_files)} NIfTI file(s)")
    logger.info(f"Target output tensor shape: {(1, 4) + TARGET_OUTPUT_SHAPE}")
    logger.info(f"Target spacing: {TARGET_SPACING_MM} mm (isotropic)")
    logger.info("✅ Using 4-channel windowing: lung, mediastinal, bone, soft_tissue")
    logger.info("✅ Orientation corrected (canonical + flip on H-axis)")
    logger.info("✅ Isotropic resampling applied before adaptive pooling")
    logger.info("✅ Output intensity range: [-1, 1]")

    results_summary = {
        'total': len(nii_files),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'target_spacing_mm': TARGET_SPACING_MM,
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

    report_file = output_dir / "nii_multichannel_processing_report_isotropic.json"
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