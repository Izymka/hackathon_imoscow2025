#!/usr/bin/env python3
"""
prepare_ct_for_medicalnet.py

Подготовка КТ-данных для модели MedicalNet с сохранением исходного разрешения.
Возвращает тензоры [1, 1, 128, 128, 128] с адаптивной процентильной нормализацией [-1, 1].
Без ресемплинга — исходное разрешение в плоскости и реальная толщина среза сохраняются.
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import pydicom
import SimpleITK as sitk
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
TARGET_OUTPUT_SHAPE = (256, 256, 256)  # Желаемый выход после адаптивного пулинга
PERCENTILE_MIN = 1.0  # Процентиль для минимального значения
PERCENTILE_MAX = 99.0  # Процентиль для максимального значения


# -------------------------------
# Адаптивная процентильная нормализация
# -------------------------------
def adaptive_percentile_normalization(image, p_min=PERCENTILE_MIN, p_max=PERCENTILE_MAX):
    """
    Адаптивная процентильная нормализация для сохранения контраста между разными тканями
    """
    # Преобразуем в numpy если тензор
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
    else:
        img_np = image

    # Вычисляем процентили
    p_min_val = np.percentile(img_np, p_min)
    p_max_val = np.percentile(img_np, p_max)

    # Нормализуем в диапазон [0, 1]
    normalized = np.clip((img_np - p_min_val) / (p_max_val - p_min_val + 1e-8), 0, 1)

    # Преобразуем обратно в тензор
    if isinstance(image, torch.Tensor):
        return torch.from_numpy(normalized).to(image.device)
    else:
        return normalized


# -------------------------------
# Настройка логирования
# -------------------------------
def setup_logging(log_file: Path):
    """Настройка логирования в файл и консоль"""
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
# Трансформации MONAI (с адаптивной нормализацией)
# -------------------------------
def get_transforms(target_output_shape=TARGET_OUTPUT_SHAPE):
    """Трансформации БЕЗ ресемплинга — сохраняем исходное разрешение"""
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        CropForeground(select_fn=lambda x: x > -1000, channel_indices=0, margin=5),
        Lambda(lambda x: adaptive_percentile_normalization(x, PERCENTILE_MIN, PERCENTILE_MAX)),
        Lambda(lambda x: F.adaptive_avg_pool3d(x, target_output_shape)),
        NormalizeIntensity(subtrahend=0.5, divisor=0.5),  # [0,1] → [-1,1]
        ToTensor()
    ])


# -------------------------------
# Валидация размеров (без модификации тензора)
# -------------------------------
def validate_tensor_size(tensor, min_size=16):
    """Проверяет минимальный размер, не изменяя тензор"""
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
# Загрузка DICOM (остаётся без изменений)
# -------------------------------
def _load_single_dicom_file(single_file: Path) -> MetaTensor:
    ds = pydicom.dcmread(str(single_file), force=True)
    num_frames = int(getattr(ds, "NumberOfFrames", 1))

    if num_frames > 1:
        pixel_array = ds.pixel_array
        spacing = [
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[0]),
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[1]),
            float(getattr(ds, "SpacingBetweenSlices", 1.0)),
        ]
        tensor = torch.from_numpy(pixel_array.astype(np.float32))
        tensor = tensor.unsqueeze(0)
        return MetaTensor(tensor, affine=np.diag(spacing + [1]))
    else:
        pixel_array = ds.pixel_array
        spacing = [
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[0]),
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[1]),
            1.0,
        ]
        tensor = torch.from_numpy(pixel_array.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return MetaTensor(tensor, affine=np.diag(spacing + [1]))


def load_dicom_series(dicom_folder: Path) -> MetaTensor:
    try:
        files = sorted([p for p in dicom_folder.iterdir()
                        if p.is_file() and not p.name.startswith('.')])
        if len(files) == 1:
            return _load_single_dicom_file(files[0])

        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
        if series_IDs:
            series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
            if len(series_file_names) < 3:
                raise ValueError(f"Too few DICOM slices: {len(series_file_names)}")

            reader.SetFileNames(series_file_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            sitk_image = reader.Execute()

            original_size = sitk_image.GetSize()
            if any(s <= 0 for s in original_size):
                raise ValueError(f"Invalid image dimensions: {original_size}")

            image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            original_spacing = sitk_image.GetSpacing()
            spacing_dhw = [original_spacing[2], original_spacing[1], original_spacing[0]]

            slope, intercept = 1.0, 0.0
            try:
                dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
                slope = float(getattr(dcm, 'RescaleSlope', 1.0))
                intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            except Exception:
                pass

            if slope != 1.0 or intercept != 0.0:
                image_array = image_array * slope - intercept

            meta_dict = {
                'spacing': spacing_dhw,
                'original_shape': image_array.shape,
                'original_spacing': original_spacing,
                'filename_or_obj': str(dicom_folder),
                'rescale_slope': slope,
                'rescale_intercept': intercept,
                'hu_range': (float(image_array.min()), float(image_array.max())),
                'source': 'sitk_series'
            }
            return MetaTensor(image_array, meta=meta_dict)

        # Fallback: если SimpleITK не нашёл серии
        if not files:
            raise FileNotFoundError(f"No files found in {dicom_folder}")

        if len(files) == 1:
            single = files[0]
            ds = pydicom.dcmread(str(single), force=True)
            num_frames = int(getattr(ds, 'NumberOfFrames', 1))
            if num_frames > 1:
                pixel_array = ds.pixel_array
                image_array = np.asarray(pixel_array).astype(np.float32)
                if image_array.ndim == 4 and image_array.shape[-1] in (1, 3):
                    image_array = image_array[..., 0] if image_array.shape[-1] == 1 else image_array[..., 0]
                if image_array.ndim != 3:
                    raise ValueError(f"Unexpected shape: {image_array.shape}")

                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    image_array = image_array * slope - intercept

                pixel_spacing = (1.0, 1.0)
                if hasattr(ds, 'PixelSpacing'):
                    ps = ds.PixelSpacing
                    pixel_spacing = (float(ps[0]), float(ps[1]))
                slice_spacing = 1.0
                if hasattr(ds, 'SpacingBetweenFrames'):
                    slice_spacing = float(ds.SpacingBetweenFrames)
                elif hasattr(ds, 'SliceThickness'):
                    slice_spacing = float(ds.SliceThickness)

                spacing_dhw = [slice_spacing, pixel_spacing[0], pixel_spacing[1]]
                meta_dict = {
                    'spacing': spacing_dhw,
                    'original_shape': image_array.shape,
                    'original_spacing': spacing_dhw,
                    'filename_or_obj': str(single),
                    'rescale_slope': slope,
                    'rescale_intercept': intercept,
                    'hu_range': (float(image_array.min()), float(image_array.max())),
                    'source': 'pydicom_multiframe'
                }
                return MetaTensor(image_array, meta=meta_dict)
            else:
                pixel_array = ds.pixel_array.astype(np.float32)
                image_array = np.expand_dims(pixel_array, axis=0) if pixel_array.ndim == 2 else pixel_array
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    image_array = image_array * slope - intercept
                pixel_spacing = (1.0, 1.0)
                if hasattr(ds, 'PixelSpacing'):
                    ps = ds.PixelSpacing
                    pixel_spacing = (float(ps[0]), float(ps[1]))
                slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
                spacing_dhw = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
                meta_dict = {
                    'spacing': spacing_dhw,
                    'original_shape': image_array.shape,
                    'original_spacing': spacing_dhw,
                    'filename_or_obj': str(single),
                    'rescale_slope': slope,
                    'rescale_intercept': intercept,
                    'hu_range': (float(image_array.min()), float(image_array.max())),
                    'source': 'pydicom_singlefile'
                }
                return MetaTensor(image_array, meta=meta_dict)

        # Fallback: сборка по InstanceNumber
        dicom_files = []
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                inst = getattr(ds, 'InstanceNumber', None)
                dicom_files.append((p, inst))
            except Exception:
                dicom_files.append((p, None))
        dicom_files_sorted = sorted(dicom_files,
                                    key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0, str(x[0])))
        series_file_names = [str(x[0]) for x in dicom_files_sorted]

        if len(series_file_names) < 3:
            raise ValueError(f"Too few DICOM slices (fallback): {len(series_file_names)}")

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        sitk_image = reader.Execute()
        image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        original_spacing = sitk_image.GetSpacing()
        spacing_dhw = [original_spacing[2], original_spacing[1], original_spacing[0]]

        slope, intercept = 1.0, 0.0
        try:
            dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        except Exception:
            pass

        if slope != 1.0 or intercept != 0.0:
            image_array = image_array * slope - intercept

        meta_dict = {
            'spacing': spacing_dhw,
            'original_shape': image_array.shape,
            'original_spacing': original_spacing,
            'filename_or_obj': str(dicom_folder),
            'rescale_slope': slope,
            'rescale_intercept': intercept,
            'hu_range': (float(image_array.min()), float(image_array.max())),
            'source': 'sitk_fallback'
        }
        return MetaTensor(image_array, meta=meta_dict)

    except Exception as e:
        raise Exception(f"Failed to load DICOM from {dicom_folder}: {str(e)}")


# -------------------------------
# Обработка одного пациента
# -------------------------------
def process_single_patient(patient_dir: Path, output_dir: Path, verbose: bool = False) -> dict:
    result = {
        'patient_name': patient_dir.name,
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
        ct_meta_tensor = load_dicom_series(patient_dir)
        original_shape = ct_meta_tensor.shape
        result.update({
            'original_shape': original_shape,
            'original_spacing': ct_meta_tensor.meta.get('spacing'),
            'hu_range_input': ct_meta_tensor.meta.get('hu_range')
        })

        if verbose:
            logging.info(f"Loaded {patient_dir.name}: {original_shape}")

        validated_tensor = validate_tensor_size(ct_meta_tensor)
        transform = get_transforms()
        transformed_tensor = transform(validated_tensor)
        final_tensor = transformed_tensor.unsqueeze(0)  # [1, C, D, H, W] → [1, 1, 128, 128, 128]

        expected_shape = (1, 1) + TARGET_OUTPUT_SHAPE
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        if torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any():
            raise ValueError("Invalid values (NaN/Inf) in final tensor")

        output_path = output_dir / f"{patient_dir.name}.pt"
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
            logging.info(f"✓ {patient_dir.name}: "
                         f"range=[{result['value_range_output'][0]:.3f}, {result['value_range_output'][1]:.3f}], "
                         f"mean={result['mean']:.3f}, std={result['std']:.3f}")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            logging.error(f"❌ {patient_dir.name}: {e}")
            logging.debug(traceback.format_exc())

    return result


# -------------------------------
# Главная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CT DICOM volumes for MedicalNet 3D ResNet models (no resampling)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with patient subdirectories containing DICOM files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_tensors.log",
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

    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not patient_dirs:
        logger.error(f"No patient directories found in {input_dir}")
        return

    logger.info(f"Found {len(patient_dirs)} patient directories")
    logger.info(f"Target output tensor shape: {(1, 1) + TARGET_OUTPUT_SHAPE}")
    logger.info("✅ NO isotropic resampling — original resolution preserved (in-plane & slice thickness)")
    logger.info(f"✅ Adaptive percentile normalization: {PERCENTILE_MIN}-{PERCENTILE_MAX}%")

    results_summary = {
        'total': len(patient_dirs),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'resampling': 'none',
            'percentile_normalization_range': [PERCENTILE_MIN, PERCENTILE_MAX],
            'output_intensity_range': [-1.0, 1.0]
        }
    }

    logger.info("Starting processing...")
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = process_single_patient(patient_dir, output_dir, args.verbose)
        if result['success']:
            results_summary['successful'] += 1
        else:
            results_summary['failed'] += 1
            results_summary['errors'].append({
                'patient': result['patient_name'],
                'error': result['error']
            })

    report_file = output_dir / "processing_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("PROCESSING COMPLETED")
    logger.info(f"Total patients: {results_summary['total']}")
    logger.info(f"Successfully processed: {results_summary['successful']}")
    logger.info(f"Failed: {results_summary['failed']}")
    logger.info(f"Success rate: {100 * results_summary['successful'] / results_summary['total']:.1f}%")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"{'=' * 60}")

    if results_summary['failed'] > 0:
        logger.warning(f"⚠️  {results_summary['failed']} patients failed. Check the report for details.")

    logger.info("Done! ✅")


if __name__ == "__main__":
    main()