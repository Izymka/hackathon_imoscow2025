#!/usr/bin/env python3
"""

Подготовка КТ-данных для модели MedicalNet с изотропным ресемплингом до 1x1x1 мм.
Возвращает тензоры [1, 1, 256, 256, 256] с фиксированной нормализацией [-1000, 600] → [-1, 1].
Полностью без MONAI - использует только SimpleITK, NumPy и PyTorch.
"""

import os
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import pydicom
import SimpleITK as sitk
import traceback
from tqdm import tqdm
import json
import torch.nn.functional as F

# -------------------------------
# Конфигурация
# -------------------------------
TARGET_OUTPUT_SHAPE = (256, 256, 256)  # Желаемый выходной размер
TARGET_SPACING = (1.0, 1.0, 1.0)  # Изотропное разрешение 1x1x1 мм
HU_MIN = -1000.0  # Минимальное значение для КТ
HU_MAX = 600.0  # Максимальное значение для КТ


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
# Вспомогательные функции для трансформаций
# -------------------------------
def ensure_channel_first(image_array: np.ndarray) -> np.ndarray:
    """Добавляет channel dimension если его нет"""
    if image_array.ndim == 3:
        return image_array[np.newaxis, :, :, :]  # [C, D, H, W]
    elif image_array.ndim == 4:
        return image_array
    else:
        raise ValueError(f"Unexpected array dimension: {image_array.ndim}")


def resample_image_sitk(sitk_image, target_spacing, target_size=None, is_label=False):
    """Ресемплинг изображения до целевого разрешения"""
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()

    if target_size is None:
        # Вычисляем размер исходя из целевого разрешения
        target_size = [
            int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
            int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
            int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
        ]

    # Определяем тип интерполяции
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    # Создаем трансформацию ресемплинга
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(target_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(interpolator)

    return resample.Execute(sitk_image)


def crop_foreground_3d(image_array: np.ndarray, threshold: float = -1000, margin: int = 5) -> np.ndarray:
    """Кроп изображения по области с значениями выше threshold"""
    if image_array.ndim != 3:
        raise ValueError("Expected 3D array for cropping")

    # Создаем маску для вокселей выше порога
    mask = image_array > threshold

    if not np.any(mask):
        return image_array

    # Находим bounding box
    coords = np.array(np.where(mask))

    min_idx = np.maximum(coords.min(axis=1) - margin, 0)
    max_idx = np.minimum(coords.max(axis=1) + margin + 1, image_array.shape)

    # Выполняем кроп
    cropped = image_array[
        min_idx[0]:max_idx[0],
        min_idx[1]:max_idx[1],
        min_idx[2]:max_idx[2]
    ]

    return cropped


def scale_intensity_range(image_array: np.ndarray,
                          input_min: float,
                          input_max: float,
                          output_min: float = -1.0,
                          output_max: float = 1.0,
                          clip: bool = True) -> np.ndarray:
    """Масштабирование интенсивности"""
    if clip:
        image_array = np.clip(image_array, input_min, input_max)

    # Линейное преобразование
    image_array = (image_array - input_min) / (input_max - input_min)
    image_array = image_array * (output_max - output_min) + output_min

    return image_array


def resize_3d_tensor(tensor: torch.Tensor, target_shape: tuple, mode: str = 'trilinear') -> torch.Tensor:
    """Ресайз 3D тензора до целевого размера"""
    if tensor.ndim == 3:
        # [D, H, W] -> [1, 1, D, H, W] для интерполяции
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        # [C, D, H, W] -> [1, C, D, H, W]
        tensor = tensor.unsqueeze(0)

    # Интерполяция
    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode=mode,
        align_corners=False
    )

    return resized


# -------------------------------
# Загрузка DICOM (исправленная для многофреймовых файлов)
# -------------------------------
def load_multiframe_dicom(dicom_file: Path) -> dict:
    """Загрузка многофреймового DICOM файла"""
    try:
        ds = pydicom.dcmread(str(dicom_file), force=True)

        # Получаем pixel array
        pixel_array = ds.pixel_array.astype(np.float32)

        # Определяем размерность
        if pixel_array.ndim == 4:
            # [frames, height, width, channels] -> [frames, height, width]
            if pixel_array.shape[-1] in [1, 3]:
                pixel_array = pixel_array[..., 0]
            else:
                raise ValueError(f"Unexpected 4D array shape: {pixel_array.shape}")
        elif pixel_array.ndim == 3:
            # [frames, height, width] - уже правильный формат
            pass
        else:
            raise ValueError(f"Unexpected pixel array dimensions: {pixel_array.ndim}")

        # Получаем параметры рескейла
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

        # Применяем рескейл
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept

        # Получаем spacing
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        slice_spacing = getattr(ds, "SpacingBetweenFrames", 1.0)
        if slice_spacing == 1.0:
            slice_spacing = getattr(ds, "SliceThickness", 1.0)

        spacing = [float(slice_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])]

        # Создаем SimpleITK изображение из numpy array
        sitk_image = sitk.GetImageFromArray(pixel_array)
        sitk_image.SetSpacing(spacing[::-1])  # SimpleITK использует [x, y, z] порядок

        return {
            'image_array': pixel_array,
            'sitk_image': sitk_image,
            'original_spacing': spacing[::-1],  # [x, y, z] для консистентности
            'original_size': pixel_array.shape[::-1],  # [x, y, z]
            'rescale_slope': slope,
            'rescale_intercept': intercept,
            'hu_range': (float(pixel_array.min()), float(pixel_array.max())),
            'is_multiframe': True
        }

    except Exception as e:
        raise Exception(f"Failed to load multiframe DICOM {dicom_file}: {str(e)}")


def load_dicom_series_sitk(dicom_folder: Path) -> dict:
    """Загрузка DICOM серии через SimpleITK с поддержкой многофреймовых файлов"""
    try:
        files = sorted([p for p in dicom_folder.iterdir()
                        if p.is_file() and not p.name.startswith('.')])

        if not files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_folder}")

        # Проверяем первый файл на многофреймовость
        first_file = files[0]
        try:
            ds_test = pydicom.dcmread(str(first_file), stop_before_pixels=True)
            num_frames = int(getattr(ds_test, "NumberOfFrames", 1))

            if num_frames > 1:
                # Это многофреймовый файл
                logging.info(f"Loading multiframe DICOM with {num_frames} frames: {first_file.name}")
                return load_multiframe_dicom(first_file)
        except Exception as e:
            logging.warning(f"Could not check multiframe status: {e}")
            # Продолжаем с обычной загрузкой

        # Обычная загрузка серии файлов
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))

        if not series_IDs:
            # Если SimpleITK не нашел серии, пробуем загрузить все файлы как серию
            series_file_names = [str(f) for f in files]
            if len(series_file_names) == 1:
                # Если только один файл, проверяем его на многофреймовость
                try:
                    ds_single = pydicom.dcmread(series_file_names[0], force=True)
                    num_frames_single = int(getattr(ds_single, "NumberOfFrames", 1))
                    if num_frames_single > 1:
                        return load_multiframe_dicom(first_file)
                    else:
                        raise ValueError(f"Single file DICOM with only {num_frames_single} frame(s)")
                except Exception as e:
                    raise ValueError(f"Cannot process single file: {e}")
        else:
            series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])

        if len(series_file_names) < 3:
            raise ValueError(f"Too few DICOM slices: {len(series_file_names)}")

        # Загружаем изображение
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        sitk_image = reader.Execute()

        # Получаем данные
        image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # [D, H, W]
        original_spacing = sitk_image.GetSpacing()  # (x, y, z)
        original_size = sitk_image.GetSize()

        # Получаем параметры рескейла из первого файла
        rescale_slope, rescale_intercept = 1.0, 0.0
        try:
            dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
            rescale_slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            rescale_intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        except Exception:
            pass

        # Применяем рескейл если нужно
        if rescale_slope != 1.0 or rescale_intercept != 0.0:
            image_array = image_array * rescale_slope + rescale_intercept

        return {
            'image_array': image_array,
            'sitk_image': sitk_image,
            'original_spacing': original_spacing,
            'original_size': original_size,
            'rescale_slope': rescale_slope,
            'rescale_intercept': rescale_intercept,
            'hu_range': (float(image_array.min()), float(image_array.max())),
            'is_multiframe': False
        }

    except Exception as e:
        raise Exception(f"Failed to load DICOM from {dicom_folder}: {str(e)}")


# -------------------------------
# Основной пайплайн обработки
# -------------------------------
def process_ct_volume(dicom_folder: Path) -> torch.Tensor:
    """Основной пайплайн обработки КТ объема"""
    # 1. Загрузка DICOM
    dicom_data = load_dicom_series_sitk(dicom_folder)
    image_array = dicom_data['image_array']  # [D, H, W] или [frames, H, W]
    sitk_image = dicom_data['sitk_image']

    # 2. Ресемплинг до изотропного разрешения
    target_spacing_xyz = TARGET_SPACING  # (x, y, z) для SimpleITK

    # Вычисляем целевой размер для ресемплинга (сохраняем примерный объем)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    target_size_xyz = [
        int(round(original_size[0] * original_spacing[0] / target_spacing_xyz[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing_xyz[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing_xyz[2]))
    ]

    # Убедимся, что размеры не нулевые
    target_size_xyz = [max(1, dim) for dim in target_size_xyz]

    resampled_sitk = resample_image_sitk(
        sitk_image,
        target_spacing_xyz,
        target_size_xyz,
        is_label=False
    )

    resampled_array = sitk.GetArrayFromImage(resampled_sitk)  # [D, H, W]

    # 3. Кроп области с данными
    cropped_array = crop_foreground_3d(resampled_array, threshold=-1000, margin=5)

    # 4. Нормализация интенсивности HU -> [-1, 1]
    normalized_array = scale_intensity_range(
        cropped_array,
        input_min=HU_MIN,
        input_max=HU_MAX,
        output_min=-1.0,
        output_max=1.0,
        clip=True
    )

    # 5. Конвертация в тензор и ресайз до финального размера
    tensor = torch.from_numpy(normalized_array).float()  # [D, H, W]

    # Ресайз до целевого размера
    resized_tensor = resize_3d_tensor(tensor, TARGET_OUTPUT_SHAPE, mode='trilinear')

    # Убедимся, что размерность правильная [1, 1, D, H, W]
    if resized_tensor.ndim == 5:
        final_tensor = resized_tensor
    elif resized_tensor.ndim == 4:
        final_tensor = resized_tensor.unsqueeze(0)  # [1, C, D, H, W]
    else:
        final_tensor = resized_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    return final_tensor, dicom_data


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
        'resampled_spacing': TARGET_SPACING,
        'hu_range_input': None,
        'value_range_output': None,
        'mean': None,
        'std': None
    }

    try:
        # Обработка объема
        final_tensor, dicom_data = process_ct_volume(patient_dir)

        # Проверка результата
        expected_shape = (1, 1) + TARGET_OUTPUT_SHAPE
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        if torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any():
            raise ValueError("Invalid values (NaN/Inf) in final tensor")

        # Сохранение
        output_path = output_dir / f"{patient_dir.name}.pt"
        torch.save(final_tensor, output_path)

        # Заполнение результата
        result.update({
            'success': True,
            'output_path': str(output_path),
            'original_shape': dicom_data['image_array'].shape,
            'final_shape': tuple(final_tensor.shape),
            'original_spacing': dicom_data['original_spacing'],
            'hu_range_input': dicom_data['hu_range'],
            'value_range_output': (float(final_tensor.min()), float(final_tensor.max())),
            'mean': float(final_tensor.mean()),
            'std': float(final_tensor.std())
        })

        if verbose:
            logging.info(f"✓ {patient_dir.name}: "
                         f"original={result['original_shape']}, "
                         f"final={result['final_shape']}, "
                         f"range=[{result['value_range_output'][0]:.3f}, {result['value_range_output'][1]:.3f}]")

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
        description="Prepare CT DICOM volumes for MedicalNet with isotropic resampling (1x1x1 mm)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with patient subdirectories containing DICOM files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_tensors_medicalnet.log",
                        help="Log file path")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file)
    logger = setup_logging(log_file)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not patient_dirs:
        # Check if there are DICOM files directly in the input directory
        dicom_files = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        if dicom_files:
            logger.info(f"Found {len(dicom_files)} files in input directory, processing as single patient case")
            patient_dirs = [input_dir]
        else:
            logger.error(f"No patient directories or DICOM files found in {input_dir}")
            sys.exit(1)


    logger.info(f"Found {len(patient_dirs)} patient directories/cases to process")
    logger.info(f"Target output tensor shape: {(1, 1) + TARGET_OUTPUT_SHAPE}")
    logger.info(f"✅ Isotropic resampling to: {TARGET_SPACING} mm")
    logger.info(f"✅ HU range normalization: [{HU_MIN}, {HU_MAX}] → [-1.0, 1.0]")
    logger.info(f"✅ MedicalNet-compatible preprocessing (MONAI-free)")
    logger.info(f"✅ Support for multiframe DICOM files")

    results_summary = {
        'total': len(patient_dirs),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_output_shape': TARGET_OUTPUT_SHAPE,
            'resampling': 'isotropic',
            'target_spacing_mm': TARGET_SPACING,
            'hu_normalization_range': [HU_MIN, HU_MAX],
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

    report_file = output_dir / "processing_report_medicalnet.json"
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