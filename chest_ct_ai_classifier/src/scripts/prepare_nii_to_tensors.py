#!/usr/bin/env python3
"""

Подготовка КТ-данных для модели MedicalNet с изотропным ресемплингом до 1x1x1 мм.
Поддержка DICOM и NIfTI форматов.
Возвращает тензоры [1, 1, 256, 256, 256] с фиксированной нормализацией [-1000, 600] → [-1, 1].
Полностью без MONAI - использует только SimpleITK, NumPy и PyTorch.
"""

import os
import argparse
import logging
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
        # Вычисляем размер исходя из целевого разрешения :cite[2]
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

    # Создаем трансформацию ресемплинга :cite[7]
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
# Загрузка NIfTI файлов :cite[1]:cite[5]:cite[6]
# -------------------------------
def load_nifti_file(nifti_path: Path) -> dict:
    """Загрузка NIfTI файла (.nii или .nii.gz)"""
    try:
        # Загрузка через SimpleITK :cite[6]
        sitk_image = sitk.ReadImage(str(nifti_path))

        # Получаем данные в numpy array :cite[5]
        image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # [D, H, W]

        # Получаем информацию о пространственных характеристиках :cite[1]
        original_spacing = sitk_image.GetSpacing()  # (x, y, z)
        original_size = sitk_image.GetSize()

        # Получаем дополнительную информацию из заголовка :cite[5]
        header_info = {}
        try:
            # Альтернативный способ получения информации через nibabel (если установлен)
            import nibabel as nib
            nib_img = nib.load(str(nifti_path))
            header_info = dict(nib_img.header)
        except ImportError:
            # Если nibabel не установлен, используем только SimpleITK
            logging.warning("nibabel not installed, using SimpleITK header info only")

        return {
            'image_array': image_array,
            'sitk_image': sitk_image,
            'original_spacing': original_spacing,
            'original_size': original_size,
            'rescale_slope': 1.0,  # NIfTI обычно уже в HU
            'rescale_intercept': 0.0,
            'hu_range': (float(image_array.min()), float(image_array.max())),
            'is_multiframe': False,
            'file_type': 'nifti',
            'header_info': header_info
        }

    except Exception as e:
        raise Exception(f"Failed to load NIfTI file {nifti_path}: {str(e)}")


# -------------------------------
# Определение типа входных данных и загрузка
# -------------------------------
def load_medical_image(input_path: Path) -> dict:
    """Определяет тип медицинского изображения и загружает соответствующим способом"""

    # Проверяем расширение файла для NIfTI
    nifti_extensions = ['.nii', '.nii.gz', '.img', '.hdr']
    dicom_extensions = ['.dcm', '.dicom']

    if input_path.is_file():
        # Один файл - проверяем на NIfTI
        file_suffix_lower = input_path.suffix.lower()
        full_suffix_lower = ''.join(input_path.suffixes).lower()

        if file_suffix_lower in nifti_extensions or full_suffix_lower == '.nii.gz':
            return load_nifti_file(input_path)
        elif file_suffix_lower in dicom_extensions:
            # Один DICOM файл - обрабатываем как многофреймовый
            return load_multiframe_dicom(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

    elif input_path.is_dir():
        # Директория - пробуем загрузить как DICOM серию
        try:
            return load_dicom_series_sitk(input_path)
        except Exception as dicom_error:
            # Если не DICOM, ищем NIfTI файлы в директории
            nifti_files = []
            for ext in nifti_extensions:
                nifti_files.extend(list(input_path.glob(f"*{ext}")))
                nifti_files.extend(list(input_path.glob(f"*{ext.upper()}")))

            if nifti_files:
                if len(nifti_files) > 1:
                    logging.warning(f"Multiple NIfTI files found, using first: {nifti_files[0]}")
                return load_nifti_file(nifti_files[0])
            else:
                raise FileNotFoundError(f"No DICOM series or NIfTI files found in {input_path}")

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


# -------------------------------
# Основной пайплайн обработки (обновленный)
# -------------------------------
def process_medical_volume(input_path: Path) -> torch.Tensor:
    """Основной пайплайн обработки медицинского объема (DICOM или NIfTI)"""

    # 1. Загрузка данных (автоматическое определение формата)
    medical_data = load_medical_image(input_path)
    image_array = medical_data['image_array']  # [D, H, W]
    sitk_image = medical_data['sitk_image']

    # 2. Ресемплинг до изотропного разрешения :cite[2]
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

    return final_tensor, medical_data


# -------------------------------
# Обработка одного пациента (обновленная)
# -------------------------------
def process_single_patient(patient_input: Path, output_dir: Path, verbose: bool = False) -> dict:
    result = {
        'patient_name': patient_input.name,
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
        'std': None,
        'file_type': None
    }

    try:
        # Обработка объема (DICOM или NIfTI)
        final_tensor, medical_data = process_medical_volume(patient_input)

        # Проверка результата
        expected_shape = (1, 1) + TARGET_OUTPUT_SHAPE
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        if torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any():
            raise ValueError("Invalid values (NaN/Inf) in final tensor")

        # Сохранение
        output_path = output_dir / f"{patient_input.name}.pt"
        torch.save(final_tensor, output_path)

        # Заполнение результата
        result.update({
            'success': True,
            'output_path': str(output_path),
            'original_shape': medical_data['image_array'].shape,
            'final_shape': tuple(final_tensor.shape),
            'original_spacing': medical_data['original_spacing'],
            'hu_range_input': medical_data['hu_range'],
            'value_range_output': (float(final_tensor.min()), float(final_tensor.max())),
            'mean': float(final_tensor.mean()),
            'std': float(final_tensor.std()),
            'file_type': medical_data.get('file_type', 'dicom')
        })

        if verbose:
            logging.info(f"✓ {patient_input.name} ({result['file_type']}): "
                         f"original={result['original_shape']}, "
                         f"final={result['final_shape']}, "
                         f"range=[{result['value_range_output'][0]:.3f}, {result['value_range_output'][1]:.3f}]")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            logging.error(f"❌ {patient_input.name}: {e}")
            logging.debug(traceback.format_exc())

    return result


# -------------------------------
# Главная функция (обновленная)
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare CT volumes (DICOM/NIfTI) for MedicalNet with isotropic resampling (1x1x1 mm)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with patient data (DICOM folders or NIfTI files)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/prepare_medical_tensors.log",
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

    # Находим все входные данные (директории и файлы)
    input_items = []
    for item in input_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            input_items.append(item)
        elif item.is_file() and not item.name.startswith('.'):
            # Проверяем, является ли файл медицинским изображением
            medical_extensions = ['.nii', '.nii.gz', '.dcm', '.dicom', '.img', '.hdr']
            if any(item.name.lower().endswith(ext) for ext in medical_extensions):
                input_items.append(item)

    if not input_items:
        logger.error(f"No DICOM directories or NIfTI files found in {input_dir}")
        return

    logger.info(f"Found {len(input_items)} input items (directories/files)")
    logger.info(f"Target output tensor shape: {(1, 1) + TARGET_OUTPUT_SHAPE}")
    logger.info(f"✅ Isotropic resampling to: {TARGET_SPACING} mm")
    logger.info(f"✅ HU range normalization: [{HU_MIN}, {HU_MAX}] → [-1.0, 1.0]")
    logger.info(f"✅ Support for DICOM and NIfTI formats")
    logger.info(f"✅ MedicalNet-compatible preprocessing (MONAI-free)")

    results_summary = {
        'total': len(input_items),
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
    for input_item in tqdm(input_items, desc="Processing medical data"):
        result = process_single_patient(input_item, output_dir, args.verbose)
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
    logger.info(f"Total items: {results_summary['total']}")
    logger.info(f"Successfully processed: {results_summary['successful']}")
    logger.info(f"Failed: {results_summary['failed']}")
    logger.info(f"Success rate: {100 * results_summary['successful'] / results_summary['total']:.1f}%")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"{'=' * 60}")

    if results_summary['failed'] > 0:
        logger.warning(f"⚠️  {results_summary['failed']} items failed. Check the report for details.")

    logger.info("Done! ✅")


if __name__ == "__main__":
    main()