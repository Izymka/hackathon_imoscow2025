#!/usr/bin/env python3
"""
prepare_ct_for_medicalnet.py

Использование вашего старого, рабочего DICOM-загрузчика + MONAI-трансформаций.
Возвращает тензоры [1, 1, 128, 128, 128] для model с учётом реального размера вокселя.
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
    Spacing,
    ScaleIntensityRange,
    CropForeground,
    Resize,
    NormalizeIntensity,
    ToTensor,
    Compose
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
import json

# -------------------------------
# Конфигурация
# -------------------------------
TARGET_SHAPE = (128, 128, 128)  # D, H, W
TARGET_SPACING = (1.0, 1.0, 1.0)  # мм/воксель [D, H, W]
HU_MIN = -1000
HU_MAX = 1000
IMAGE_NET_MEAN = 0.485
IMAGE_NET_STD = 0.229


# -------------------------------
# Настройка логирования
# -------------------------------
def setup_logging(log_file: Path):
    """Настройка логирования в файл и консоль"""
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
# Трансформации MONAI с учётом_spacing
# -------------------------------
def get_transforms():
    """Создание композиции трансформаций"""
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Spacing(pixdim=TARGET_SPACING, mode="bilinear", padding_mode="zeros"),
        ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX,
                            b_min=0, b_max=1, clip=True),
        CropForeground(select_fn=lambda x: x > 0.05,
                       margin=10),
        Resize(spatial_size=TARGET_SHAPE,
               mode="trilinear", align_corners=False),
        #NormalizeIntensity(subtrahend=IMAGE_NET_MEAN,
        #                   divisor=IMAGE_NET_STD),
        ToTensor()
    ])


# -------------------------------
# ЗАГРУЗЧИК DICOM с учётом_spacing
# -------------------------------
def load_dicom_series(dicom_folder: Path) -> MetaTensor:
    """
    Загружает DICOM серию и возвращает MetaTensor с метаданными о spacing
    Возвращает: MetaTensor с формой [D, H, W] и метаданными
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
        if not series_IDs:
            raise FileNotFoundError(f"No DICOM series found in {dicom_folder}")

        series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
        reader.SetFileNames(series_file_names)
        sitk_image = reader.Execute()

        # Получаем spacing из метаданных (SimpleITK возвращает [x, y, z] = [W, H, D])
        original_spacing = sitk_image.GetSpacing()  # [W, H, D] в мм
        spacing_reordered = [original_spacing[2], original_spacing[1], original_spacing[0]]  # [D, H, W]

        # Преобразуем в numpy
        image_array = sitk.GetArrayFromImage(sitk_image)  # [D, H, W]
        image_array = image_array.astype(np.float32)

        # Проверка на пустой массив
        if image_array.size == 0:
            raise ValueError(f"Loaded DICOM array is empty! Folder: {dicom_folder}")

        # Получаем RescaleSlope и RescaleIntercept из метаданных SimpleITK
        # Пытаемся получить из первого файла серии
        try:
            first_file = series_file_names[0]
            dcm = pydicom.dcmread(first_file, stop_before_pixels=True)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        except Exception:
            # Если не удалось прочитать через pydicom, пробуем через SimpleITK
            try:
                slope = float(sitk_image.GetMetaData('0028|1053'))  # RescaleSlope
                intercept = float(sitk_image.GetMetaData('0028|1052'))  # RescaleIntercept
            except Exception:
                slope = 1.0
                intercept = 0.0

        if slope != 1.0 or intercept != 0.0:
            image_array = image_array * slope + intercept

        # Создаём MetaTensor правильно
        meta_dict = {
            'spacing': spacing_reordered,
            'original_shape': image_array.shape,
            'filename_or_obj': str(dicom_folder)
        }

        meta_tensor = MetaTensor(image_array, meta=meta_dict)
        return meta_tensor

    except Exception as e:
        raise Exception(f"Error loading DICOM series from {dicom_folder}: {str(e)}")


# -------------------------------
# Обработка одного пациента
# -------------------------------
def process_single_patient(patient_dir: Path, output_dir: Path, verbose: bool = False) -> dict:
    """
    Обработка одного пациента - загрузка, трансформация и сохранение
    Возвращает словарь с результатом обработки
    """
    result = {
        'patient_name': patient_dir.name,
        'success': False,
        'error': None,
        'output_path': None,
        'final_shape': None,
        'value_range': None,
        'mean': None,
        'std': None
    }

    try:
        # Загружаем данные как MetaTensor с метаданными
        ct_meta_tensor = load_dicom_series(patient_dir)

        # Применяем MONAI трансформации с учётом_spacing
        transform = get_transforms()
        tensor = transform(ct_meta_tensor)  # результат: [1, 128, 128, 128]
        tensor = tensor.unsqueeze(0)  # → [1, 1, 128, 128, 128] для model

        # Проверяем финальную форму
        if tensor.shape != (1, 1, 128, 128, 128):
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        # Сохраняем результат
        output_path = output_dir / f"{patient_dir.name}.pt"
        torch.save(tensor, output_path)

        # Заполняем результат
        result['success'] = True
        result['output_path'] = str(output_path)
        result['final_shape'] = tuple(tensor.shape)
        result['value_range'] = (float(tensor.min()), float(tensor.max()))
        result['mean'] = float(tensor.mean())
        result['std'] = float(tensor.std())

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            logging.error(f"Error processing {patient_dir.name}: {e}")
            logging.debug(traceback.format_exc())

    return result


# -------------------------------
# Главная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare DICOM CT volumes for model with real voxel size consideration")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to folder with patient subfolders (each with DICOM series)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory for .pt files")
    parser.add_argument("--verbose", action="store_true", help="Print progress and tensor stats")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--log-file", type=str, default="processing.log",
                        help="Log file path (default: processing.log)")

    args = parser.parse_args()

    # Настройка логирования
    log_file = Path(args.log_file)
    logger = setup_logging(log_file)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список пациентов
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patient folders in {input_dir}")

    # Файл для записи ошибок
    errors_file = output_dir / "processing_errors.json"
    results_summary = {
        'total': len(patient_dirs),
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    # Параллельная обработка
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Создаем задачи
        future_to_patient = {
            executor.submit(process_single_patient, patient_dir, output_dir, args.verbose): patient_dir
            for patient_dir in patient_dirs
        }

        # Обрабатываем результаты с прогресс-баром
        with tqdm(total=len(patient_dirs), desc="Processing patients") as pbar:
            for future in as_completed(future_to_patient):
                patient_dir = future_to_patient[future]
                try:
                    result = future.result()
                    if result['success']:
                        results_summary['successful'] += 1
                        if args.verbose:
                            logger.info(f"✓ Processed {result['patient_name']}")
                            logger.info(f"  Shape: {result['final_shape']}")
                            logger.info(f"  Range: {result['value_range']}")
                            logger.info(f"  Mean: {result['mean']:.3f}, Std: {result['std']:.3f}")
                    else:
                        results_summary['failed'] += 1
                        error_info = {
                            'patient': result['patient_name'],
                            'error': result['error']
                        }
                        results_summary['errors'].append(error_info)
                        logger.error(f"❌ Failed {result['patient_name']}: {result['error']}")
                except Exception as e:
                    results_summary['failed'] += 1
                    error_info = {
                        'patient': patient_dir.name,
                        'error': str(e)
                    }
                    results_summary['errors'].append(error_info)
                    logger.error(f"❌ Exception processing {patient_dir.name}: {e}")
                finally:
                    pbar.update(1)

    # Сохраняем отчет об ошибках
    with open(errors_file, 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Выводим итоговый отчет
    logger.info(f"\n{'=' * 50}")
    logger.info("PROCESSING SUMMARY:")
    logger.info(f"Total patients: {results_summary['total']}")
    logger.info(f"Successful: {results_summary['successful']}")
    logger.info(f"Failed: {results_summary['failed']}")
    logger.info(f"Errors log saved to: {errors_file}")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'=' * 50}")

    if results_summary['failed'] > 0:
        logger.warning(f"⚠️  {results_summary['failed']} patients failed processing. Check errors log for details.")

    logger.info("✅ Done!")


if __name__ == "__main__":
    main()