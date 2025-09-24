#!/usr/bin/env python3
"""
prepare_ct_for_medicalnet.py

Подготовка КТ-данных для модели MedicalNet с учетом особенностей предобученной модели.
Возвращает тензоры [1, 1, 128, 128, 128] с правильной нормализацией для ResNet3D.
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
# Конфигурация для MedicalNet
# -------------------------------
TARGET_SHAPE = (128, 128, 128)  # D, H, W для 3D ResNet
# Для CT данных используем изотропное разрешение 1мм³
TARGET_SPACING = (1.0, 1.0, 1.0)  # мм/воксель [D, H, W]
# Стандартные HU диапазоны для CT (расширенные для коррекции)
HU_MIN = -1000  # воздух
HU_MAX = 1000  # плотная кость (увеличено для лучшей коррекции)
# MedicalNet нормализация (НЕ ImageNet!)
# Для медицинских данных обычно используется zero-mean unit-variance
MEDICAL_MEAN = 0.0
MEDICAL_STD = 1.0


# -------------------------------
# Настройка логирования
# -------------------------------
def setup_logging(log_file: Path):
    """Настройка логирования в файл и консоль"""
    log_file.parent.mkdir(parents=True, exist_ok=True)  # Создаем директорию для логов
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
# Трансформации MONAI для MedicalNet
# -------------------------------
def get_transforms():
    """Создание композиции трансформаций, адаптированной для MedicalNet"""
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        # Сначала нормализуем spacing
        Spacing(pixdim=TARGET_SPACING, mode="bilinear", padding_mode="zeros"),
        # Обрезаем фон ПОСЛЕ ресемплинга и ПОСЛЕ коррекции HU
        CropForeground(select_fn=lambda x: x > -800,  # Адаптивный порог после коррекции
                       channel_indices=0, margin=5),
        # Масштабируем HU в диапазон [0, 1] для стабильности
        ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX,
                            b_min=0.0, b_max=1.0, clip=True),
        # Изменяем размер ПОСЛЕ нормализации интенсивности
        Resize(spatial_size=TARGET_SHAPE, mode="trilinear", align_corners=False),
        # Финальная нормализация для нейронной сети (zero-mean, unit-variance)
        NormalizeIntensity(subtrahend=0.5, divisor=0.5),  # Преобразует [0,1] в [-1,1]
        ToTensor()
    ])


# -------------------------------
# Функция для валидации размеров
# -------------------------------
def validate_tensor_size(tensor, min_size=16):
    """Проверяет, что тензор имеет минимальный размер для обработки"""
    if len(tensor.shape) == 3:
        d, h, w = tensor.shape
    elif len(tensor.shape) == 4 and tensor.shape[0] == 1:
        _, d, h, w = tensor.shape
        tensor = tensor.squeeze(0)  # Убираем лишнее измерение
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    if min(d, h, w) < min_size:
        raise ValueError(f"Tensor too small: {(d, h, w)}, minimum size: {min_size}")

    return tensor

def _load_single_dicom_file(single_file: Path) -> MetaTensor:
    """
    Читает одиночный DICOM-файл (multi-frame или single-frame).
    Возвращает Tensor формата (C, D, H, W).
    """
    ds = pydicom.dcmread(str(single_file), force=True)
    num_frames = int(getattr(ds, "NumberOfFrames", 1))

    if num_frames > 1:
        # ---- Multi-frame ----
        pixel_array = ds.pixel_array  # (frames, H, W)
        spacing = [
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[0]),
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[1]),
            float(getattr(ds, "SpacingBetweenSlices", 1.0)),
        ]
        tensor = torch.from_numpy(pixel_array.astype(np.float32))
        tensor = tensor.unsqueeze(0)  # (1, D, H, W)
        return MetaTensor(tensor, affine=np.diag(spacing + [1]))

    else:
        # ---- Single-frame ----
        pixel_array = ds.pixel_array  # (H, W)
        spacing = [
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[0]),
            float(getattr(ds, "PixelSpacing", [1.0, 1.0])[1]),
            1.0,
        ]
        tensor = torch.from_numpy(pixel_array.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return MetaTensor(tensor, affine=np.diag(spacing + [1]))

# -------------------------------
# ЗАГРУЗЧИК DICOM с улучшенной обработкой ошибок
# -------------------------------
# Замените существующую функцию load_dicom_series на эту версию
def load_dicom_series(dicom_folder: Path) -> MetaTensor:
    """
    Загружает DICOM серию и возвращает MetaTensor с правильными метаданными.
    Поддерживает:
      - классическую серию отдельных DICOM-файлов (SimpleITK ImageSeriesReader)
      - мультифреймовый DICOM в одном файле (pydicom, NumberOfFrames > 1)
    """

    try:
        # -------------------------------------------------------------
        # 1️⃣  Если в папке ровно один файл — сразу пробуем мультифрейм
        # -------------------------------------------------------------
        files = sorted([p for p in dicom_folder.iterdir()
                        if p.is_file() and not p.name.startswith('.')])
        if len(files) == 1:
            single = files[0]
            # 🔑 переходим сразу в ветку pydicom_multiframe
            return _load_single_dicom_file(single)  # <- см. реализацию ниже

        # -------------------------------------------------------------
        # 2️⃣  Иначе пробуем обычную серию через SimpleITK
        # -------------------------------------------------------------
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
        if series_IDs:
            # Берем первую серию (обычно основную)
            series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
            if len(series_file_names) < 3:  # Минимум 3 среза для 3D
                raise ValueError(f"Too few DICOM slices: {len(series_file_names)}")

            reader.SetFileNames(series_file_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            sitk_image = reader.Execute()

            # Получаем параметры изображения
            original_size = sitk_image.GetSize()  # [W, H, D]
            original_spacing = sitk_image.GetSpacing()  # [W, H, D] в мм

            # Валидация размеров
            if any(s <= 0 for s in original_size):
                raise ValueError(f"Invalid image dimensions: {original_size}")

            image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # [D, H, W]

            # Попытка получить rescale из первого файла серии
            slope, intercept = 1.0, 0.0
            try:
                first_file = series_file_names[0]
                dcm = pydicom.dcmread(first_file, stop_before_pixels=True)
                slope = float(getattr(dcm, 'RescaleSlope', 1.0))
                intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            except Exception:
                pass

            # spacing в формате [D,H,W]
            spacing_dhw = [original_spacing[2], original_spacing[1], original_spacing[0]]

            meta_dict = {
                'spacing': spacing_dhw,
                'original_shape': image_array.shape,
                'original_spacing': original_spacing,
                'filename_or_obj': str(dicom_folder),
                'rescale_slope': slope,
                'rescale_intercept': intercept,
                'source': 'sitk_series'
            }

            meta_tensor = MetaTensor(image_array, meta=meta_dict)
            logging.debug(f"Loaded SITK series {dicom_folder.name}: shape={image_array.shape}, spacing={spacing_dhw}")
            return meta_tensor

        # Если серии нет — возможно один файл (multi-frame) или просто нераспознанная структура
        # Находим файлы в папке
        files = sorted([p for p in dicom_folder.iterdir() if p.is_file() and not p.name.startswith('.')])
        if not files:
            raise FileNotFoundError(f"No files found in {dicom_folder}")

        # Если в папке ровно один файл — проверим мультифрейм DICOM
        if len(files) == 1:
            single = files[0]
            try:
                ds = pydicom.dcmread(str(single), force=True)
            except Exception as e:
                raise Exception(f"Cannot read DICOM file {single}: {e}")

            num_frames = int(getattr(ds, 'NumberOfFrames', 1))
            # Если мультифреймный DICOM
            if num_frames > 1:
                # Попытка получить pixel array (потребуются соответствующие pixel handlers)
                try:
                    pixel_array = ds.pixel_array  # shape: (frames, rows, cols) обычно
                except Exception as e:
                    raise Exception(f"Could not extract pixel_array from {single}: {e}")

                # Приводим к float32
                image_array = np.asarray(pixel_array).astype(np.float32)

                # Проверяем форму (если есть каналы)
                if image_array.ndim == 4 and image_array.shape[-1] in (1, 3):
                    # Если последняя ось - каналы, трансформируем в (frames, H, W)
                    if image_array.shape[-1] == 1:
                        image_array = image_array[..., 0]
                    else:
                        # Если RGB - берем первый канал (обычно для CT не применяется)
                        image_array = image_array[..., 0]

                # Приводим shape к [D, H, W]
                if image_array.ndim != 3:
                    raise ValueError(f"Unexpected pixel array shape for multi-frame DICOM: {image_array.shape}")

                # Получаем рескейл параметры
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

                # Считываем spacing: PixelSpacing -> [row, col], и SpacingBetweenFrames/SliceThickness
                pixel_spacing = None
                slice_spacing = None

                if hasattr(ds, 'PixelSpacing'):
                    try:
                        ps = ds.PixelSpacing
                        # PixelSpacing обычно [rowSpacing, colSpacing]
                        pixel_spacing = (float(ps[0]), float(ps[1]))
                    except Exception:
                        pixel_spacing = None

                # Способ1: SpacingBetweenFrames
                if hasattr(ds, 'SpacingBetweenFrames'):
                    try:
                        slice_spacing = float(ds.SpacingBetweenFrames)
                    except Exception:
                        slice_spacing = None

                # Способ2: SliceThickness
                if slice_spacing is None and hasattr(ds, 'SliceThickness'):
                    try:
                        slice_spacing = float(ds.SliceThickness)
                    except Exception:
                        slice_spacing = None

                # Попытка вычислить spacing между кадрами по Per-frame Functional Groups Sequence
                if slice_spacing is None and hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
                    try:
                        seq = ds.PerFrameFunctionalGroupsSequence
                        if len(seq) >= 2:
                            ipp0 = None
                            ipp1 = None
                            # Некоторые DICOM используют ImagePositionPatient внутри FrameContentSequence -> PlanePositionSequence
                            # Но стандартнее в PerFrameFunctionalGroupsSequence/.../PlanePositionSequence/ImagePositionPatient
                            def get_ipp(frame_item):
                                try:
                                    if hasattr(frame_item, 'PlanePositionSequence'):
                                        pps = frame_item.PlanePositionSequence
                                        return [float(x) for x in pps[0].ImagePositionPatient]
                                    # Newer structure
                                    if hasattr(frame_item, 'FrameContentSequence') and hasattr(frame_item.FrameContentSequence[0], 'PlanePositionSequence'):
                                        pps = frame_item.FrameContentSequence[0].PlanePositionSequence
                                        return [float(x) for x in pps[0].ImagePositionPatient]
                                except Exception:
                                    return None
                                return None

                            ipp0 = get_ipp(seq[0]) or None
                            ipp1 = get_ipp(seq[1]) or None
                            if ipp0 is not None and ipp1 is not None:
                                # z spacing = abs(z1 - z0) (предполагается согласованность направления)
                                slice_spacing = abs(ipp1[2] - ipp0[2])
                    except Exception:
                        slice_spacing = None

                # Если ничего не найдено — поставим 1.0mm (безопасный дефолт)
                if pixel_spacing is None:
                    logging.warning(f"No PixelSpacing found in {single}, defaulting to (1.0, 1.0)")
                    pixel_spacing = (1.0, 1.0)
                if slice_spacing is None:
                    logging.warning(f"No slice spacing found in {single}, defaulting to 1.0 mm")
                    slice_spacing = 1.0

                # Применяем rescale (HU)
                if slope != 1.0 or intercept != 0.0:
                    image_array = image_array * slope + intercept

                # Формируем spacing в формате [D, H, W]
                spacing_dhw = [slice_spacing, pixel_spacing[0], pixel_spacing[1]]

                # Финальная проверка
                hu_min, hu_max = float(image_array.min()), float(image_array.max())
                if hu_min == hu_max:
                    raise ValueError(f"Constant image values in {single}: {hu_min}")

                meta_dict = {
                    'spacing': spacing_dhw,
                    'original_shape': image_array.shape,
                    'original_spacing': spacing_dhw,
                    'filename_or_obj': str(single),
                    'rescale_slope': slope,
                    'rescale_intercept': intercept,
                    'hu_range': (hu_min, hu_max),
                    'source': 'pydicom_multiframe',
                    'num_frames': num_frames
                }

                meta_tensor = MetaTensor(image_array, meta=meta_dict)
                logging.debug(f"Loaded multi-frame DICOM {single.name}: frames={num_frames}, shape={image_array.shape}, spacing={spacing_dhw}")
                return meta_tensor

            # Если single файл, но NumberOfFrames == 1 — возможно обычный single-slice DICOM; но для 3D нужен набор
            # Попробуем обработать как серия из одного среза (вернём 1-срезный том)
            else:
                try:
                    pixel_array = ds.pixel_array.astype(np.float32)
                except Exception as e:
                    raise Exception(f"Could not extract pixel_array from single-frame DICOM {single}: {e}")

                # Сделаем shape [1, H, W]
                if pixel_array.ndim == 2:
                    image_array = np.expand_dims(pixel_array, axis=0).astype(np.float32)
                elif pixel_array.ndim == 3:
                    image_array = pixel_array.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected pixel array shape for single file: {pixel_array.shape}")

                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    image_array = image_array * slope + intercept

                pixel_spacing = (1.0, 1.0)
                if hasattr(ds, 'PixelSpacing'):
                    try:
                        ps = ds.PixelSpacing
                        pixel_spacing = (float(ps[0]), float(ps[1]))
                    except Exception:
                        pass
                slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))

                spacing_dhw = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
                hu_min, hu_max = float(image_array.min()), float(image_array.max())

                meta_dict = {
                    'spacing': spacing_dhw,
                    'original_shape': image_array.shape,
                    'original_spacing': spacing_dhw,
                    'filename_or_obj': str(single),
                    'rescale_slope': slope,
                    'rescale_intercept': intercept,
                    'hu_range': (hu_min, hu_max),
                    'source': 'pydicom_singlefile'
                }

                return MetaTensor(image_array, meta=meta_dict)

        # Если в папке >1 файл, но SimpleITK не вернул series_IDs (редкий кейс) — попробуем простую сборку по сортировке по InstanceNumber
        # Поддержка: читаем все файлы, сортируем по InstanceNumber или по filename, собираем массив
        dicom_files = []
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                inst = getattr(ds, 'InstanceNumber', None)
                dicom_files.append((p, inst))
            except Exception:
                # если чтение метаданных не получилось — добавим в конец
                dicom_files.append((p, None))
        # сортируем по InstanceNumber если возможно
        dicom_files_sorted = sorted(dicom_files, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0, str(x[0])))
        series_file_names = [str(x[0]) for x in dicom_files_sorted]

        # Если мало файлов — ошибка
        if len(series_file_names) < 3:
            raise ValueError(f"Too few DICOM slices (fallback aggregation): {len(series_file_names)}")

        # Используем SITK читатель на этой упорядоченной последовательности
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        sitk_image = reader.Execute()
        image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # [D,H,W]
        original_spacing = sitk_image.GetSpacing()  # [W,H,D]
        spacing_dhw = [original_spacing[2], original_spacing[1], original_spacing[0]]

        # Попытка взять rescale из первого файла
        slope, intercept = 1.0, 0.0
        try:
            dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        except Exception:
            pass

        if slope != 1.0 or intercept != 0.0:
            image_array = image_array * slope + intercept

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
    """
    Обработка одного пациента с детальным логированием
    """
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
        # 1. Загружаем DICOM
        ct_meta_tensor = load_dicom_series(patient_dir)
        original_shape = ct_meta_tensor.shape
        result['original_shape'] = original_shape
        result['original_spacing'] = ct_meta_tensor.meta.get('spacing')
        result['hu_range_input'] = ct_meta_tensor.meta.get('hu_range')

        if verbose:
            logging.info(f"Loaded {patient_dir.name}: {original_shape}")

        # 2. Валидация размера
        validated_tensor = validate_tensor_size(ct_meta_tensor)

        # 3. Применяем трансформации
        transform = get_transforms()
        transformed_tensor = transform(validated_tensor)

        # 4. Добавляем batch dimension
        final_tensor = transformed_tensor.unsqueeze(0)  # [1, 1, 128, 128, 128]

        # 5. Финальная валидация
        expected_shape = (1, 1, 128, 128, 128)
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        # Проверка на валидность данных
        if torch.isnan(final_tensor).any():
            raise ValueError("NaN values in final tensor")

        if torch.isinf(final_tensor).any():
            raise ValueError("Infinite values in final tensor")

        # 6. Сохраняем
        output_path = output_dir / f"{patient_dir.name}.pt"
        torch.save(final_tensor, output_path)

        # 7. Заполняем результат
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
        description="Prepare CT DICOM volumes for MedicalNet 3D ResNet models")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with patient subdirectories containing DICOM files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging with detailed progress")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_medicalnet.log",
                        help="Log file path")

    args = parser.parse_args()

    # Настройка директорий и логирования
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file)
    logger = setup_logging(log_file)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Получаем список папок пациентов
    patient_dirs = [d for d in input_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')]

    if not patient_dirs:
        logger.error(f"No patient directories found in {input_dir}")
        return

    logger.info(f"Found {len(patient_dirs)} patient directories")
    logger.info(f"Target tensor shape: {(1, 1) + TARGET_SHAPE}")
    logger.info(f"Target spacing: {TARGET_SPACING} mm")

    # Инициализация отчета
    results_summary = {
        'total': len(patient_dirs),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'config': {
            'target_shape': TARGET_SHAPE,
            'target_spacing': TARGET_SPACING,
            'hu_range': [HU_MIN, HU_MAX],
            'normalization': 'zero_mean_unit_variance'
        }
    }

    # Обработка (последовательно для отладки)
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

    # Сохранение отчета
    report_file = output_dir / "processing_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Итоговый отчет
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