#!/usr/bin/env python3
"""

Подготовка КТ-данных для модели MedicalNet с изотропным ресемплингом до 1x1x1 мм.
Возвращает тензоры [1, 1, 256, 256, 256] с фиксированной нормализацией [-1000, 600] → [-1, 1].
Полностью без MONAI - использует только SimpleITK, NumPy и PyTorch.
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import psutil
import pydicom
import torch
import torch.nn.functional as F


# -------------------------------
# Утилита для мониторинга памяти
# -------------------------------
def get_memory_info():
    """Получение информации о потреблении памяти процессом"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size в MB
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size в MB
    }


def log_memory(logger, message: str, debug: bool = False):
    """Логирование информации о памяти"""
    if debug:
        mem = get_memory_info()
        logger.debug(f"{message} | Memory: RSS={mem['rss_mb']:.2f}MB, VMS={mem['vms_mb']:.2f}MB")


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
def setup_logging(log_file: Path, debug: bool = False):
    """Настройка логирования в файл"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if debug else logging.NullHandler()
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
def load_multiframe_dicom(dicom_file: Path, debug: bool = False) -> dict:
    """Загрузка многофреймового DICOM файла"""
    logger = logging.getLogger(__name__)
    try:
        log_memory(logger, f"[load_multiframe_dicom] Start loading {dicom_file.name}", debug)

        ds = pydicom.dcmread(str(dicom_file), force=True)
        log_memory(logger, f"[load_multiframe_dicom] DICOM loaded", debug)

        # Получаем pixel array (используем float16 для экономии памяти)
        pixel_array = ds.pixel_array.astype(np.float16)
        log_memory(logger, f"[load_multiframe_dicom] Pixel array extracted, shape={pixel_array.shape}, dtype={pixel_array.dtype}", debug)

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
        hu_min, hu_max = float(pixel_array.min()), float(pixel_array.max())
        original_size = pixel_array.shape[::-1]

        # Создаем SimpleITK изображение из numpy array
        sitk_image = sitk.GetImageFromArray(pixel_array)
        sitk_image.SetSpacing(spacing[::-1])

        # Освобождаем DICOM dataset
        del ds
        gc.collect()
        log_memory(logger, f"[load_multiframe_dicom] After cleanup", debug)

        return {
            'image_array': pixel_array,
            'sitk_image': sitk_image,
            'original_spacing': spacing[::-1],
            'original_size': original_size,
            'rescale_slope': slope,
            'rescale_intercept': intercept,
            'hu_range': (hu_min, hu_max),
            'is_multiframe': True
        }

    except Exception as e:
        raise Exception(f"Failed to load multiframe DICOM {dicom_file}: {str(e)}")


def load_dicom_series_sitk(dicom_folder: Path, debug: bool = False) -> dict:
    """Загрузка DICOM серии через SimpleITK с поддержкой многофреймовых файлов"""
    logger = logging.getLogger(__name__)
    try:
        log_memory(logger, f"[load_dicom_series] Start loading {dicom_folder.name}", debug)

        files = sorted([p for p in dicom_folder.iterdir()
                        if p.is_file() and not p.name.startswith('.')])

        if not files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_folder}")

        # Проверяем первый файл на многофреймовость
        first_file = files[0]
        try:
            ds_test = pydicom.dcmread(str(first_file), stop_before_pixels=True)
            num_frames = int(getattr(ds_test, "NumberOfFrames", 1))
            del ds_test

            if num_frames > 1:
                return load_multiframe_dicom(first_file, debug)
        except Exception:
            pass

        # Обычная загрузка серии файлов
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))

        if not series_IDs:
            series_file_names = [str(f) for f in files]
            if len(series_file_names) == 1:
                try:
                    ds_single = pydicom.dcmread(series_file_names[0], force=True)
                    num_frames_single = int(getattr(ds_single, "NumberOfFrames", 1))
                    del ds_single
                    if num_frames_single > 1:
                        return load_multiframe_dicom(first_file, debug)
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
        log_memory(logger, f"[load_dicom_series] Before reader.Execute()", debug)
        sitk_image = reader.Execute()
        log_memory(logger, f"[load_dicom_series] After reader.Execute()", debug)

        # Получаем данные (используем float16 для экономии памяти)
        image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float16)
        log_memory(logger, f"[load_dicom_series] Image array extracted, shape={image_array.shape}, dtype={image_array.dtype}", debug)
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()

        # Получаем параметры рескейла из первого файла
        rescale_slope, rescale_intercept = 1.0, 0.0
        try:
            dcm = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
            rescale_slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            rescale_intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            del dcm
        except Exception:
            pass

        # Применяем рескейл если нужно
        if rescale_slope != 1.0 or rescale_intercept != 0.0:
            image_array = image_array * rescale_slope + rescale_intercept

        hu_min, hu_max = float(image_array.min()), float(image_array.max())

        del reader
        gc.collect()
        log_memory(logger, f"[load_dicom_series] After cleanup", debug)

        return {
            'image_array': image_array,
            'sitk_image': sitk_image,
            'original_spacing': original_spacing,
            'original_size': original_size,
            'rescale_slope': rescale_slope,
            'rescale_intercept': rescale_intercept,
            'hu_range': (hu_min, hu_max),
            'is_multiframe': False
        }

    except Exception as e:
        raise Exception(f"Failed to load DICOM from {dicom_folder}: {str(e)}")


# -------------------------------
# Основной пайплайн обработки
# -------------------------------
def process_ct_volume(dicom_folder: Path, debug: bool = False) -> torch.Tensor:
    """Основной пайплайн обработки КТ объема"""
    logger = logging.getLogger(__name__)
    log_memory(logger, f"[process_ct_volume] START for {dicom_folder.name}", debug)

    # 1. Загрузка DICOM
    dicom_data = load_dicom_series_sitk(dicom_folder, debug)
    image_array = dicom_data['image_array']
    sitk_image = dicom_data['sitk_image']

    if debug:
        logger.debug(f"[process_ct_volume] Loaded volume shape: {image_array.shape}, "
                     f"spacing: {dicom_data['original_spacing']}, "
                     f"HU range: {dicom_data['hu_range']}")
    log_memory(logger, f"[process_ct_volume] After DICOM loading", debug)

    # 2. Ресемплинг до изотропного разрешения
    target_spacing_xyz = TARGET_SPACING

    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    target_size_xyz = [
        int(round(original_size[0] * original_spacing[0] / target_spacing_xyz[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing_xyz[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing_xyz[2]))
    ]

    target_size_xyz = [max(1, dim) for dim in target_size_xyz]

    resampled_sitk = resample_image_sitk(
        sitk_image,
        target_spacing_xyz,
        target_size_xyz,
        is_label=False
    )

    resampled_array = sitk.GetArrayFromImage(resampled_sitk)

    if debug:
        logger.debug(f"[process_ct_volume] Resampled shape: {resampled_array.shape}")
    log_memory(logger, f"[process_ct_volume] After resampling", debug)

    # Освобождаем память
    del sitk_image, resampled_sitk
    gc.collect()
    log_memory(logger, f"[process_ct_volume] After cleanup #1 (sitk objects)", debug)

    # 3. Кроп области с данными
    cropped_array = crop_foreground_3d(resampled_array, threshold=-1000, margin=5)

    if debug:
        logger.debug(f"[process_ct_volume] Cropped shape: {cropped_array.shape}")
    log_memory(logger, f"[process_ct_volume] After cropping", debug)

    del resampled_array
    gc.collect()
    log_memory(logger, f"[process_ct_volume] After cleanup #2 (resampled array)", debug)

    # 4. Нормализация интенсивности HU -> [-1, 1]
    normalized_array = scale_intensity_range(
        cropped_array,
        input_min=HU_MIN,
        input_max=HU_MAX,
        output_min=-1.0,
        output_max=1.0,
        clip=True
    )

    if debug:
        logger.debug(
            f"[process_ct_volume] Normalized range: [{normalized_array.min():.4f}, {normalized_array.max():.4f}]")
    log_memory(logger, f"[process_ct_volume] After normalization", debug)

    del cropped_array
    gc.collect()
    log_memory(logger, f"[process_ct_volume] After cleanup #3 (cropped array)", debug)

    # 5. Конвертация в тензор и ресайз до финального размера
    # Используем float16 для экономии памяти в промежуточных операциях
    tensor = torch.from_numpy(normalized_array).half()
    log_memory(logger, f"[process_ct_volume] After tensor conversion (float16)", debug)

    del normalized_array
    gc.collect()
    log_memory(logger, f"[process_ct_volume] After cleanup #4 (normalized array)", debug)

    # Ресайз до целевого размера
    resized_tensor = resize_3d_tensor(tensor, TARGET_OUTPUT_SHAPE, mode='trilinear')

    if debug:
        logger.debug(f"[process_ct_volume] Resized tensor shape: {resized_tensor.shape}, dtype: {resized_tensor.dtype}")
    log_memory(logger, f"[process_ct_volume] After resizing", debug)

    del tensor
    gc.collect()
    log_memory(logger, f"[process_ct_volume] After cleanup #5 (original tensor)", debug)

    # Убедимся, что размерность правильная [1, 1, D, H, W]
    if resized_tensor.ndim == 5:
        final_tensor = resized_tensor
    elif resized_tensor.ndim == 4:
        final_tensor = resized_tensor.unsqueeze(0)
    else:
        final_tensor = resized_tensor.unsqueeze(0).unsqueeze(0)

    # Конвертируем обратно в float32 для сохранения и совместимости с моделями
    final_tensor = final_tensor.float()

    if debug:
        logger.debug(f"[process_ct_volume] Final tensor shape: {final_tensor.shape}, "
                     f"dtype: {final_tensor.dtype}, "
                     f"range: [{final_tensor.min():.4f}, {final_tensor.max():.4f}]")
    log_memory(logger, f"[process_ct_volume] END", debug)

    return final_tensor, dicom_data


# -------------------------------
# Обработка одного пациента
# -------------------------------
def process_single_patient(patient_dir: Path, output_dir: Path, debug: bool = False) -> dict:
    logger = logging.getLogger(__name__)
    result = {
        'patient_name': patient_dir.name,
        'success': False,
        'error': None,
        'output_path': None
    }

    try:
        log_memory(logger, f"[process_patient] START patient {patient_dir.name}", debug)

        # Обработка объема
        final_tensor, dicom_data = process_ct_volume(patient_dir, debug)
        log_memory(logger, f"[process_patient] After process_ct_volume", debug)

        # Проверка результата
        expected_shape = (1, 1) + TARGET_OUTPUT_SHAPE
        if final_tensor.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {final_tensor.shape}, expected: {expected_shape}")

        if torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any():
            raise ValueError("Invalid values (NaN/Inf) in final tensor")

        # Сохранение
        output_path = output_dir / f"{patient_dir.name}.pt"
        torch.save(final_tensor, output_path)
        log_memory(logger, f"[process_patient] After saving tensor", debug)

        result['success'] = True
        result['output_path'] = str(output_path)

        # Освобождаем память
        del final_tensor, dicom_data
        gc.collect()
        log_memory(logger, f"[process_patient] After final cleanup for {patient_dir.name}", debug)

    except Exception as e:
        result['error'] = str(e)
        logging.error(f"Failed: {patient_dir.name}: {e}")

    return result


# -------------------------------
# Основная функция для использования как библиотеки
# -------------------------------
def prepare_ct_tensor(
    input_dir,
    output_dir,
    log_file=None,
    debug=False,
    force_cpu=False,
    verbose=True
):
    """
    Подготовка КТ-данных для модели MedicalNet с изотропным ресемплингом до 1x1x1 мм.

    Args:
        input_dir (str or Path): Входная директория с поддиректориями пациентов, содержащими DICOM файлы
        output_dir (str or Path): Выходная директория для сохранения обработанных .pt тензоров
        log_file (str or Path, optional): Путь к лог-файлу. Если None, логирование будет только в консоль
        debug (bool): Включить режим отладки с подробным логированием использования памяти
        force_cpu (bool): Принудительно использовать CPU вместо CUDA
        verbose (bool): Выводить прогресс в консоль

    Returns:
        dict: Словарь с результатами обработки:
            - 'total': общее количество пациентов
            - 'successful': количество успешно обработанных
            - 'failed': количество неудачных
            - 'errors': список ошибок с деталями
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Настройка логирования
    if log_file is not None:
        log_file = Path(log_file)
        logger = setup_logging(log_file, debug=debug)
    else:
        logger = logging.getLogger()

    if debug:
        logger.info("=" * 80)
        logger.info("DEBUG MODE ENABLED - Detailed memory monitoring active")
        logger.info("=" * 80)

    if not input_dir.exists():
        error_msg = f"Input directory does not exist: {input_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if force_cpu:
        # Отключить CUDA, если она доступна
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
        if verbose or debug:
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if verbose or debug:
            logger.info("Using CPU device")
    del device

    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not patient_dirs:
        dicom_files = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        if dicom_files:
            patient_dirs = [input_dir]
        else:
            error_msg = f"No patient directories or DICOM files found in {input_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    results_summary = {
        'total': len(patient_dirs),
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    log_memory(logger, "[prepare_ct_tensor] Before processing patients", debug)
    if verbose:
        print(f"Processing {len(patient_dirs)} patients...")

    for idx, patient_dir in enumerate(patient_dirs, 1):
        if debug:
            logger.debug(f"\n{'=' * 60}")
            logger.debug(f"Processing patient {idx}/{len(patient_dirs)}: {patient_dir.name}")
            logger.debug(f"{'=' * 60}")

        result = process_single_patient(patient_dir, output_dir, debug)

        if result['success']:
            results_summary['successful'] += 1
        else:
            results_summary['failed'] += 1
            results_summary['errors'].append({
                'patient': result['patient_name'],
                'error': result['error']
            })

        # Периодическая принудительная сборка мусора
        if idx % 10 == 0:
            gc.collect()
            log_memory(logger, f"[prepare_ct_tensor] After processing {idx} patients (periodic GC)", debug)
            if verbose:
                print(f"Processed {idx}/{len(patient_dirs)}")

    log_memory(logger, "[prepare_ct_tensor] After processing all patients", debug)
    if verbose:
        print(f"\nCompleted: {results_summary['successful']}/{results_summary['total']} successful")

    if debug:
        logger.info("=" * 80)
        logger.info("DEBUG MODE COMPLETED")
        logger.info("=" * 80)

    return results_summary


# -------------------------------
# CLI обертка для запуска из командной строки
# -------------------------------
def main():
    """CLI обертка для prepare_ct_tensor"""
    parser = argparse.ArgumentParser(
        description="Prepare CT DICOM volumes for MedicalNet with isotropic resampling (1x1x1 mm)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with patient subdirectories containing DICOM files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for processed .pt tensor files")
    parser.add_argument("--log-file", type=str, default="logs/prepare_ct_tensors_medicalnet.log",
                        help="Log file path")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with detailed memory usage logging")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU Usage")

    args = parser.parse_args()

    try:
        results = prepare_ct_tensor(
            input_dir=args.input,
            output_dir=args.output,
            log_file=args.log_file,
            debug=args.debug,
            force_cpu=args.force_cpu,
            verbose=True
        )

        if results['failed'] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
