import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom import dcmread
from typing import Union, Tuple, List, Dict
import warnings


plt.style.use('default')


def apply_dicom_rescaling(pixel_array, ds):
    """
    Применяет rescale intercept и slope к DICOM pixel array.
    Поддерживает Enhanced CT и конвертирует DSfloat → float.
    """
    # Enhanced CT: ищем в SharedFunctionalGroupsSequence
    if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
        fg = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(fg, 'CTPixelValueRescaleSequence') and len(fg.CTPixelValueRescaleSequence) > 0:
            rs = fg.CTPixelValueRescaleSequence[0]
            rescale_intercept = float(getattr(rs, 'RescaleIntercept', 0))  # ← float()
            rescale_slope = float(getattr(rs, 'RescaleSlope', 1))          # ← float()
            return pixel_array * rescale_slope + rescale_intercept

    # Стандартный CT
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0))  # ← float()
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1))          # ← float()

    return pixel_array * rescale_slope + rescale_intercept


def apply_ct_window(image, window_center, window_width):
    """
    Применяет CT window к изображению в HU.
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    windowed_image = np.clip(image, window_min, window_max)
    windowed_image = (windowed_image - window_min) / (window_max - window_min) * 255.0
    return windowed_image.astype(np.uint8)


def get_dicom_window_params(ds):
    """
    Извлекает параметры окна. Поддерживает Enhanced CT.
    """
    if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
        fg = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(fg, 'CTWindowSequence') and len(fg.CTWindowSequence) > 0:
            ws = fg.CTWindowSequence[0]
            wc = getattr(ws, 'WindowCenter', 40)
            ww = getattr(ws, 'WindowWidth', 400)
            if isinstance(wc, (list, np.ndarray)): wc = wc[0]
            if isinstance(ww, (list, np.ndarray)): ww = ww[0]
            return float(wc), float(ww)

    window_center = getattr(ds, 'WindowCenter', None)
    window_width = getattr(ds, 'WindowWidth', None)

    if window_center is not None and window_width is not None:
        if isinstance(window_center, (list, np.ndarray)): window_center = window_center[0]
        if isinstance(window_width, (list, np.ndarray)): window_width = window_width[0]
        return float(window_center), float(window_width)

    return 40, 400


def get_pixel_data(ds):
    """
    Универсальная функция для извлечения pixel_array и количества фреймов.
    Гарантированно возвращает num_frames как int.
    """
    if not hasattr(ds, 'PixelData'):
        raise ValueError("PixelData отсутствует в DICOM файле")

    # Извлекаем NumberOfFrames и конвертируем в int
    num_frames = 1
    if hasattr(ds, 'NumberOfFrames'):
        try:
            # Поддержка DSfloat, строк, чисел
            num_frames = int(float(ds.NumberOfFrames))  # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        except (ValueError, TypeError):
            num_frames = 1

    pixel_array = ds.pixel_array
    return pixel_array, num_frames


def show_enhanced_ct_frames(
        file_path: str,
        step: int = 1,
        grid_rows: int = 4,
        grid_cols: int = 4,
        apply_rescaling: bool = True,
        apply_windowing: bool = True,
        start_frame: int = 0,
        max_frames: int = None
):
    try:
        ds = dcmread(file_path, force=True)
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return

    if not hasattr(ds, 'PixelData'):
        print("Файл не содержит PixelData")
        return

    # ИСПРАВЛЕНО: используем get_pixel_data для корректного num_frames
    pixel_data, num_frames = get_pixel_data(ds)

    # Больше не проверяем num_frames == 1 отдельно — работаем со всеми случаями
    print(f"Файл содержит {num_frames} фреймов.")

    if len(pixel_data.shape) == 3:
        if pixel_data.shape[0] == num_frames:
            frame_axis = 0
        elif pixel_data.shape[2] == num_frames:
            frame_axis = 2
        else:
            print(f"Неожиданная форма массива: {pixel_data.shape}. Предполагаем ось 0.")
            frame_axis = 0
    else:
        print(f"Неожиданная размерность массива: {pixel_data.shape}. Отображаем как есть.")
        pixel_data = [pixel_data]
        num_frames = 1
        frame_axis = None

    if apply_rescaling:
        pixel_data = apply_dicom_rescaling(pixel_data, ds)

    window_center, window_width = get_dicom_window_params(ds)

    end_frame = min(start_frame + max_frames, num_frames) if max_frames else num_frames
    frame_indices = list(range(start_frame, end_frame, step))
    if not frame_indices:
        print("Нет фреймов для отображения с заданными параметрами.")
        return

    images_per_page = grid_rows * grid_cols
    total_pages = math.ceil(len(frame_indices) / images_per_page)

    print(f"Отображаем {len(frame_indices)} фреймов из {num_frames} (шаг {step}) на {total_pages} страницах.")

    for page in range(total_pages):
        plt.figure(figsize=(grid_cols * 3, grid_rows * 3))
        plt.suptitle(
            f'Файл: {os.path.basename(file_path)} | Страница {page + 1}/{total_pages} | '
            f'Фреймы {start_frame}-{end_frame-1} шаг {step} | Rescale: {apply_rescaling}',
            fontsize=12
        )

        for i in range(images_per_page):
            idx_in_list = page * images_per_page + i
            if idx_in_list >= len(frame_indices):
                break

            frame_idx = frame_indices[idx_in_list]

            if frame_axis == 0:
                display_image = pixel_data[frame_idx]
            elif frame_axis == 2:
                display_image = pixel_data[:, :, frame_idx]
            else:
                display_image = pixel_data[0]

            if apply_windowing:
                display_image = apply_ct_window(display_image, window_center, window_width)
                cmap = 'gray'
            else:
                cmap = 'viridis'

            ax = plt.subplot(grid_rows, grid_cols, i + 1)
            ax.imshow(display_image, cmap=cmap)
            ax.set_title(f"Фрейм {frame_idx}", fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def show_dicom_pages(
        folder_path: str,
        step: int = 1,
        grid_rows: int = 4,
        grid_cols: int = 4,
        apply_rescaling: bool = True,
        apply_windowing: bool = True
):
    """
    Отображает DICOM-файлы из папки постранично.
    """
    dicom_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    dicom_files.sort()

    if not dicom_files:
        print(f"Не найдено DICOM файлов в папке: {folder_path}")
        return

    selected_files = dicom_files[::step]
    if not selected_files:
        print(f"Нет файлов для отображения (step={step})")
        return

    images_per_page = grid_rows * grid_cols
    total_pages = math.ceil(len(selected_files) / images_per_page)

    for page in range(total_pages):
        plt.figure(figsize=(grid_cols * 3, grid_rows * 3))
        plt.suptitle(
            f'Страница {page + 1}/{total_pages} | Шаг {step} | Rescale: {apply_rescaling}',
            fontsize=16
        )

        for i in range(images_per_page):
            file_index = page * images_per_page + i
            if file_index < len(selected_files):
                ax = plt.subplot(grid_rows, grid_cols, i + 1)
                file_path = os.path.join(folder_path, selected_files[file_index])

                try:
                    ds = dcmread(file_path, force=True)
                    pixel_data, num_frames = get_pixel_data(ds)

                    if apply_rescaling:
                        pixel_data = apply_dicom_rescaling(pixel_data, ds)

                    if num_frames > 1:
                        if len(pixel_data.shape) == 3:
                            if pixel_data.shape[0] == num_frames:
                                display_image = pixel_data[num_frames // 2]
                            elif pixel_data.shape[2] == num_frames:
                                display_image = pixel_data[:, :, num_frames // 2]
                            else:
                                display_image = pixel_data[0]
                        else:
                            display_image = pixel_data[0]
                    else:
                        display_image = pixel_data

                    if apply_windowing:
                        window_center, window_width = get_dicom_window_params(ds)
                        display_image = apply_ct_window(display_image, window_center, window_width)
                        cmap = 'gray'
                    else:
                        cmap = 'viridis'

                    ax.imshow(display_image, cmap=cmap)

                    title = f"{selected_files[file_index]}"
                    if apply_rescaling:
                        if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
                            fg = ds.SharedFunctionalGroupsSequence[0]
                            if hasattr(fg, 'CTPixelValueRescaleSequence') and len(fg.CTPixelValueRescaleSequence) > 0:
                                rs = fg.CTPixelValueRescaleSequence[0]
                                slope = getattr(rs, 'RescaleSlope', 1)
                                intercept = getattr(rs, 'RescaleIntercept', 0)
                                title += f"\n[Enhanced] Slope: {slope}, Intercept: {intercept}"
                        else:
                            slope = getattr(ds, 'RescaleSlope', 1)
                            intercept = getattr(ds, 'RescaleIntercept', 0)
                            title += f"\nSlope: {slope}, Intercept: {intercept}"

                    ax.set_title(title, fontsize=8)
                    ax.axis('off')

                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                    ax.set_title(f"Error: {selected_files[file_index]}", fontsize=8, color='red')
                    ax.axis('off')

        plt.tight_layout()
        plt.show()


def print_dicom_rescale_info(folder_path: str):
    """
    Выводит информацию о rescale параметрах для всех DICOM файлов в папке.
    """
    dicom_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    print("Rescale параметры DICOM файлов:")
    print("-" * 50)

    for file in dicom_files[:10]:
        file_path = os.path.join(folder_path, file)
        try:
            ds = dcmread(file_path, force=True)

            if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
                fg = ds.SharedFunctionalGroupsSequence[0]
                if hasattr(fg, 'CTPixelValueRescaleSequence') and len(fg.CTPixelValueRescaleSequence) > 0:
                    rs = fg.CTPixelValueRescaleSequence[0]
                    slope = getattr(rs, 'RescaleSlope', 1)
                    intercept = getattr(rs, 'RescaleIntercept', 0)
                    print(f"{file}: [Enhanced CT] Slope={slope}, Intercept={intercept}")
                    continue

            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            print(f"{file}: Slope={slope}, Intercept={intercept}")

        except Exception as e:
            print(f"{file}: Ошибка чтения - {e}")


def visualize_dicom(input_path: str, **kwargs):
    """
    Универсальная функция для визуализации DICOM.

    Если input_path — папка → вызывает show_dicom_pages.
    Если input_path — файл → вызывает show_enhanced_ct_frames.

    Поддерживаемые kwargs:
        step, grid_rows, grid_cols, apply_rescaling, apply_windowing,
        start_frame, max_frames (для файлов)
    """
    if os.path.isdir(input_path):
        print(f"📂 Обнаружена папка: {input_path}")
        show_dicom_pages(
            folder_path=input_path,
            step=kwargs.get('step', 1),
            grid_rows=kwargs.get('grid_rows', 4),
            grid_cols=kwargs.get('grid_cols', 4),
            apply_rescaling=kwargs.get('apply_rescaling', True),
            apply_windowing=kwargs.get('apply_windowing', True)
        )
    elif os.path.isfile(input_path):
        print(f"📄 Обнаружен файл: {input_path}")
        show_enhanced_ct_frames(
            file_path=input_path,
            step=kwargs.get('step', 1),
            grid_rows=kwargs.get('grid_rows', 4),
            grid_cols=kwargs.get('grid_cols', 4),
            apply_rescaling=kwargs.get('apply_rescaling', True),
            apply_windowing=kwargs.get('apply_windowing', True),
            start_frame=kwargs.get('start_frame', 0),
            max_frames=kwargs.get('max_frames', None)
        )
    else:
        print(f"❌ Путь не существует: {input_path}")

def get_number_of_frames(dicom_path):
    ds = dcmread(dicom_path, force=True)
    num_frames = getattr(ds, 'NumberOfFrames', 1)  # 1 — значение по умолчанию для single-frame
    return int(num_frames)

def analyze_dicom_volume(dicom_path: str,
                         low_att_threshold: int = -950,
                         visualize: bool = False) -> Union[Tuple[float, float, float, Dict], None]:
    """
    Анализирует DICOM-данные с учетом реальных физических размеров вокселей.
    Работает как с отдельными файлами срезов, так и с многосрезовыми файлами.
    """

    slices = None  # Инициализируем переменную

    # Загрузка и обработка DICOM-данных
    if os.path.isdir(dicom_path):
        # Чтение множества отдельных файлов
        slices = []

        # Получаем все файлы в папке
        dicom_files = [f for f in os.listdir(dicom_path)
                       if os.path.isfile(os.path.join(dicom_path, f))]

        if not dicom_files:
            raise ValueError(f"В указанной папке {dicom_path} не найдены файлы")

        dicom_files.sort()

        for file_name in dicom_files:
            file_path = os.path.join(dicom_path, file_name)
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
                else:
                    warnings.warn(f"Файл {file_name} не содержит pixel_array")
            except Exception as e:
                warnings.warn(f"Не удалось прочитать файл {file_name}: {str(e)}")

        if not slices:
            raise ValueError("Не удалось загрузить ни одного среза с данными")

        print(f"Успешно загружено срезов: {len(slices)}")

        # Сортировка срезов по позиции
        try:
            if hasattr(slices[0], 'ImagePositionPatient'):
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                print("Срезы отсортированы по позиции")
            else:
                print("Информация о позиции недоступна, используется порядок файлов")
        except Exception as e:
            warnings.warn(f"Не удалось отсортировать срезы: {str(e)}")

        # Создание 3D-массива
        pixel_data = np.stack([s.pixel_array for s in slices])
        ds = slices[0]  # используем первую slice как reference

    else:
        # Чтение одного файла с объемом
        try:
            ds = pydicom.dcmread(dicom_path)
            if hasattr(ds, 'pixel_array'):
                pixel_data = ds.pixel_array
                print(f"Успешно загружен многосрезовый файл: {os.path.basename(dicom_path)}")
                print(f"Размер файла: {pixel_data.shape}")
            else:
                raise ValueError("DICOM-файл не содержит pixel data.")
        except Exception as e:
            raise ValueError(f"Не удалось прочитать файл {dicom_path}: {str(e)}")

    # Извлечение метаданных о размерах вокселей
    # Для многосрезовых файлов передаем None вместо slices
    voxel_size_x, voxel_size_y, voxel_size_z = get_voxel_dimensions(ds, slices)
    voxel_volume_mm3 = voxel_size_x * voxel_size_y * voxel_size_z

    print(f"Размер вокселя: {voxel_size_x:.3f} × {voxel_size_y:.3f} × {voxel_size_z:.3f} мм")
    print(f"Объем вокселя: {voxel_volume_mm3:.3f} мм³")

    # Преобразование в единицы Хаунсфилда (HU)
    slope = ds.RescaleSlope if hasattr(ds, 'RescaleSlope') else 1
    intercept = ds.RescaleIntercept if hasattr(ds, 'RescaleIntercept') else 0
    hu_volume = pixel_data * slope + intercept

    print(f"Размер объема: {hu_volume.shape}")
    print(f"Диапазон HU: [{np.min(hu_volume):.1f}, {np.max(hu_volume):.1f}]")

    # Вычисление статистики для всего объема
    mean_hu = np.mean(hu_volume)
    std_hu = np.std(hu_volume)

    # Процент низкоаттенуированной ткани (по количеству вокселей)
    low_att_voxels = np.sum(hu_volume <= low_att_threshold)
    total_voxels = hu_volume.size
    pct_low_attenuation = (low_att_voxels / total_voxels) * 100

    # Дополнительные метрики с учетом объема
    total_volume_cm3 = total_voxels * voxel_volume_mm3 / 1000  # в см³
    low_att_volume_cm3 = low_att_voxels * voxel_volume_mm3 / 1000  # в см³
    pct_low_att_volume = (low_att_volume_cm3 / total_volume_cm3) * 100

    # Сбор метаданных
    metadata = {
        'voxel_size_mm': (voxel_size_x, voxel_size_y, voxel_size_z),
        'voxel_volume_mm3': voxel_volume_mm3,
        'volume_shape': hu_volume.shape,
        'total_voxels': total_voxels,
        'total_volume_cm3': total_volume_cm3,
        'low_att_voxels': low_att_voxels,
        'low_att_volume_cm3': low_att_volume_cm3,
        'pct_low_att_volume': pct_low_att_volume,
        'dicom_metadata': extract_dicom_metadata(ds),
        'is_multi_slice_file': (slices is None)  # True для многосрезовых файлов
    }

    # Визуализация
    if visualize:
        visualize_dicom_analysis(hu_volume, mean_hu, std_hu,
                                 pct_low_attenuation, low_att_threshold,
                                 metadata)

    return mean_hu, std_hu, pct_low_attenuation, metadata


def get_voxel_dimensions(ds, slices=None):
    """
    Вычисляет физические размеры вокселя из метаданных DICOM.
    Работает как для отдельных срезов, так и для многосрезовых файлов.
    """
    # Размер вокселя в плоскости XY (из PixelSpacing)
    if hasattr(ds, 'PixelSpacing'):
        pixel_spacing = ds.PixelSpacing
        voxel_size_x = float(pixel_spacing[0])
        voxel_size_y = float(pixel_spacing[1])
    else:
        voxel_size_x = voxel_size_y = 1.0
        warnings.warn("PixelSpacing не найден, используется значение по умолчанию 1.0 мм")

    # Размер вокселя по оси Z
    if hasattr(ds, 'SliceThickness'):
        voxel_size_z = float(ds.SliceThickness)
    elif hasattr(ds, 'SpacingBetweenSlices'):
        voxel_size_z = float(ds.SpacingBetweenSlices)
    else:
        voxel_size_z = 1.0
        warnings.warn("SliceThickness/SpacingBetweenSlices не найден, используется значение по умолчанию 1.0 мм")

    # Для многосрезовых файлов пытаемся определить расстояние между кадрами
    if slices is None and hasattr(ds, 'pixel_array') and ds.pixel_array.ndim == 3:
        try:
            # Если это многосрезовый файл, проверяем наличие информации о позициях
            if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
                # Для Enhanced DICOM с последовательностью кадров
                num_frames = ds.pixel_array.shape[0]
                if num_frames > 1:
                    # Пытаемся получить информацию о первом и последнем кадре
                    first_frame = ds.PerFrameFunctionalGroupsSequence[0]
                    last_frame = ds.PerFrameFunctionalGroupsSequence[-1]

                    if hasattr(first_frame, 'PlanePositionSequence') and hasattr(last_frame, 'PlanePositionSequence'):
                        first_pos = float(first_frame.PlanePositionSequence[0].ImagePositionPatient[2])
                        last_pos = float(last_frame.PlanePositionSequence[0].ImagePositionPatient[2])
                        total_distance = abs(last_pos - first_pos)
                        voxel_size_z = total_distance / (num_frames - 1)
                        print(f"Расчетное расстояние между кадрами: {voxel_size_z:.3f} мм")
        except:
            pass

    # Для отдельных файлов срезов используем информацию о расстоянии между срезами
    elif slices and len(slices) > 1:
        try:
            # Вычисляем расстояние между срезами из ImagePositionPatient
            z_positions = []
            for slice_ds in slices:
                if hasattr(slice_ds, 'ImagePositionPatient'):
                    z_positions.append(float(slice_ds.ImagePositionPatient[2]))

            if len(z_positions) > 1:
                z_positions.sort()
                slice_gap = np.mean(np.diff(z_positions))
                if abs(slice_gap - voxel_size_z) > 0.1:  # если разница значительная
                    print(f"Обнаружен gap между срезами: {slice_gap:.3f} мм")
                    voxel_size_z = abs(slice_gap)  # используем фактическое расстояние
        except:
            pass

    return voxel_size_x, voxel_size_y, voxel_size_z


def extract_dicom_metadata(ds):
    """
    Извлекает важные технические параметры из метаданных DICOM.
    """
    metadata = {}

    # Основные технические параметры
    technical_attributes = [
        'ReconstructionDiameter', 'DataCollectionDiameter',
        'SpiralPitchFactor', 'DetectorConfiguration', 'ConvolutionKernel',
        'KVP', 'XRayTubeCurrent', 'ExposureTime', 'SliceThickness',
        'PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient',
        'SpacingBetweenSlices', 'NumberOfFrames', 'Rows', 'Columns'
    ]

    for attr in technical_attributes:
        if hasattr(ds, attr):
            try:
                metadata[attr.lower()] = getattr(ds, attr)
            except:
                pass

    # Дополнительная информация для многосрезовых файлов
    if hasattr(ds, 'pixel_array') and ds.pixel_array.ndim == 3:
        metadata['is_multi_frame'] = True
        metadata['num_frames'] = ds.pixel_array.shape[0]
    else:
        metadata['is_multi_frame'] = False

    return metadata


def visualize_dicom_analysis(hu_volume, mean_hu, std_hu,
                             pct_low_attenuation, low_att_threshold,
                             metadata):
    """Визуализирует результаты анализа DICOM-данных с метаданными."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Срез в середине объема
    mid_slice_idx = hu_volume.shape[0] // 2
    im = axes[0, 0].imshow(hu_volume[mid_slice_idx], cmap='gray',
                           vmin=-1000, vmax=400)
    axes[0, 0].set_title(f'Срез #{mid_slice_idx}')
    plt.colorbar(im, ax=axes[0, 0], label='HU')

    # 2. Гистограмма распределения HU
    axes[0, 1].hist(hu_volume.flatten(), bins=200, range=(-1200, 600),
                    alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(low_att_threshold, color='red', linestyle='--',
                       label=f'Порог низкой аттенуации ({low_att_threshold} HU)')
    axes[0, 1].set_xlabel('Значение HU')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение значений HU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Визуализация низкоаттенуированных областей
    low_att_slice = hu_volume[mid_slice_idx] <= low_att_threshold
    axes[1, 0].imshow(hu_volume[mid_slice_idx], cmap='gray',
                      vmin=-1000, vmax=400)
    axes[1, 0].imshow(np.ma.masked_where(~low_att_slice, low_att_slice),
                      cmap='jet', alpha=0.5)
    axes[1, 0].set_title(f'Низкоаттенуированные области (<{low_att_threshold} HU)')

    # 4. Текстовое отображение статистики и метаданных
    axes[1, 1].axis('off')

    voxel_x, voxel_y, voxel_z = metadata['voxel_size_mm']
    file_type = "Многосрезовый файл" if metadata['is_multi_slice_file'] else "Отдельные срезы"

    stats_text = f"""
    СТАТИСТИКА ОБЪЕМА ({file_type}):
    Среднее HU: {mean_hu:.2f}
    Стандартное отклонение HU: {std_hu:.2f}
    Низкоаттенуированная ткань: {pct_low_attenuation:.2f}%
    Объем низкоаттенуированной ткани: {metadata['low_att_volume_cm3']:.1f} см³

    ФИЗИЧЕСКИЕ ПАРАМЕТРЫ:
    Размер вокселя: {voxel_x:.3f} × {voxel_y:.3f} × {voxel_z:.3f} мм
    Объем вокселя: {metadata['voxel_volume_mm3']:.3f} мм³
    Общий объем: {metadata['total_volume_cm3']:.1f} см³
    Размер объема: {metadata['volume_shape']}
    """

    axes[1, 1].text(0.1, 0.9, stats_text, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ==============================
    # 🔹 Пример 1: Визуализация папки с DICOM-файлами
    # ==============================
    folder_path = "data/dicom"  # замени на свой путь
    visualize_dicom(
        folder_path,
        step=29,
        grid_rows=4,
        grid_cols=4,
        apply_rescaling=True,
        apply_windowing=True
    )

    # ==============================
    # 🔹 Пример 2: Визуализация одного Enhanced CT файла
    # ==============================
    file_path = "data/dicom/pneumothorax.dcm"  # замени на свой файл
    visualize_dicom(
        file_path,
        step=5,
        grid_rows=4,
        grid_cols=4,
        apply_rescaling=True,
        apply_windowing=True,
        start_frame=10,
        max_frames=50
    )

    # ==============================
    # 🔹 Пример 3: Вывод информации о rescale
    # ==============================
    print("\n" + "="*60)
    print_dicom_rescale_info(folder_path)