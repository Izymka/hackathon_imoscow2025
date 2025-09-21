import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread


def get_nested_tag(node, nested, tag_name):
    if hasattr(node, nested) :
        node_seq = node[nested][0]
        if hasattr(node_seq, tag_name):
            return getattr(node_seq, tag_name)
    return None

def get_dicom_tag(ds, tag_name, default='N/A'):
    """
    Универсальная функция для извлечения тега из DICOM.
    Поддерживает Enhanced CT через SharedFunctionalGroupsSequence.
    """
    # Сначала пробуем из корня
    if hasattr(ds, tag_name):
        val = getattr(ds, tag_name)
        if isinstance(val, (list, tuple)) and len(val) == 1:
            return val[0]
        return val

    if tag_name == 'CTDIvol':
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and len(ds.PerFrameFunctionalGroupsSequence) > 0:
            fg = ds.PerFrameFunctionalGroupsSequence[0]
            nested = get_nested_tag(fg, 'CTExposureSequence', tag_name)
            if nested is not None:
                return nested

    # Потом из SharedFunctionalGroupsSequence
    if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
        fg = ds.SharedFunctionalGroupsSequence[0]

        if tag_name == 'SpiralPitchFactor':
            nested = get_nested_tag(fg, 'CTTableDynamicsSequence', tag_name)
            if nested is not None:
                return nested

        # Обработка ReconstructionDiameter
        if tag_name == 'ReconstructionDiameter':
            nested = get_nested_tag(fg, 'CTReconstructionSequence', tag_name)
            if nested is not None:
                return nested

        # Для специальных случаев - PixelSpacing
        if tag_name == 'PixelSpacing':
            nested = get_nested_tag(fg, 'PixelMeasuresSequence', tag_name)
            if nested is not None:
                return nested

        # Для специальных случаев — Rescale
        if tag_name == 'RescaleSlope' or tag_name == 'RescaleIntercept':
            nested = get_nested_tag(fg, 'CTPixelValueRescaleSequence', tag_name)
            if nested is not None:
                return nested
            nested = get_nested_tag(fg, 'PixelValueTransformationSequence', tag_name)
            if nested is not None:
                return nested

        # Для KVP, Exposure и т.д.
        if hasattr(fg, 'CTExposureSequence') and len(fg.CTExposureSequence) > 0:
            nested = get_nested_tag(fg, 'CTExposureSequence', tag_name)
            if nested is not None:
                return nested

        if hasattr(fg, 'CTGeometrySequence') and len(fg.CTGeometrySequence) > 0:
            geo_seq = fg.CTGeometrySequence[0]
            if hasattr(geo_seq, tag_name):
                return getattr(geo_seq, tag_name)

        # Общий случай — если тег есть прямо в функциональной группе
        if hasattr(fg, tag_name):
            return getattr(fg, tag_name)

    return default


def apply_dicom_rescaling(pixel_array, ds):
    """
    Применяет rescale intercept и slope к DICOM pixel array.
    Поддерживает Enhanced CT.
    """
    # Enhanced CT
    if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
        fg = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(fg, 'CTPixelValueRescaleSequence') and len(fg.CTPixelValueRescaleSequence) > 0:
            rs = fg.CTPixelValueRescaleSequence[0]
            rescale_intercept = float(getattr(rs, 'RescaleIntercept', 0))
            rescale_slope = float(getattr(rs, 'RescaleSlope', 1))
            return pixel_array * rescale_slope + rescale_intercept

    # Стандартный CT
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0))
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1))

    return pixel_array * rescale_slope + rescale_intercept


def get_image_position_patient(ds, frame_index=0):
    """
    Извлекает ImagePositionPatient для указанного фрейма.
    Для Enhanced CT — из PerFrameFunctionalGroupsSequence.
    Для обычного DICOM — из корня.
    """
    if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and len(ds.PerFrameFunctionalGroupsSequence) > frame_index:
        frame_fg = ds.PerFrameFunctionalGroupsSequence[frame_index]
        if hasattr(frame_fg, 'PlanePositionSequence') and len(frame_fg.PlanePositionSequence) > 0:
            plane_pos = frame_fg.PlanePositionSequence[0]
            if hasattr(plane_pos, 'ImagePositionPatient'):
                return getattr(plane_pos, 'ImagePositionPatient')

    # fallback: из корня
    if hasattr(ds, 'ImagePositionPatient'):
        return getattr(ds, 'ImagePositionPatient')

    return [0, 0, 0]


def analyze_dicom_series(folder_path):
    """
    Анализирует серию DICOM-файлов и выводит подробную информацию о параметрах исследования.
    Теперь использует утилиту parse_dicom для извлечения метаданных и сведений по всей серии.
    """
    from .dicom_parser import parse_dicom

    # Получаем сводную информацию
    summary = parse_dicom(folder_path)
    if summary is None:
        print("Не удалось найти корректные DICOM файлы в папке")
        return pd.DataFrame()

    # Подсчет/тип
    print(f"Всего {summary['source_files_total']} DICOM файлов")
    print(f"\nТип файла: {'Enhanced CT' if summary['is_enhanced_ct'] else 'Стандартный CT'}")
    if summary.n_frames > 0:
        print(f"Количество фреймов: {summary.n_frames}")

    # === ПАРАМЕТРЫ ИССЛЕДОВАНИЯ ===
    print("\n=== ПАРАМЕТРЫ ИССЛЕДОВАНИЯ ===")
    manuf = summary.get('manufacturer') or 'N/A'
    model = summary.get('manufacturer_model_name') or 'N/A'
    print(f"Тип сканера: {manuf} {model}")
    print(f"Модальность: {summary.get('modality', 'N/A')}")
    print(f"Область исследования: {summary.get('body_part_examined', 'N/A')}")
    print(f"Тип изображения: {summary.get('image_type', 'N/A')}")

    # === ПАРАМЕТРЫ СКАНИРОВАНИЯ ===
    print("\n=== ПАРАМЕТРЫ СКАНИРОВАНИЯ ===")
    print(f"Напряжение на трубке (кВ): {summary.get('kvp', 'N/A')}")
    print(f"Ток на трубке (мА): {summary.get('xray_tube_current', 'N/A')}")
    print(f"Экспозиция (мАс): {summary.get('exposure', 'N/A')}")
    print(f"Толщина среза (мм): {summary.get('slice_thickness', 'N/A')}")
    print(f"Алгоритм реконструкции: {summary.get('convolution_kernel', 'N/A')}")
    print(f"Шаг спирали (Pitch): {summary.get('spiral_pitch_factor', 'N/A')}")
    print(f"Объемная доза КТ (мГр): {summary.get('ctdi_vol', 'N/A')}")

    # === ГЕОМЕТРИЯ ===
    print("\n=== ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ ===")
    rows = summary.get('rows', 'N/A')
    cols = summary.get('cols', 'N/A')
    print(f"Размер изображения: {rows}×{cols} пикселей")

    pixel_spacing = summary.get('pixel_spacing') or ['N/A', 'N/A']
    slice_thickness = summary.get('slice_thickness', 'N/A')
    if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2 and all(isinstance(v, (int, float)) for v in pixel_spacing[:2]):
        print(f"Размер пикселя (X,Y): {pixel_spacing[0]}×{pixel_spacing[1]} мм")
        print(f"Размер вокселя (X,Y,Z): {pixel_spacing[0]}×{pixel_spacing[1]}×{slice_thickness} мм")
    else:
        print(f"Размер пикселя: {pixel_spacing} мм")

    print(f"Диаметр реконструкции: {summary.get('reconstruction_diameter', 'N/A')} мм")

    # === КООРДИНАТНАЯ СИСТЕМА ===
    print("\n=== КООРДИНАТНАЯ СИСТЕМА ===")
    print(f"Положение пациента: {summary.get('patient_position', 'N/A')}")
    print(f"Ориентация пациента: {summary.get('patient_orientation', 'N/A')}")
    print(f"Ориентация изображения: {summary.get('image_orientation_patient', 'N/A')}")
    print(f"Позиция среза (первого): {summary.get('first_image_position_patient', 'N/A')}")
    print(f"Локация среза: {summary.get('slice_location', 'N/A')} мм")

    # === ВИЗУАЛИЗАЦИЯ ===
    print("\n=== ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ ===")
    print(f"Фотометрическая интерпретация: {summary.get('photometric_interpretation', 'N/A')}")
    print(f"Битовая глубина: {summary.get('bits_stored', 'N/A')} бит")
    pixel_repr = summary.get('pixel_representation', 0) or 0
    print(f"Представление пикселей: {'со знаком' if pixel_repr == 1 else 'без знака'}")

    # === WINDOW & RESCALE ===
    window_center = summary.get('window_center', 'N/A')
    window_width = summary.get('window_width', 'N/A')
    rescale_slope = summary.get('rescale_slope', 'N/A')
    rescale_intercept = summary.get('rescale_intercept', 'N/A')

    print(f"\n=== ПАРАМЕТРЫ ОКНА И RESCALE ===")
    print(f"Уровень окна: {window_center}")
    print(f"Ширина окна: {window_width}")
    print(f"Наклон пересчета: {rescale_slope}")
    print(f"Интерцепт пересчета: {rescale_intercept}")

    # Пример преобразования и диапазон HU (опционально, читаем пиксели только при необходимости)
    try:
        # Попробуем открыть один представительный файл полностью, чтобы получить пиксельные данные
        rep_path = summary.get('representative_path')
        if rep_path:
            ds = dcmread(rep_path, force=True)
            if hasattr(ds, 'pixel_array') and rescale_slope is not None and rescale_intercept is not None:
                hu_data = apply_dicom_rescaling(ds.pixel_array, ds)
                print(f"Диапазон HU значений: [{hu_data.min():.1f}, {hu_data.max():.1f}]")
    except Exception:
        pass

    # === ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ ===
    print("\n=== ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ ===")
    print(f"Наклон гентри: {summary.get('gantry_detector_tilt', 'N/A')} градусов")
    print(f"Высота стола: {summary.get('table_height', 'N/A')} мм")
    print(f"Направление вращения: {summary.get('rotation_direction', 'N/A')}")
    print(f"Фокусные пятна: {summary.get('focal_spots', 'N/A')} мм")
    print(f"Тип фильтра: {summary.get('filter_type', 'N/A')}")
    print(f"Мощность генератора: {summary.get('generator_power', 'N/A')} кВт")
    print(f"Тип модуляции экспозиции: {summary.get('exposure_modulation_type', 'N/A')}")

    # === СБОР ДАННЫХ ПО ВСЕЙ СЕРИИ ===
    names = summary.get('series_file_names') or []
    insts = summary.get('instance_numbers') or []
    z_positions = summary.get('z_positions') or []
    slopes = summary.get('rescale_slopes_series') or []
    intercepts = summary.get('rescale_intercepts_series') or []

    df_z = pd.DataFrame({
        'filename': names if names else [f'frame_{i:04d}' for i in range(len(z_positions))],
        'instance_number': insts if insts else list(range(1, len(z_positions)+1)),
        'z_position': z_positions,
        'rescale_slope': slopes if slopes else [rescale_slope]*len(z_positions),
        'rescale_intercept': intercepts if intercepts else [rescale_intercept]*len(z_positions),
    })
    df_z = df_z.sort_values('z_position').reset_index(drop=True)

    # Анализ rescale
    unique_slopes = df_z['rescale_slope'].dropna().unique()
    unique_intercepts = df_z['rescale_intercept'].dropna().unique()

    print(f"\n=== АНАЛИЗ ПАРАМЕТРОВ RESCALE ПО СЕРИИ ===")
    print(f"Уникальные значения rescale_slope: {unique_slopes}")
    print(f"Уникальные значения rescale_intercept: {unique_intercepts}")

    if len(unique_slopes) == 1 and len(unique_intercepts) == 1:
        print("✓ Параметры rescale консистентны по всей серии")
    else:
        print("⚠ Внимание: параметры rescale различаются между файлами/фреймами!")

    print(f"\n=== СТАТИСТИКА Z-КООРДИНАТ ===")
    if len(df_z) > 0 and df_z['z_position'].notna().any():
        z_clean = df_z['z_position'].dropna()
        print(f"Минимальная Z-координата: {z_clean.min():.2f} мм")
        print(f"Максимальная Z-координата: {z_clean.max():.2f} мм")
        print(f"Общий диапазон: {z_clean.max() - z_clean.min():.2f} мм")
        print(f"Количество уникальных Z-positions: {z_clean.nunique()}")

        if len(z_clean) > 1:
            z_differences = np.diff(z_clean.values)
            print(f"\n=== АНАЛИЗ ИНТЕРВАЛОВ ===")
            print(f"Средний интервал между срезами: {np.mean(z_differences):.4f} мм")
            print(f"Минимальный интервал: {np.min(z_differences):.4f} мм")
            print(f"Максимальный интервал: {np.max(z_differences):.4f} мм")
            print(f"Стандартное отклонение интервалов: {np.std(z_differences):.4f} мм")

            expected_interval = float(slice_thickness) if isinstance(slice_thickness, (int, float, np.floating)) else 1.0
            deviation = np.abs(z_differences - expected_interval)
            print(f"\nОтклонение от ожидаемой толщины {expected_interval} мм:")
            print(f"Максимальное отклонение: {np.max(deviation):.4f} мм")
            print(f"Среднее отклонение: {np.mean(deviation):.4f} мм")
    else:
        print("Недостаточно данных по Z для анализа")

    return df_z


def visualize_dicom_geometry(df_z, expected_interval=1.0):
    if len(df_z) <= 1:
        print("Недостаточно данных для визуализации")
        return

    z_differences = np.diff(df_z['z_position'].values)

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(df_z['z_position'], 'o-', markersize=2)
    plt.xlabel('Номер среза/фрейма')
    plt.ylabel('Z-координата (мм)')
    plt.title('Распределение Z-координат')
    plt.grid(True)

    plt.subplot(132)
    plt.plot(z_differences, 'o-', markersize=2)
    #plt.axhline(y=expected_interval, color='r', linestyle='--',
      #          label=f'Ожидаемый интервал ({expected_interval} мм)')
    plt.xlabel('Интервал между срезами')
    plt.ylabel('Расстояние (мм)')
    plt.title('Интервалы между Z-координатами')
    plt.legend()
    plt.grid(True)

    plt.subplot(133)
    plt.hist(z_differences, bins=30, alpha=0.7)
    #plt.axvline(x=expected_interval, color='r', linestyle='--', label='Ожидаемый интервал')
    plt.xlabel('Интервал (мм)')
    plt.ylabel('Частота')
    plt.title('Гистограмма интервалов')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def show_hu_distribution(folder_path, sample_size=5):
    dicom_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not dicom_files:
        print("Не найдено DICOM файлов")
        return

    sample_files = np.random.choice(dicom_files, min(sample_size, len(dicom_files)), replace=False)
    plt.figure(figsize=(15, 3 * sample_size))

    for i, filename in enumerate(sample_files):
        file_path = os.path.join(folder_path, filename)
        try:
            ds = dcmread(file_path, force=True)
            hu_data = apply_dicom_rescaling(ds.pixel_array, ds)

            plt.subplot(sample_size, 2, i * 2 + 1)
            plt.imshow(hu_data, cmap='gray')
            plt.title(f'{filename}\nHU range: [{hu_data.min():.1f}, {hu_data.max():.1f}]')
            plt.axis('off')

            plt.subplot(sample_size, 2, i * 2 + 2)
            plt.hist(hu_data.flatten(), bins=100, alpha=0.7)
            plt.xlabel('HU Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of HU Values')
            plt.grid(True)

        except Exception as e:
            print(f"Ошибка при анализе {filename}: {e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = "data/dicom"

    df_metadata = analyze_dicom_series(folder_path)

    if not df_metadata.empty:
        # Берем толщину среза из первого файла
        first_file = os.path.join(folder_path, os.listdir(folder_path)[0])
        ds = dcmread(first_file, force=True)
        expected_thickness = float(get_dicom_tag(ds, 'SliceThickness', 1.0))
        visualize_dicom_geometry(df_metadata, expected_thickness)
        show_hu_distribution(folder_path, sample_size=3)

    print("\nПервые 5 записей метаданных:")
    print(df_metadata.head())