import pandas as pd
import os
import pydicom
import time
import json
from json import JSONEncoder

DCOM_FILES_TO_ANALYSYS = 1

# Базовый путь к сетевому хранилищу
base_path = "/Volumes/DS/DataSets/MosMed/ds_cancer_1/MosMedData-LDCT-LUNGCR-type I-v 1/"

# Полные пути к файлам
excel_path = os.path.join(base_path, 'dataset_registry.xlsx')
studies_dir = os.path.join(base_path, 'studies')

print("=" * 50)
print("НАЧАЛО ЗАГРУЗКИ ДАТАСЕТА")
print("=" * 50)

# Проверяем доступность путей
print(f"Проверяем базовый путь: {base_path}")
print(f"Существует: {os.path.exists(base_path)}")

print(f"Проверяем Excel файл: {excel_path}")
print(f"Существует: {os.path.exists(excel_path)}")

print(f"Проверяем папку studies: {studies_dir}")
print(f"Существует: {os.path.exists(studies_dir)}")

# Загрузка меток из Excel
print("\n1. ЗАГРУЗКА METOK ИЗ EXCEL...")
start_time = time.time()

try:
    df = pd.read_excel(excel_path)
    excel_load_time = time.time() - start_time
    print(f"✓ Excel загружен за {excel_load_time:.2f} секунд")
    print("Загружено меток:", len(df))
    print("Первые 5 строк:")
    print(df.head())

except Exception as e:
    print(f"✗ Ошибка загрузки Excel: {e}")
    exit()

# Создаем словарь для быстрого поиска меток по StudyInstanceUID
print("\n2. СОЗДАНИЕ СЛОВАРЯ МЕТОК...")
start_time = time.time()

labels_dict = dict(zip(df['study_instance_anon'], df['pathology']))
dict_time = time.time() - start_time
print(f"✓ Словарь создан за {dict_time:.2f} секунд")
print("Пример словаря меток (первые 5):")
for i, (key, value) in enumerate(list(labels_dict.items())[:5]):
    print(f"  {i + 1}. {key} -> {value}")

# Проверяем доступные исследования
print("\n3. ПРОВЕРКА ДОСТУПНЫХ ИССЛЕДОВАНИЙ...")
start_time = time.time()

try:
    available_studies = os.listdir(studies_dir)
    list_studies_time = time.time() - start_time
    print(f"✓ Список исследований получен за {list_studies_time:.2f} секунд")
    print(f"Найдено папок исследований: {len(available_studies)}")
    print("Первые 10 studyUID:")
    for i, study in enumerate(available_studies[:10]):
        print(f"  {i + 1}. {study}")

    # Проверяем соответствие меток и исследований
    studies_with_labels = set(labels_dict.keys())
    studies_in_folder = set(available_studies)
    matched_studies = studies_in_folder.intersection(studies_with_labels)

    print(f"\nСоответствие меток и исследований:")
    print(f"  Исследований с метками: {len(matched_studies)}")
    print(f"  Исследований без меток: {len(studies_in_folder - studies_with_labels)}")
    print(f"  Меток без исследований: {len(studies_with_labels - studies_in_folder)}")

except Exception as e:
    print(f"✗ Ошибка при получении списка исследований: {e}")
    exit()


# Функция для обработки одного исследования
def process_study(study_folder_path, study_uid):
    """
    Обрабатывает одно исследование и возвращает данные с меткой
    """
    print(f"  Обработка исследования: {study_uid}")
    study_start_time = time.time()

    study_data = {
        'study_uid': study_uid,
        'pathology': labels_dict.get(study_uid, None),
        'series': [],
        'dicom_files': []
    }

    if study_data['pathology'] is None:
        print(f"  ✗ Для исследования {study_uid} не найдена метка!")
        return None

    try:
        # Проходим по всем сериям в исследовании
        series_list = os.listdir(study_folder_path)
        print(f"    Найдено серий: {len(series_list)}")

        for series_uid in series_list:
            series_path = os.path.join(study_folder_path, series_uid)

            if os.path.isdir(series_path):
                series_data = {
                    'series_uid': series_uid,
                    'dicom_files': []
                }

                # Читаем все DICOM файлы в серии
                try:
                    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
                    print(f"    Серия {series_uid}: {len(dicom_files)} DICOM файлов")

                    for filename in dicom_files[:DCOM_FILES_TO_ANALYSYS]:  # Обрабатываем только первые 3 файла для отладки
                        dicom_path = os.path.join(series_path, filename)
                        try:
                            # Читаем только заголовок для экономии памяти
                            dicom_data = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                            series_data['dicom_files'].append({
                                'filename': filename,
                                'path': dicom_path,
                                'dicom_data': dicom_data
                            })
                        except Exception as e:
                            print(f"      ✗ Ошибка чтения {filename}: {e}")

                except Exception as e:
                    print(f"    ✗ Ошибка при обработке серии {series_uid}: {e}")

                study_data['series'].append(series_data)
                study_data['dicom_files'].extend(series_data['dicom_files'])

        study_time = time.time() - study_start_time
        print(f"  ✓ Исследование {study_uid} обработано за {study_time:.2f} секунд")
        return study_data

    except Exception as e:
        print(f"  ✗ Критическая ошибка при обработке исследования {study_uid}: {e}")
        return None


# Основной цикл обработки всех исследований
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)


def save_dataset_to_json(dataset, output_path):
    print(f"\nСохранение датасета в JSON файл: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
        print(f"✓ Датасет успешно сохранен")
    except Exception as e:
        print(f"✗ Ошибка при сохранении JSON: {e}")


def load_json_dataset(json_path):
    """Загрузка датасета из JSON файла"""
    print(f"\nЗагрузка датасета из JSON: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✓ Датасет успешно загружен из JSON")
        print(f"Количество исследований: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"✗ Ошибка при загрузке JSON: {e}")
        return None


def load_dataset():
    print("\n4. ОБРАБОТКА ИССЛЕДОВАНИЙ...")
    dataset = []
    processed_count = 0
    total_start_time = time.time()

    try:
        study_list = os.listdir(studies_dir)
        print(f"Всего исследований для обработки: {len(study_list)}")

        for i, study_uid in enumerate(study_list):
            study_path = os.path.join(studies_dir, study_uid)

            if os.path.isdir(study_path):
                print(f"\n[{i + 1}/{len(study_list)}] ", end="")
                study_data = process_study(study_path, study_uid)

                if study_data:
                    dataset.append(study_data)
                    processed_count += 1

                    # Выводим прогресс каждые 5 исследований
                    if processed_count % 5 == 0:
                        elapsed_time = time.time() - total_start_time
                        print(f"  Прогресс: {processed_count} исследований обработано за {elapsed_time:.2f} секунд")

                # Добавляем небольшую задержку для отладки
                time.sleep(0.1)

    except Exception as e:
        print(f"✗ Ошибка в основном цикле: {e}")

    total_time = time.time() - total_start_time
    print(f"\n✓ Обработка завершена за {total_time:.2f} секунд")
    print(f"Успешно обработано исследований: {len(dataset)}")

    return dataset


# Загрузка датасета
print("\n" + "=" * 50)
print("ЗАПУСК ОСНОВНОЙ ОБРАБОТКИ")
print("=" * 50)

json_path = os.path.join(base_path, 'dataset.json')

if os.path.exists(json_path):
    dataset = load_json_dataset(json_path)
else:
    dataset = load_dataset()
    save_dataset_to_json(dataset, json_path)

print("\n" + "=" * 50)
print("ЗАВЕРШЕНИЕ РАБОТЫ")
print("=" * 50)
print(f"Итоговый размер датасета: {len(dataset)} исследований")