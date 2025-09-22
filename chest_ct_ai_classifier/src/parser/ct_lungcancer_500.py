import argparse
import csv
import json
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path
from dotenv import load_dotenv

from chest_ct_ai_classifier.src.utils.dicom_parser import parse_dicom

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Глобальная переменная для отслеживания первой записи
csv_initialized = False

base_path = os.getenv("DS_CT_LUNGCANCER_500_FILE")
if not base_path:
    logging.error("DS_CT_LUNGCANCER_500_FILE environment variable is not set")
    exit(1)
base_path = Path(base_path)
if not base_path.exists():
    logging.error("Directory does not exist: %s", base_path)
    exit(1)

dicom_path = base_path / "dicom"
if not dicom_path.exists():
    logging.error("Dicom directory does not exist: %s", dicom_path)
    exit(1)

dataset_target_path = os.getenv("DS_TARGET_PATH")
if not dataset_target_path:
    logging.error("DS_TARGET_PATH environment variable is not set")
    exit(1)
dataset_target_path = Path(dataset_target_path)
if not dataset_target_path.exists():
    logging.error("Target DS Path does not exist: %s", base_path)
    exit(1)


result_csv_output_file = base_path / "pathology_results.csv"

def extract_archive(file):
    # If this is a file and looks like an archive, check for extracted directory
    try:
        if file.is_file():
            name_lower = file.name.lower()
            # handle tar.gz and tgz
            if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
                # determine base name without the compression extensions
                if name_lower.endswith(".tar.gz"):
                    base_name = file.name[:-7]  # strip ".tar.gz"
                else:
                    base_name = file.name[:-4]  # strip ".tgz"
                target_dir = dicom_path / base_name

                if not target_dir.exists() or not target_dir.is_dir():
                    logging.info("Archive found: %s. Extracting to: %s", file.name, target_dir)
                    # ensure target directory exists
                    target_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        # open with auto-detection for compression
                        with tarfile.open(file, "r:*") as tf:
                            tf.extractall(path=target_dir)
                        logging.info("Extraction completed: %s", target_dir)
                    except Exception:
                        logging.exception("Failed to extract tar archive: %s", file)
                else:
                    logging.info("Already extracted: %s", target_dir)

            # optionally handle zip archives as well
            elif name_lower.endswith(".zip"):
                base_name = file.name[:-4]  # strip ".zip"
                target_dir = dicom_path / base_name

                if not target_dir.exists() or not target_dir.is_dir():
                    logging.info("ZIP archive found: %s. Extracting to: %s", file.name, target_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.unpack_archive(str(file), str(target_dir))
                        logging.info("ZIP extraction completed: %s", target_dir)
                    except Exception:
                        logging.exception("Failed to extract zip archive: %s", file)
                else:
                    logging.info("Already extracted: %s", target_dir)
    except PermissionError:
        logging.error("Permission denied when processing file: %s", file)
    except Exception:
        logging.exception("Error processing file: %s", file)


def extract_all():
    try:
        files = list(dicom_path.iterdir())
        logging.info("Dicom directory exist. Found %d files:", len(files))
        for file in files:
            logging.info(file.name)
            extract_archive(file)
    except PermissionError:
        logging.error("Permission denied accessing directory: %s", base_path)
    except Exception:
        logging.exception("Error accessing directory: %s", base_path)

def write_to_csv(result, output_file, clear_file=False):
    """
    Записывает результат в CSV файл.

    Args:
        result (dict): Словарь с данными для записи
        output_file (Path): Путь к выходному CSV файлу
        clear_file (bool): Очистить ли файл перед записью
    """
    global csv_initialized

    fieldnames = ['id', 'patology', 'doctor_1_comment', 'doctor_2_comment',
                  'doctor_3_comment', 'doctor_4_comment', 'doctor_5_comment', 'doctor_6_comment',
                  'window_center', 'window_width', 'x', 'y', 'z']

    # Если это первая запись или требуется очистка файла
    if clear_file or not csv_initialized:
        # Создаем/перезаписываем файл с заголовком
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(result)
        csv_initialized = True
    else:
        # Добавляем строку к существующему файлу
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(result)

def read_existing_results(output_file):
    """
    Читает существующий CSV файл с результатами и возвращает множество уже обработанных study_id.

    Args:
        output_file (Path): Путь к CSV файлу с результатами

    Returns:
        set: Множество уже обработанных study_id
    """
    existing_ids = set()

    if not output_file.exists():
        logging.info("CSV файл с результатами не найден, будет создан новый: %s", output_file)
        return existing_ids

    try:
        with open(output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                study_id = row.get('id', '').strip()
                if study_id:
                    existing_ids.add(study_id)

        logging.info("Найдено %d уже обработанных записей в файле: %s", len(existing_ids), output_file)
    except Exception as e:
        logging.error("Ошибка чтения существующего CSV файла %s: %s", output_file, e)

    return existing_ids


def collect_meta(collect_dicom_data=True, recollect=False):
    registry_file = base_path / "dataset_registry.csv"
    protocols_dir = base_path / "protocols"

    if not registry_file.exists():
        logging.error("Файл реестра не найден: %s", registry_file)
        return

    if not protocols_dir.exists():
        logging.error("Папка protocols не найдена: %s", protocols_dir)
        return

    def has_pathology(json_data, check_comments=False):
        """
        Анализирует JSON данные и определяет наличие патологий.
        Возвращает True если есть патологии, False если нет.
        """
        # Проверяем комментарии врачей
        if check_comments:
            doctors = json_data.get("doctors", [])
            for doctor in doctors:
                comment = doctor.get("comment", "").strip().lower()
                if comment and comment not in ["нет", "нет очагов", "достоверно очагов нет", ""]:
                    return True

        # Проверяем nodules
        nodules = json_data.get("nodules", [])
        if not nodules:
            return False

        for nodule_group in nodules:
            for nodule_dict in nodule_group:
                for doctor_id, nodule_data in nodule_dict.items():
                    if nodule_data is not None:
                        return True

        return False

    def get_doctor_comments(json_data):
        doctor_comments = []

        # Собираем комментарии врачей
        doctors = json_data.get("doctors", [])
        for doctor in doctors:
            comment = doctor.get("comment", "").strip()
            doctor_comments.append(comment)

        return doctor_comments

    global csv_initialized
    csv_initialized = False  # Сброс флага для очистки файла
    records_processed = 0
    skipped_count = 0

    # Читаем уже существующие записи из CSV файла
    existing_study_ids = []
    if not recollect:
        existing_study_ids = read_existing_results(result_csv_output_file)
        if existing_study_ids:
            csv_initialized = True

    try:
        # Читаем файл реестра
        with open(registry_file, 'r', encoding='utf-8') as csvfile:
            # Пропускаем заголовок если есть
            first_line = csvfile.readline().strip()
            if first_line.lower() == 'study':
                pass  # это заголовок, пропускаем
            else:
                csvfile.seek(0)  # возвращаемся к началу файла

            for line in csvfile:
                study_id = line.strip()
                if not study_id or study_id.lower() == 'study':
                    continue

                # Проверяем, не был ли этот study_id уже обработан
                if study_id in existing_study_ids:
                    logging.info("Пропускаем уже обработанный файл: %s", study_id)
                    skipped_count += 1
                    continue

                # Ищем соответствующий JSON файл
                json_file = protocols_dir / f"{study_id}.json"

                logging.info("Обработка файла: %s", study_id)

                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8-sig') as f:
                            json_data = json.load(f)

                        # Анализируем наличие патологий и собираем комментарии
                        pathology = has_pathology(json_data)

                        doctor_comments = get_doctor_comments(json_data)

                        result = {
                            'id': study_id,
                            'patology': 1 if pathology else 0
                        }

                        # Добавляем комментарии врачей (до 6)
                        for i in range(6):
                            doctor_key = f"doctor_{i+1}_comment"
                            result[doctor_key] = doctor_comments[i] if i < len(doctor_comments) else "-"

                        if collect_dicom_data:
                            dcom_dir = dicom_path / (study_id)
                            if not dcom_dir.exists():
                                dcom_tar = dicom_path / (study_id + '.tar.gz')
                                if not dcom_tar.exists():
                                    raise FileNotFoundError(f"DICOM directory and archive not found for study_id: {study_id}")
                                extract_archive(dcom_tar)

                            summary = parse_dicom(dcom_dir)
                            if summary:
                                result['window_center'] = str(summary.window_center)
                                result['window_width'] = str(summary.window_width)
                                result['x'] = str(summary.pixel_spacing[0] or 'N/A')
                                result['y'] = str(summary.pixel_spacing[1] or 'N/A')
                                result['z'] = str(summary.pixel_representation or 'N/A')
                                result['hu_volume'] = ''

                        # Записываем результат сразу в CSV файл
                        write_to_csv(result, result_csv_output_file)
                        records_processed += 1
                        logging.info("Обработан файл: %s, патология: %s", study_id, pathology)
                    except Exception as e:
                        logging.error("Ошибка обработки файла %s: %s", json_file, e)
                        result = {'id': study_id, 'patology': -1}
                        # Записываем результат сразу в CSV файл
                        write_to_csv(result, result_csv_output_file)
                        records_processed += 1
                else:
                    logging.warning("JSON файл не найден: %s", json_file)
                    result = {'id': study_id, 'patology': -2}
                    # Записываем результат сразу в CSV файл
                    write_to_csv(result, result_csv_output_file)
                    records_processed += 1

        logging.info("Результаты сохранены в файл: %s", result_csv_output_file)
        logging.info("Обработано новых записей: %d", records_processed)
        logging.info("Пропущено уже обработанных записей: %d", skipped_count)
        logging.info("Всего записей в существующем файле: %d", len(existing_study_ids) + records_processed)

    except Exception as e:
        logging.error("Ошибка при обработке файлов: %s", e)


def move_dataset():
    logging.info("Moving dataset to target path: %s", dataset_target_path)
    # read results from result_csv_output_file, for each study_id, move dicom files to target path
    with open(result_csv_output_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            study_id = row.get('id', '').strip()
            if study_id:
                if row.get('pathology') != "0":
                    logging.info("Skipping study ID: %s, pathology is not present", study_id)
                    continue
                logging.info("Processing study ID: %s", study_id)
                dcom_dir = dicom_path / study_id / study_id
                if dcom_dir.exists():
                    logging.info("Found DICOM directory: %s", dcom_dir)
                    target_dir = dataset_target_path / study_id
                    if not target_dir.exists():
                        logging.info("Creating target directory: %s", target_dir)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        start_time = time.time()
                        file_count = 0
                        for file in dcom_dir.iterdir():
                            shutil.copy(str(file), str(target_dir))
                            file_count += 1
                        elapsed_time = time.time() - start_time
                        logging.info("Moved %d files from %s to %s in %.2f seconds", file_count, dcom_dir, target_dir,
                                     elapsed_time)
                    else:
                        logging.info("Target directory already exists: %s", target_dir)
                else:
                    logging.warning("DICOM directory not found: %s", dcom_dir)


def run():
    parser = argparse.ArgumentParser(description='Process CT scan archives')
    parser.add_argument('action', help='Action to run: extract, collect, move', choices=['extract', 'collect', 'move'])
    parser.add_argument('--extract', help='Extract archives')
    parser.add_argument('--collect', help='Collect metadata')
    parser.add_argument('--recollect', help='Recollect')

    args = parser.parse_args()

    if args.action == 'extract':
        extract_all()
    elif args.action == 'collect':
        collect_meta()
    elif args.action == 'move':
        move_dataset()
    else:
        parser.print_help()


if __name__ == '__main__':
    run()
