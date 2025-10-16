import argparse
import csv
from datetime import datetime
import gzip
import io
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from chest_ct_ai_classifier.src.utils.dicom_parser import parse_dicom

logging.basicConfig(level=logging.INFO)
load_dotenv()

USE_ROBOCOPY = os.name == 'nt'
if USE_ROBOCOPY:
    import subprocess
    check_network = subprocess.run(['net', 'use'], capture_output=True, text=True)


def extract_archive(file, dicom_path):
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

            # handle .gz archives (not .tar.gz which is already handled above)
            elif name_lower.endswith(".gz") and not (name_lower.endswith(".tar.gz")):
                base_name = file.name[:-3]  # strip ".gz"
                target_file = dicom_path / base_name

                if not target_file.exists():
                    logging.info("GZ archive found: %s. Extracting to: %s", file.name, target_file)
                    try:
                        with gzip.open(file, 'rb') as gz_file:
                            with open(target_file, 'wb') as out_file:
                                shutil.copyfileobj(gz_file, out_file)
                        logging.info("GZ extraction completed: %s", target_file)
                    except Exception:
                        logging.exception("Failed to extract gz archive: %s", file)
                else:
                    logging.info("Already extracted: %s", target_file)
    except PermissionError:
        logging.error("Permission denied when processing file: %s", file)
    except Exception:
        logging.exception("Error processing file: %s", file)


def write_to_csv(result, output_file, fieldnames):
    output_file = Path(output_file)
    file_exists = output_file.exists()

    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def is_google_spreadsheet_url(url_or_path):
    """Проверяет, является ли переданный путь ссылкой на Google Spreadsheet"""
    if isinstance(url_or_path, str):
        return url_or_path.startswith('https://docs.google.com/spreadsheets/')
    return False


def convert_google_sheet_url_to_csv(google_sheet_url):
    """Конвертирует ссылку на Google Spreadsheet в ссылку для скачивания CSV"""
    if '/edit' in google_sheet_url:
        # Извлекаем ID документа
        sheet_id = google_sheet_url.split('/d/')[1].split('/')[0]
        return f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
    return google_sheet_url


def download_csv_from_google_sheet(google_sheet_url, encoding='utf-8'):
    """Скачивает CSV данные из Google Spreadsheet и возвращает список строк"""
    try:
        csv_url = convert_google_sheet_url_to_csv(google_sheet_url)
        logging.info("Загрузка данных из Google Spreadsheet: %s", csv_url)

        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()

        # Декодируем содержимое с указанной кодировкой
        csv_content = response.content.decode(encoding)

        # Читаем CSV из строки
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)

        logging.info("Успешно загружено %d строк из Google Spreadsheet", len(rows))
        return rows

    except requests.exceptions.RequestException as e:
        logging.error("Ошибка при загрузке Google Spreadsheet: %s", e)
        raise
    except Exception as e:
        logging.error("Ошибка при обработке CSV данных из Google Spreadsheet: %s", e)
        raise



def run():
    parser = argparse.ArgumentParser(description='Move CT dicom files to target dataset')
    parser.add_argument('--source', help='Source directory')
    parser.add_argument('--target', help='Target directory')
    parser.add_argument('--studies_csv', help='CSV file with studies list')
    parser.add_argument('--recollect', help='Recollect')
    parser.add_argument('--csv_delim', default=',', help='csv delimeter')
    parser.add_argument('--csv_encoding', default='utf-8', help='csv encoding')
    parser.add_argument('--read-study-id-from-dicom', default=False)
    parser.add_argument('--transfer', default=True)
    parser.add_argument('--labels-only', default=False)
    parser.add_argument("--multiple-series-strategy",
                        default="skip",
                        choices=["skip", "first", "rich", "largest"],
                        help="How to handle multiple series in a study")
    parser.add_argument('--labels', default='binary', choices=['binary', 'all'],
                        help='What to put into labels.csv: binary (patology/pathology) or all (all columns from source)')

    args = parser.parse_args()

    config_logger()

    reading_study_id_from_dicom = args.read_study_id_from_dicom
    do_transfer = str(args.transfer).lower() in ('true', '1', 't', 'y', 'yes')

    source_path = Path(args.source)
    if not source_path.exists():
        logging.error("Source path does not exist: %s", source_path)
        exit(1)
    target_path = Path(args.target)
    if not target_path.exists():
        if target_path.parent.exists():
            # create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            logging.info("Created target directory: %s", target_path)
        else:
            logging.error("Target path and parent does not exist: %s", target_path)
            exit(1)
    
    # Определяем тип источника данных и читаем файл реестра
    if is_google_spreadsheet_url(args.studies_csv):
        # Загружаем данные из Google Spreadsheet
        try:
            rows = download_csv_from_google_sheet(args.studies_csv, args.csv_encoding)
        except Exception as e:
            logging.error("Не удалось загрузить данные из Google Spreadsheet: %s", e)
            exit(1)
    else:
        # Обрабатываем как локальный файл
        studies_csv_file = Path(args.studies_csv)
        if not studies_csv_file.exists():
            logging.error("Studies CSV file does not exist: %s", studies_csv_file)
            exit(1)

        # Читаем локальный файл реестра
        with open(studies_csv_file, 'r', encoding=args.csv_encoding) as csvfile:
            csvfile = csv.DictReader(csvfile, delimiter=args.csv_delim)
            rows = list(csvfile)

    # Определяем поля для выходного CSV в зависимости от режима меток
    if args.labels == 'binary':
        output_fieldnames = ['filename', 'label']
    else:
        # Собираем упорядоченное объединение всех колонок из исходного CSV
        ordered_columns = []
        seen = set()
        for r in rows:
            for k in (r.keys() if isinstance(r, dict) else []):
                if k not in seen:
                    seen.add(k)
                    ordered_columns.append(k)
        output_fieldnames = ['filename'] + ordered_columns
    logging.info("Режим сохранения меток: %s. Поля итогового CSV: %s", args.labels, output_fieldnames)

    def move_dicom_files(dicom_series_path, target_dir):
        dicom_series_path = Path(os.path.normpath(str(dicom_series_path)))
        start_time = time.time()
        dirs = [f for f in dicom_series_path.iterdir() if f.is_dir()]
        if dirs:
            if len(dirs) > 1 or args.multiple_series_strategy == "first":
                if args.multiple_series_strategy == "skip":
                    logging.warning("  🙅  Found %d series folders in: %s. Skipping", len(dirs), dicom_series_path)
                    return False
                if args.multiple_series_strategy == "largest":
                    # Find directory with most files
                    largest_dir = max(dirs, key=lambda d: len([f for f in d.iterdir() if not f.name.startswith('.')]))
                    logging.info("  📁  Selected largest series folderfrom %d folders: %s",len(dirs), largest_dir.name)
                    return move_dicom_files(largest_dir, target_dir)
            return move_dicom_files(dirs[0], target_dir)
        files = [f for f in dicom_series_path.iterdir() if not f.name.startswith('.')]
        if not files:
            logging.error("  🚫  No files found in: %s", dicom_series_path)
            return False
        if USE_ROBOCOPY:
            subprocess.run(['robocopy', str(dicom_series_path), str(target_dir)], capture_output=True)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy(str(file), str(target_dir))
        elapsed_time = time.time() - start_time
        logging.info("  🚀  Moved %d files from %s to %s in %.2f seconds", len(files), dicom_series_path, target_dir,
                     elapsed_time)
        return True

    # Инициализация статистики
    start_time = time.time()
    stats = {
        'total': 0,
        'processed': 0,
        'skipped_no_dicom': 0,
        'skipped_no_study_id': 0,
        'skipped_multiple_series': 0,
        'already_exists': 0,
        'errors': 0
    }

    i = 0
    for row in rows:
        i += 1
        nii_gz_train_set=False
        id_labels = ['id', 'VolumeName']
        study_id = (row.get('id', '') or row.get('VolumeName')).strip()
        if study_id.endswith('.nii.gz'):
            study_id = study_id[:-3]
            nii_gz_train_set=True

        label_value = row.get('patology') or row.get('pathology')
        stats['total'] += 1
        logging.info("[%d/%d] Processing study ID: %s", i, len(rows), study_id)

        try:
            dicom_dir = source_path / study_id
            if not dicom_dir.exists():

                if nii_gz_train_set:
                    # Разбиваем study_id на части по подчеркиванию
                    # Например: train_100_a_2 -> ['train', '100', 'a', '2']
                    parts = study_id.split('_')
                    if len(parts) >= 3:
                        # Формируем первую поддиректорию: train_100
                        first_subdir = '_'.join(parts[:2])
                        # Формируем вторую поддиректорию: train_100_a
                        second_subdir = '_'.join(parts[:3])
                        # Итоговый путь: source_path / train_100 / train_100_a / train_100_a_2.nii.gz
                        dcom_tar = source_path / first_subdir / second_subdir / (study_id + '.gz')
                    else:
                        # Если формат не соответствует ожидаемому, используем простой путь
                        dcom_tar = source_path / 'train' / (study_id + '.nii.gz')
                else:
                    dcom_tar = source_path / (study_id + '.tar.gz')
                if not dcom_tar.exists():
                    logging.warning("  ❌  DICOM directory and archive not found for study_id: %s", dicom_dir)
                    stats['skipped_no_dicom'] += 1
                    continue
                if args.labels_only:
                    if args.labels == 'binary':
                        out_row = {
                            'filename': study_id + '.pt',
                            'label': label_value
                        }
                    else:
                        # Переносим все поля из исходной строки + добавляем filename
                        out_row = {'filename': study_id + '.pt'}
                        if isinstance(row, dict):
                            for k, v in row.items():
                                out_row[k] = v
                    write_to_csv(out_row, target_path / 'labels.csv', output_fieldnames)
                    continue
                extract_archive(dcom_tar, source_path)
                
            if reading_study_id_from_dicom:
                summary = parse_dicom(dicom_dir)
                study_id = summary.study_uid
                if not study_id:
                    logging.error('  ❌  Unable to read study ID from DICOM directory: %s', dicom_dir)
                    stats['skipped_no_study_id'] += 1
                    continue

            if dicom_dir.exists():
                logging.info("  🕵️  Found DICOM directory: %s", dicom_dir)
                if study_id.endswith('.nii'):
                    study_id = study_id[:-4]
                target_dir = target_path / study_id
                if do_transfer:
                    if not target_dir.exists() or len(list(target_dir.iterdir())) == 0:
                        logging.info("  🎯  Target directory: %s", target_dir)
                        move_result = move_dicom_files(dicom_dir, target_dir)
                        if move_result is False:
                            # move_dicom_files returned False (multiple series skip)
                            stats['skipped_multiple_series'] += 1
                        elif move_result:
                            if args.labels == 'binary':
                                out_row = {
                                    'filename': study_id + '.pt',
                                    'label': label_value
                                }
                            else:
                                # Переносим все поля из исходной строки + добавляем filename
                                out_row = {'filename': study_id + '.pt'}
                                if isinstance(row, dict):
                                    for k, v in row.items():
                                        out_row[k] = v
                            write_to_csv(out_row, target_path / 'labels.csv', output_fieldnames)
                            stats['processed'] += 1
                    else:
                        logging.info("  😏  Target directory already exists: %s", target_dir)
                        stats['already_exists'] += 1
                else:
                    logging.info("  ✅ OK [transfer disabled]")
                    stats['processed'] += 1
            else:
                logging.warning("  🤔  DICOM directory not found: %s", dicom_dir)
                stats['skipped_no_dicom'] += 1
        except Exception as e:
            logging.error("  💥  Error processing study %s: %s", study_id, str(e))
            stats['errors'] += 1

    # Вывод итоговой статистики
    end_time = time.time()
    total_time = end_time - start_time

    logging.info("=" * 80)
    logging.info("🎯 ИТОГОВАЯ СТАТИСТИКА ВЫПОЛНЕНИЯ")
    logging.info("=" * 80)
    logging.info("📊 Всего исследований: %d", stats['total'])
    logging.info("✅ Успешно обработано: %d", stats['processed'])
    logging.info("🏠 Уже существует: %d", stats['already_exists'])
    logging.info("❌ Пропущено (нет DICOM): %d", stats['skipped_no_dicom'])
    logging.info("🆔 Пропущено (нет Study ID): %d", stats['skipped_no_study_id'])
    logging.info("📁 Пропущено (множественные серии): %d", stats['skipped_multiple_series'])
    logging.info("💥 Ошибки: %d", stats['errors'])
    logging.info("⏱️ Общее время выполнения: %.2f сек (%.2f мин)", total_time, total_time / 60)

    if stats['total'] > 0:
        success_rate = (stats['processed'] + stats['already_exists']) / stats['total'] * 100
        logging.info("📈 Процент успешности: %.1f%%", success_rate)

    logging.info("=" * 80)


def config_logger():
    # Настройка логирования в файл с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создаем директорию logs, если она не существует
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_filename = logs_dir / f"collect_dataset_{timestamp}.log"

    # Создаем форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Получаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Добавляем обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Добавляем обработчик для файла
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Запуск сбора датасета. Лог файл: %s", log_filename)


if __name__ == '__main__':
    run()
