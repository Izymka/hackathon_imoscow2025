import argparse
import csv
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()


def run():
    parser = argparse.ArgumentParser(description='Сбор статистики по директории исходных исследований на основе списка из CSV')
    parser.add_argument('--source', required=True, help='Путь к исходной директории с исследованиями (папки с id)')
    parser.add_argument('--studies_csv', required=True, help='CSV файл со списком исследований (колонка id)')
    parser.add_argument('--output_csv', required=True, help='Путь к итоговому CSV со статусами')
    parser.add_argument('--csv_delim', default=',', help='Разделитель CSV входного файла')
    parser.add_argument('--csv_encoding', default='utf-8', help='Кодировка CSV входного файла')

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists() or not source_path.is_dir():
        logging.error('Source path does not exist or is not a directory: %s', source_path)
        raise SystemExit(1)

    studies_csv_file = Path(args.studies_csv)
    if not studies_csv_file.exists():
        logging.error('Studies CSV file does not exist: %s', studies_csv_file)
        raise SystemExit(1)

    output_csv = Path(args.output_csv)
    if not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Читаем файл реестра и собираем множество id из колонки "id"
    with open(studies_csv_file, 'r', encoding=args.csv_encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=args.csv_delim)
        csv_rows = list(reader)
        csv_ids = {str(r.get('id', '')).strip() for r in csv_rows if str(r.get('id', '')).strip()}

    # Получаем список вложенных директорий первого уровня в source
    source_dirs = {p.name for p in source_path.iterdir() if p.is_dir()}

    # Сопоставление
    ok_ids = sorted(csv_ids & source_dirs)
    missing_ids = sorted(csv_ids - source_dirs)
    extra_dirs = sorted(source_dirs - csv_ids)

    # Пишем итоговый CSV
    fieldnames = ['id', 'status', 'path']
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # OK
        for sid in ok_ids:
            writer.writerow({'id': sid, 'status': 'ok', 'path': str(source_path / sid)})
        # MISSING
        for sid in missing_ids:
            writer.writerow({'id': sid, 'status': 'missing', 'path': ''})
        # EXTRA
        for d in extra_dirs:
            writer.writerow({'id': d, 'status': 'extra', 'path': str(source_path / d)})

    # Общая статистика
    total_in_csv = len(csv_ids)
    total_in_source = len(source_dirs)
    count_ok = len(ok_ids)
    count_missing = len(missing_ids)
    count_extra = len(extra_dirs)

    logging.info('Статистика сопоставления:')
    logging.info('  Всего ID в CSV: %d', total_in_csv)
    logging.info('  Всего директорий в source: %d', total_in_source)
    logging.info('  Совпало (ok): %d', count_ok)
    logging.info('  Отсутствует в source (missing): %d', count_missing)
    logging.info('  Лишние в source (extra): %d', count_extra)


if __name__ == '__main__':
    run()
