import argparse
import csv
import logging
import shutil
import tarfile
import time
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()


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
    except PermissionError:
        logging.error("Permission denied when processing file: %s", file)
    except Exception:
        logging.exception("Error processing file: %s", file)


def write_to_csv(result, output_file, clear_file=False):
    fieldnames = ['filename', 'label']

    if not Path(output_file).exists():
        # Создаем/перезаписываем файл с заголовком
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(result)
    else:
        # Добавляем строку к существующему файлу
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(result)


def run():
    parser = argparse.ArgumentParser(description='Move CT dicom files to target dataset')
    parser.add_argument('--source', help='Source directory')
    parser.add_argument('--target', help='Target directory')
    parser.add_argument('--studies_csv', help='CSV file with studies list')
    parser.add_argument('--recollect', help='Recollect')
    parser.add_argument('--csv_delim', default=',', help='csv delimeter')
    parser.add_argument('--csv_encoding', default='utf-8', help='csv encoding')

    args = parser.parse_args()

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
    studies_csv_file = Path(args.studies_csv)
    if not studies_csv_file.exists():
        logging.error("Studies CSV file does not exist: %s", studies_csv_file)
        exit(1)

    # Читаем файл реестра
    with open(studies_csv_file, 'r', encoding=args.csv_encoding) as csvfile:
        # Пропускаем заголовок если есть
        csvfile = csv.DictReader(csvfile, delimiter=args.csv_delim)
        rows = list(csvfile)

    def move_dicom_files(dicom_series_path, target_dir):
        start_time = time.time()
        files = list(dicom_series_path.iterdir())
        for file in files:
            if file.is_dir():
                if len(files) > 1:
                    logging.error("  🙅🚫⛔  Found %d series folders in: %s", len(files), dcom_dir)
                    return False
                return move_dicom_files(file, target_dir)
            shutil.copy(str(file), str(target_dir))
        elapsed_time = time.time() - start_time
        logging.info("  🚀  Moved %d files from %s to %s in %.2f seconds", len(files), dcom_dir, target_dir,
                     elapsed_time)
        return True

    for row in rows:
        study_id = row.get('id', '').strip()
        logging.info("[%d/%d] Processing study ID: %s", rows.index(row) + 1, len(rows), study_id)
        dcom_dir = source_path / study_id
        if not dcom_dir.exists():
            dcom_tar = source_path / (study_id + '.tar.gz')
            if not dcom_tar.exists():
                logging.warning("  ❌  DICOM directory and archive not found for study_id: %s", dcom_dir)
                continue
            extract_archive(dcom_tar, source_path)
        if dcom_dir.exists():
            logging.info("  🕵️  Found DICOM directory: %s", dcom_dir)
            target_dir = target_path / study_id
            if not target_dir.exists() or len(list(target_dir.iterdir())) == 0:
                logging.info("  👶  Creating target directory: %s", target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                if move_dicom_files(dcom_dir, target_dir):
                    write_to_csv({
                        'filename': study_id + '.pt',
                        'label': row.get('patology')
                    }, target_path / 'labels.csv')
            else:
                logging.info("  😏  Target directory already exists: %s", target_dir)
        else:
            logging.warning("  🤔  DICOM directory not found: %s", dcom_dir)


if __name__ == '__main__':
    run()
