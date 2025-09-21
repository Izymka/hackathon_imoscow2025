import argparse
import csv
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
base_path = Path(os.getenv("DS_CT_LUNGCANCER_500_FILE", default=None))

if not base_path.exists():
    logging.warning("Directory does not exist: %s", base_path)
    exit(1)


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


def extract_all():
    try:
        dicom_dir = base_path / "dicom"
        if not dicom_dir.exists():
            logging.error("Dicom directory does not exist: %s", dicom_dir)
            return
        files = list(dicom_dir.iterdir())
        logging.info("Dicom directory exist. Found %d files:", len(files))
        for file in files:
            logging.info(file.name)
            extract_archive(file, dicom_dir)
    except PermissionError:
        logging.error("Permission denied accessing directory: %s", base_path)
    except Exception:
        logging.exception("Error accessing directory: %s", base_path)




def collect_meta():
    registry_file = base_path / "dataset_registry.csv"
    protocols_dir = base_path / "protocols"
    output_file = base_path / "pathology_results.csv"

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

    results = []

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

                # Ищем соответствующий JSON файл
                json_file = protocols_dir / f"{study_id}.json"

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

                        results.append(result)
                        logging.info("Обработан файл: %s, патология: %s", study_id, pathology)
                    except Exception as e:
                        logging.error("Ошибка обработки файла %s: %s", json_file, e)
                        result = {'id': study_id, 'patology': -1}
                        for i in range(1, 7):
                            result[f"doctor_{i}_comment"] = ""
                        results.append(result)
                else:
                    logging.warning("JSON файл не найден: %s", json_file)
                    result = {'id': study_id, 'patology': -2}
                    for i in range(1, 7):
                        result[f"doctor_{i}_comment"] = ""
                    results.append(result)

        # Записываем результаты в CSV файл
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'patology', 'doctor_1_comment', 'doctor_2_comment',
                          'doctor_3_comment', 'doctor_4_comment', 'doctor_5_comment', 'doctor_6_comment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        logging.info("Результаты сохранены в файл: %s", output_file)
        logging.info("Обработано записей: %d", len(results))

    except Exception as e:
        logging.error("Ошибка при обработке файлов: %s", e)


def run():
    parser = argparse.ArgumentParser(description='Process CT scan archives')
    parser.add_argument('--extract', action='store_true', help='Extract archives')
    parser.add_argument('--collect', action='store_true', help='Collect metadata')

    args = parser.parse_args()

    if args.extract:
        extract_all()
    elif args.collect:
        collect_meta()
    else:
        parser.print_help()


if __name__ == '__main__':
    run()
