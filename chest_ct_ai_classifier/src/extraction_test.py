import sys
import os

from pydicom import dcmread

from utils.metadata_extraction import analyze_dicom_series
from utils.data_analysis import get_voxel_dimensions

def main():
    if len(sys.argv) != 2:
        print("Usage: python extraction_test.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)


    analyze_dicom_series(folder_path)

    """
        Анализирует серию DICOM-файлов и выводит подробную информацию о параметрах исследования.
        Поддерживает Enhanced CT.
        """
    dicom_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    dicom_files.sort()

    print(f"Найдено {len(dicom_files)} DICOM файлов")

    if not dicom_files:
        return pd.DataFrame()

    # Читаем первый файл
    first_file = os.path.join(folder_path, dicom_files[0])
    ds = dcmread(first_file, force=True)
    print(get_voxel_dimensions(ds))

if __name__ == "__main__":
    main()
