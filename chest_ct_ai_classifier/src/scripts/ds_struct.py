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



if __name__ == "__main__":
    main()
