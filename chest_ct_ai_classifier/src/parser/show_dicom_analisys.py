import argparse
from pathlib import Path
import logging
import time
from chest_ct_ai_classifier.src.utils.metadata_extraction import analyze_dicom_series


def main():
    parser = argparse.ArgumentParser(description='Analyze DICOM series')
    parser.add_argument('--dicom-path', type=str, required=True,
                        help='Path to DICOM directory')

    args = parser.parse_args()
    dicom_path = Path(args.dicom_path)

    if not dicom_path.exists():
        logging.error(f"Directory does not exist: {dicom_path}")
        return

    start_time = time.time()
    analyze_dicom_series(dicom_path)
    elapsed_time = time.time() - start_time
    print("=" * 20)
    print(f"Analysis completed in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
