#!/usr/bin/env python3
"""
prepare_ct_tensors.py

Подготовка DICOM-серий в тензоры для 3D-модели (EfficientNet-3D и т.п.).
- Создает для каждого исследования 3 окна (lungs, mediastinum, bones) и сохраняет их как отдельные .pt файлы
- Выполняет ресэмплинг в изотропный spacing (по умолчанию 1x1x1 mm)
- Выполняет crop/pad/interpolate до целевого (D,H,W)
- Сохраняет данные в float16 (half)

Пример:
python prepare_ct_tensors.py /data/studies --output ./tensors --depth 128 --height 256 --width 256
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List

class DICOMProcessor:
    def __init__(self,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_shape: Tuple[int, int, int] = (128, 256, 256),
                 window_settings: Dict[str, Tuple[int, int]] = None):
        self.input_folder = Path('../data/processed/train')
        self.target_spacing = target_spacing
        self.target_shape = target_shape  # (D, H, W)
        self.window_settings = window_settings or {
            'lungs': (-600, 1500),
            'mediastinum': (40, 400),
            'bones': (400, 1500)
        }
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder {self.input_folder} does not exist")

    def load_dicom_series(self, dicom_folder: Path) -> sitk.Image:
        """Загрузка DICOM серии с помощью SimpleITK"""
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
        if not series_IDs:
            raise FileNotFoundError(f"No DICOM series found in {dicom_folder}")
        # берем первую серию (обычно одна)
        series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        return image

    def apply_hu_transform(self, image_array: np.ndarray, sample_dicom: pydicom.FileDataset) -> np.ndarray:
        """Применение HU преобразования с учетом метаданных RescaleSlope/Intercept"""
        intercept = float(getattr(sample_dicom, 'RescaleIntercept', 0.0))
        slope = float(getattr(sample_dicom, 'RescaleSlope', 1.0))
        # image_array из SimpleITK обычно уже в "raw" значениях, применяем slope/intercept
        hu = image_array.astype(np.float32) * slope + intercept
        return hu

    def apply_window(self, hu_image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """Оконное преобразование и нормализация к [0,1]"""
        w_min = window_center - window_width / 2.0
        w_max = window_center + window_width / 2.0
        windowed = np.clip(hu_image, w_min, w_max)
        normalized = (windowed - w_min) / (w_max - w_min)
        return normalized.astype(np.float32)

    def resample_to_isotropic(self, image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
        """Ресемплирование SimpleITK image к целевому spacing (x,y,z)"""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        # SimpleITK size ordering is (x, y, z)
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled = resampler.Execute(image)
        return resampled

    def center_crop(self, np_vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
        """Центральный крап по (D,H,W). np_vol shape = (z,y,x)"""
        z, y, x = np_vol.shape
        td, th, tw = target
        start_z = max((z - td) // 2, 0)
        start_y = max((y - th) // 2, 0)
        start_x = max((x - tw) // 2, 0)
        end_z = start_z + td
        end_y = start_y + th
        end_x = start_x + tw
        cropped = np_vol[start_z:end_z, start_y:end_y, start_x:end_x]
        return cropped

    def pad_to_min(self, np_vol: np.ndarray, min_shape: Tuple[int, int, int]) -> np.ndarray:
        """Симметричный паддинг до min_shape (если текущая размерность меньше)"""
        z, y, x = np_vol.shape
        mz, my, mx = min_shape
        pad_z = max(mz - z, 0)
        pad_y = max(my - y, 0)
        pad_x = max(mx - x, 0)
        # распределяем паддинг симметрично
        pad_before_z = pad_z // 2
        pad_after_z = pad_z - pad_before_z
        pad_before_y = pad_y // 2
        pad_after_y = pad_y - pad_before_y
        pad_before_x = pad_x // 2
        pad_after_x = pad_x - pad_before_x
        padded = np.pad(np_vol,
                        ((pad_before_z, pad_after_z),
                         (pad_before_y, pad_after_y),
                         (pad_before_x, pad_after_x)),
                        mode='constant', constant_values=0)
        return padded

    def resize_tensor_trilinear(self, tensor: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Интерполирует tensor (1, z, y, x) до точного target_shape (D,H,W) с помощью trilinear.
        Возвращает tensor (1, D, H, W).
        """
        if tensor.ndim != 4:
            raise ValueError("tensor must be (1, z, y, x)")
        # добавляем batch dim: (N=1, C=1, D, H, W)
        t = tensor.unsqueeze(0)  # (1,1,D,H,W)
        t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
        t = t.squeeze(0)  # (1, D, H, W)
        return t

    def process_study(self, study_folder: Path) -> Dict[str, torch.Tensor]:
        """Обработка одного исследования -> возвращает словарь {window_name: tensor(1,D,H,W)}"""
        dicom_folder = study_folder
        if not dicom_folder.exists():
            raise FileNotFoundError(f"DICOM folder not found: {dicom_folder}")

        sitk_image = self.load_dicom_series(dicom_folder)
        resampled = self.resample_to_isotropic(sitk_image, self.target_spacing)
        image_array = sitk.GetArrayFromImage(resampled)  # shape = (z, y, x)

        # берём один dcm файл для получения RescaleSlope/Intercept (в большинстве серий одинаковы)
        dicom_files = [f for f in dicom_folder.iterdir() if f.is_file()]
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files in {dicom_folder}")
        sample_dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)

        hu_array = self.apply_hu_transform(image_array, sample_dcm)  # float32 numpy (z,y,x)

        # Сначала центр-крап если значительно больше, затем паддинг, затем интерполяция
        td, th, tw = self.target_shape

        # если размер >> target, выполняем центр-крап сначала
        z0, y0, x0 = hu_array.shape
        crop_thresh = 1.5  # если размер больше 1.5x целевого — делаем crop
        if z0 > int(td * crop_thresh) or y0 > int(th * crop_thresh) or x0 > int(tw * crop_thresh):
            hu_array = self.center_crop(hu_array, (min(z0, td), min(y0, th), min(x0, tw)))

        # паддим, если необходимо
        hu_array = self.pad_to_min(hu_array, (td, th, tw))

        tensors = {}
        for window_name, (center, width) in self.window_settings.items():
            windowed = self.apply_window(hu_array, center, width)  # (z,y,x) float32 in [0,1]
            # конвертируем в torch tensor shape (1,z,y,x)
            tensor = torch.from_numpy(windowed).unsqueeze(0)  # (1,z,y,x)
            # теперь интерполируем до точного target (D,H,W)
            tensor = self.resize_tensor_trilinear(tensor, (td, th, tw))  # (1, D, H, W)
            # нормализация: z-score по исследованию (можно заменить на глобальные mean/std)
            mean = tensor.mean()
            std = tensor.std(unbiased=False)
            if std == 0:
                std = 1.0
            tensor = (tensor - mean) / std
            tensors[window_name] = tensor  # float32 tensor
        return tensors

    def process_and_save_all_studies(self, output_dir: str):
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)

        study_folders = sorted([f for f in self.input_folder.iterdir() if f.is_dir()])
        print(f"Found {len(study_folders)} studies in {self.input_folder}")

        for i, study in enumerate(study_folders):
            print(f"[{i + 1}/{len(study_folders)}] Processing {study.name} ...")
            try:
                tensors = self.process_study(study)
                for window_name, tensor in tensors.items():
                    fname = outp / f"{study.name}_{window_name}.pt"
                    torch.save(tensor.half(), fname)
                    print(f"  Saved {fname}")
            except Exception as ex:
                print(f"  Error processing {study.name}: {ex}")


    def save_tensors(self, tensors_dict: Dict[str, List[torch.Tensor]], output_dir: str, meta_filename: str = "meta.json"):
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        meta = {}
        for window_name, tensor_list in tensors_dict.items():
            if not tensor_list:
                continue
            batch = torch.stack(tensor_list, dim=0)  # (N,1,D,H,W)
            # сохраняем в float16
            batch_half = batch.half()
            fname = outp / f"{window_name}_tensors.pt"
            torch.save(batch_half, str(fname))
            meta[window_name] = {
                "file": str(fname.name),
                "count": batch_half.shape[0],
                "dtype": "float16",
                "shape_per_sample": list(batch_half.shape[1:])  # (1,D,H,W)
            }
            print(f"Saved {batch_half.shape[0]} samples for window '{window_name}' -> {fname} (dtype float16)")
        # сохраняем мета-информацию
        meta_path = outp / meta_filename
        with open(meta_path, "w") as f:
            json.dump({
                "target_spacing": list(self.target_spacing),
                "target_shape": list(self.target_shape),
                "windows": self.window_settings,
                "meta": meta
            }, f, indent=2)
        print(f"Saved metadata -> {meta_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Prepare DICOM studies into 3D tensors for EfficientNet-3D")

    p.add_argument("--output", type=str, default="./data/data_tensors/studies", help="Output folder for .pt files")
    p.add_argument("--spacing", type=float, nargs=3, default=(1.0,1.0,1.0), help="Target isotropic spacing (x y z) in mm")
    p.add_argument("--depth", type=int, default=128, help="Target depth (D) in slices")
    p.add_argument("--height", type=int, default=256, help="Target height (H)")
    p.add_argument("--width", type=int, default=256, help="Target width (W)")
    return p.parse_args()

def main():
    args = parse_args()

    proc = DICOMProcessor(
        target_spacing=tuple(args.spacing),
        target_shape=(args.depth, args.height, args.width)
    )
    proc.process_and_save_all_studies(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
