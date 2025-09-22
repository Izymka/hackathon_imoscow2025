#!/usr/bin/env python3
"""
prepare_ct_tensors.py

Исправленная версия с правильной нормализацией данных.
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

# Глобальные константы для нормализации (задаются отдельно)
GLOBAL_MEAN_LUNGS = 0.0
GLOBAL_STD_LUNGS = 1.0
GLOBAL_MEAN_ORGANS = 0.0
GLOBAL_STD_ORGANS = 1.0
GLOBAL_MEAN_BONES = 0.0
GLOBAL_STD_BONES = 1.0


class DICOMProcessor:
    def __init__(self,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_shape: Tuple[int, int, int] = (128, 160, 160),
                 window_settings: Dict[str, Tuple[int, int]] = None,
                 normalization_mode: str = "global_zscore"):
        """
        Args:
            normalization_mode:
                - "window_only": только windowing в [0,1]
                - "global_zscore": z-score по глобальной статистике (рекомендуется)
                - "hu_clip": клиппинг HU + нормализация в [-1,1]
        """
        self.input_folder = Path('../data/processed/train')
        self.target_spacing = target_spacing
        self.target_shape = target_shape  # (D, H, W) = (срезы, высота, ширина)
        self.normalization_mode = normalization_mode

        self.window_settings = window_settings or {
            'lungs': (-600, 1500),  # center, width
            'organs': (40, 400),  # медиастинум и органы
            'bones': (400, 1500)  # кости
        }

        # Глобальная статистика для z-score
        self.global_stats = {
            'lungs': {'mean': GLOBAL_MEAN_LUNGS, 'std': GLOBAL_STD_LUNGS},
            'organs': {'mean': GLOBAL_MEAN_ORGANS, 'std': GLOBAL_STD_ORGANS},
            'bones': {'mean': GLOBAL_MEAN_BONES, 'std': GLOBAL_STD_BONES}
        }

        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder {self.input_folder} does not exist")

    def load_dicom_series(self, dicom_folder: Path) -> sitk.Image:
        """Загрузка DICOM серии с помощью SimpleITK"""
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(dicom_folder))
        if not series_IDs:
            raise FileNotFoundError(f"No DICOM series found in {dicom_folder}")
        series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_IDs[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        return image

    def apply_hu_transform(self, image_array: np.ndarray, sample_dicom: pydicom.FileDataset) -> np.ndarray:
        """Применение HU преобразования с учетом метаданных RescaleSlope/Intercept"""
        intercept = float(getattr(sample_dicom, 'RescaleIntercept', 0.0))
        slope = float(getattr(sample_dicom, 'RescaleSlope', 1.0))
        hu = image_array.astype(np.float32) * slope + intercept
        return hu

    def apply_window_normalization(self, hu_image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """Оконное преобразование и нормализация к [0,1]"""
        w_min = window_center - window_width / 2.0
        w_max = window_center + window_width / 2.0
        windowed = np.clip(hu_image, w_min, w_max)
        normalized = (windowed - w_min) / (w_max - w_min)
        return normalized.astype(np.float32)

    def clip_to_hu_range(self, windowed_image: np.ndarray) -> np.ndarray:
        """Обрезка оконного изображения к диапазону [-1000, 1000]"""
        # Оконное изображение в [0,1], преобразуем в HU диапазон [-1000, 1000]
        hu_min, hu_max = -1000.0, 1000.0
        hu_image = windowed_image * (hu_max - hu_min) + hu_min
        clipped = np.clip(hu_image, hu_min, hu_max)
        return clipped.astype(np.float32)

    def normalize_with_global_stats(self, hu_image: np.ndarray, window_name: str) -> np.ndarray:
        """Нормализация с использованием глобальных значений mean и std"""
        stats = self.global_stats.get(window_name, {'mean': 0.0, 'std': 1.0})
        normalized = (hu_image - stats['mean']) / stats['std']
        return normalized.astype(np.float32)

    def apply_normalization(self, data: torch.Tensor, window_name: str) -> torch.Tensor:
        """Применение выбранного метода нормализации"""
        if self.normalization_mode == "window_only":
            # Данные уже нормализованы windowing в [0,1]
            return data

        elif self.normalization_mode == "global_zscore":
            # Z-score с глобальной статистикой
            stats = self.global_stats.get(window_name, {'mean': 0.0, 'std': 1.0})
            return (data - stats['mean']) / stats['std']

        elif self.normalization_mode == "hu_clip":
            # Для режима hu_clip нормализация уже применена
            return data

        else:
            return data

    def resample_to_isotropic(self, image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
        """Ресемплирование SimpleITK image к целевому spacing (x,y,z)"""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
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
        """Центральный кроп по (D,H,W). np_vol shape = (z,y,x)"""
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
        """Симметричный паддинг до min_shape"""
        z, y, x = np_vol.shape
        mz, my, mx = min_shape
        pad_z = max(mz - z, 0)
        pad_y = max(my - y, 0)
        pad_x = max(mx - x, 0)

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
        """Интерполяция tensor (1, z, y, x) до target_shape (D,H,W)"""
        if tensor.ndim != 4:
            raise ValueError("tensor must be (1, z, y, x)")
        t = tensor.unsqueeze(0)  # (1,1,D,H,W)
        t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
        t = t.squeeze(0)  # (1, D, H, W)
        return t

    def process_study(self, study_folder: Path) -> torch.Tensor:
        """Обработка одного исследования - возвращает трехканальный тензор формата [1, 3, 128, 160, 160]"""
        dicom_folder = study_folder
        if not dicom_folder.exists():
            raise FileNotFoundError(f"DICOM folder not found: {dicom_folder}")

        # Загрузка и ресемплинг
        sitk_image = self.load_dicom_series(dicom_folder)
        resampled = self.resample_to_isotropic(sitk_image, self.target_spacing)
        image_array = sitk.GetArrayFromImage(resampled)  # shape = (z, y, x)

        # Получение HU-значений
        dicom_files = [f for f in dicom_folder.iterdir() if f.is_file()]
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files in {dicom_folder}")
        sample_dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
        hu_array = self.apply_hu_transform(image_array, sample_dcm)

        # Пространственные преобразования
        td, th, tw = self.target_shape
        z0, y0, x0 = hu_array.shape
        crop_thresh = 1.5

        if z0 > int(td * crop_thresh) or y0 > int(th * crop_thresh) or x0 > int(tw * crop_thresh):
            hu_array = self.center_crop(hu_array, (min(z0, td), min(y0, th), min(x0, tw)))

        hu_array = self.pad_to_min(hu_array, (td, th, tw))

        # Создание трехканального тензора
        channels = []

        if self.normalization_mode == "hu_clip":
            # Для режима hu_clip: оконное преобразование -> обрезка [-1000,1000] -> нормализация
            for window_name, (center, width) in self.window_settings.items():
                # 1. Оконное преобразование
                windowed = self.apply_window_normalization(hu_array, center, width)

                # 2. Обрезка к диапазону [-1000, 1000]
                hu_clipped = self.clip_to_hu_range(windowed)

                # 3. Нормализация с глобальными значениями
                normalized = self.normalize_with_global_stats(hu_clipped, window_name)

                # Преобразование в tensor и изменение размера
                tensor = torch.from_numpy(normalized).unsqueeze(0)  # (1, z, y, x)
                tensor = self.resize_tensor_trilinear(tensor, (td, th, tw))  # (1, D, H, W)
                channels.append(tensor)
        else:
            # Для других режимов: оконное преобразование -> нормализация
            for window_name, (center, width) in self.window_settings.items():
                # 1. Оконное преобразование
                windowed = self.apply_window_normalization(hu_array, center, width)

                # Преобразование в tensor и изменение размера
                tensor = torch.from_numpy(windowed).unsqueeze(0)  # (1, z, y, x)
                tensor = self.resize_tensor_trilinear(tensor, (td, th, tw))  # (1, D, H, W)

                # 2. Применение финальной нормализации
                tensor = self.apply_normalization(tensor, window_name)

                channels.append(tensor)

        # Объединяем каналы в трехканальный тензор (3, D, H, W)
        multi_channel_tensor = torch.cat(channels, dim=0)

        # Преобразуем в формат [1, 3, D, H, W]
        final_tensor = multi_channel_tensor.unsqueeze(0)

        return final_tensor

    def compute_global_statistics(self, output_stats_file: str = None):
        """Вычисление глобальной статистики для z-score нормализации"""
        print("Computing global statistics...")

        stats = {window: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for window in self.window_settings}

        study_folders = sorted([f for f in self.input_folder.iterdir() if f.is_dir()])

        # Временно меняем режим нормализации
        original_mode = self.normalization_mode
        self.normalization_mode = "window_only"

        for i, study in enumerate(study_folders):
            if i % 10 == 0:
                print(f"Processing {i + 1}/{len(study_folders)} studies...")
            try:
                # Обрабатываем как отдельные каналы для сбора статистики
                dicom_folder = study
                sitk_image = self.load_dicom_series(dicom_folder)
                resampled = self.resample_to_isotropic(sitk_image, self.target_spacing)
                image_array = sitk.GetArrayFromImage(resampled)

                dicom_files = [f for f in dicom_folder.iterdir() if f.is_file()]
                sample_dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
                hu_array = self.apply_hu_transform(image_array, sample_dcm)

                td, th, tw = self.target_shape
                z0, y0, x0 = hu_array.shape
                if z0 > int(td * 1.5) or y0 > int(th * 1.5) or x0 > int(tw * 1.5):
                    hu_array = self.center_crop(hu_array, (min(z0, td), min(y0, th), min(x0, tw)))
                hu_array = self.pad_to_min(hu_array, (td, th, tw))

                for window_name, (center, width) in self.window_settings.items():
                    windowed = self.apply_window_normalization(hu_array, center, width)
                    tensor = torch.from_numpy(windowed).unsqueeze(0)
                    tensor = self.resize_tensor_trilinear(tensor, (td, th, tw))
                    data = tensor.flatten()
                    stats[window_name]['sum'] += data.sum().item()
                    stats[window_name]['sum_sq'] += (data ** 2).sum().item()
                    stats[window_name]['count'] += data.numel()

            except Exception as e:
                print(f"Error processing {study.name}: {e}")

        # Восстанавливаем режим
        self.normalization_mode = original_mode

        # Вычисляем финальную статистику
        for window_name in stats:
            s = stats[window_name]
            if s['count'] > 0:
                mean = s['sum'] / s['count']
                variance = (s['sum_sq'] / s['count']) - (mean ** 2)
                std = np.sqrt(max(variance, 1e-8))
                self.global_stats[window_name] = {'mean': mean, 'std': std}
                print(f"{window_name}: mean={mean:.4f}, std={std:.4f}")

        if output_stats_file:
            with open(output_stats_file, 'w') as f:
                json.dump(self.global_stats, f, indent=2)
            print(f"Global statistics saved to {output_stats_file}")

    def load_global_statistics(self, stats_file: str):
        """Загрузка предвычисленной глобальной статистики"""
        with open(stats_file, 'r') as f:
            self.global_stats = json.load(f)
        print(f"Loaded global statistics from {stats_file}")

    def process_and_save_all_studies(self, output_dir: str):
        """Обработка и сохранение всех исследований"""
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)

        study_folders = sorted([f for f in self.input_folder.iterdir() if f.is_dir()])
        print(f"Found {len(study_folders)} studies in {self.input_folder}")
        print(f"Normalization mode: {self.normalization_mode}")

        for i, study in enumerate(study_folders):
            print(f"[{i + 1}/{len(study_folders)}] Processing {study.name} ...")
            try:
                # Получаем трехканальный тензор формата [1, 3, 128, 160, 160]
                multi_channel_tensor = self.process_study(study)

                # Проверяем формат
                expected_shape = (1, 3, self.target_shape[0], self.target_shape[1], self.target_shape[2])
                if multi_channel_tensor.shape != expected_shape:
                    print(f"  Warning: unexpected tensor shape {multi_channel_tensor.shape}, expected {expected_shape}")

                # Сохраняем трехканальный тензор
                fname = outp / f"{study.name}.pt"
                torch.save(multi_channel_tensor, fname)

                # Выводим статистику для контроля
                if i < 3:  # для первых нескольких исследований
                    print(f"  Tensor shape: {multi_channel_tensor.shape}")
                    print(f"  Min: {multi_channel_tensor.min():.3f}, Max: {multi_channel_tensor.max():.3f}")
                    print(f"  Mean: {multi_channel_tensor.mean():.3f}, Std: {multi_channel_tensor.std():.3f}")

                print(f"  Saved to {fname}")
            except Exception as ex:
                print(f"  Error processing {study.name}: {ex}")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare DICOM studies into 3D tensors")

    p.add_argument("--output", type=str, default="./data/tensors",
                   help="Output folder for .pt files")
    p.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                   help="Target isotropic spacing (x y z) in mm")
    p.add_argument("--depth", type=int, default=128, help="Target depth (D) - number of slices")
    p.add_argument("--height", type=int, default=160, help="Target height (H)")
    p.add_argument("--width", type=int, default=160, help="Target width (W)")
    p.add_argument("--norm", type=str, default="global_zscore",
                   choices=["window_only", "global_zscore", "hu_clip"],
                   help="Normalization method")
    p.add_argument("--compute-stats", action="store_true",
                   help="Compute global statistics for z-score normalization")
    p.add_argument("--stats-file", type=str, default="global_stats.json",
                   help="File to save/load global statistics")

    return p.parse_args()


def main():
    args = parse_args()

    proc = DICOMProcessor(
        target_spacing=tuple(args.spacing),
        target_shape=(args.depth, args.height, args.width),
        normalization_mode=args.norm
    )

    if args.compute_stats:
        proc.compute_global_statistics(args.stats_file)
        return

    if args.norm == "global_zscore" and os.path.exists(args.stats_file):
        proc.load_global_statistics(args.stats_file)

    proc.process_and_save_all_studies(args.output)
    print("Done.")


if __name__ == "__main__":
    main()