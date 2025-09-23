#!/usr/bin/env python3
"""
CTTensorQualityAssessment для оценки качества тензоров, подготовленных prepare_ct_for_medicalnet.py
"""

import torch
from monai.data.meta_tensor import MetaTensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import os
import glob
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm


warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CTTensorQualityAssessment:
    """
    Класс для оценки качества нормализованных тензоров КТ снимков грудной клетки
    перед подачей в 3D EfficientNet модель
    """

    def __init__(self, expected_shape: Tuple[int, int, int, int, int] = (1, 1, 128, 128, 128)):
        """
        Args:
            expected_shape: Ожидаемая форма тензора (batch, channels, depth, height, width)
                           Для model: (1, 1, 128, 128, 128)
        """
        self.expected_shape = expected_shape
        self.supported_formats = ['.pt', '.pth', '.npy', '.npz']
        self.quality_reports = []

    def load_tensor_from_file(self, file_path: str,
                              target_shape: Optional[Tuple[int, int, int, int, int]] = None) -> torch.Tensor:
        """
        Загрузка тензора из файла различных форматов

        Args:
            file_path: Путь к файлу с тензором
            target_shape: Целевая форма тензора (если нужно изменить размер)

        Returns:
            Загруженный тензор
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        print(f"Загрузка тензора из: {file_path}")

        # Определяем формат файла
        suffix = file_path.suffix.lower()

        if suffix in ['.pt', '.pth']:
            # PyTorch тензоры
            from monai.data.meta_tensor import MetaTensor
            import torch

            # Разрешаем загрузку MetaTensor для совместимости с MONAI
            torch.serialization.add_safe_globals([MetaTensor])

            tensor = torch.load(file_path, map_location='cpu')

        elif suffix == '.npy':
            # NumPy массивы
            array = np.load(file_path)
            tensor = torch.from_numpy(array).float()

        elif suffix == '.npz':
            # Сжатые NumPy массивы
            data = np.load(file_path)
            # Берем первый массив из архива
            key = list(data.keys())[0]
            array = data[key]
            tensor = torch.from_numpy(array).float()

        else:
            raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

        # Проверяем размерность тензора
        print(f"Загружен тензор формы: {tensor.shape}")
        return tensor

    def load_tensors_from_directory(self, directory_path: str,
                                    pattern: str = "*",
                                    max_files: Optional[int] = None,
                                    num_workers: int = 10) -> List[Tuple[torch.Tensor, str]]:
        """
        Загрузка всех тензоров из директории с параллелизмом

        Args:
            directory_path: Путь к директории
            pattern: Паттерн для поиска файлов
            max_files: Максимальное количество файлов для загрузки
            num_workers: Количество параллельных процессов

        Returns:
            Список кортежей (тензор, имя_файла)
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")

        # Ищем файлы поддерживаемых форматов
        all_files = []
        for ext in self.supported_formats:
            files = list(directory_path.glob(f"{pattern}{ext}"))
            all_files.extend(files)

        if not all_files:
            raise ValueError(f"Файлы не найдены в директории: {directory_path}")

        # Ограничиваем количество файлов если указано
        if max_files:
            all_files = all_files[:max_files]

        logger.info(f"Найдено {len(all_files)} файлов для загрузки")

        tensors_and_names = []

        # Параллельная загрузка
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Создаем задачи для загрузки
            future_to_file = {
                executor.submit(self._load_single_file, str(file_path)): file_path
                for file_path in all_files
            }

            # Обрабатываем результаты с прогресс-баром
            with tqdm(total=len(all_files), desc="Загрузка тензоров") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result is not None:
                            tensor, filename = result
                            tensors_and_names.append((tensor, filename))
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {file_path}: {e}")
                    finally:
                        pbar.update(1)

        logger.info(f"Успешно загружено {len(tensors_and_names)} тензоров")
        return tensors_and_names

    @staticmethod
    def _load_single_file(file_path: str) -> Optional[Tuple[torch.Tensor, str]]:
        """Вспомогательный метод для параллельной загрузки одного файла"""
        try:
            file_path_obj = Path(file_path)
            suffix = file_path_obj.suffix.lower()

            if suffix in ['.pt', '.pth']:
                tensor = torch.load(file_path, map_location='cpu')
            elif suffix == '.npy':
                array = np.load(file_path)
                tensor = torch.from_numpy(array).float()
            elif suffix == '.npz':
                data = np.load(file_path)
                key = list(data.keys())[0]
                array = data[key]
                tensor = torch.from_numpy(array).float()
            else:
                return None

            return tensor, file_path_obj.stem
        except Exception:
            return None

    def assess_tensor_quality(self, tensor: torch.Tensor,
                              tensor_name: str = "CT_tensor") -> Dict:
        """
        Полная оценка качества нормализованного тензора для model

        Args:
            tensor: Входной нормализованный тензор КТ снимка
            tensor_name: Название тензора для отчета

        Returns:
            Словарь с метриками качества
        """
        logger.info(f"=== Оценка качества нормализованного тензора: {tensor_name} ===")

        # Базовые проверки
        basic_checks = self._basic_checks(tensor)

        # Статистический анализ
        statistical_metrics = self._statistical_analysis(tensor)

        # Проверка на артефакты и аномалии
        artifact_checks = self._artifact_detection(tensor)

        # Анализ пространственного распределения
        spatial_analysis = self._spatial_analysis(tensor)

        # Проверка готовности для модели model
        model_readiness = self._model_readiness_check(tensor)

        # Объединение всех метрик
        all_metrics = {
            'tensor_name': tensor_name,
            'basic_checks': basic_checks,
            'statistical_metrics': statistical_metrics,
            'artifact_checks': artifact_checks,
            'spatial_analysis': spatial_analysis,
            'model_readiness': model_readiness
        }

        # Общая оценка качества
        overall_quality = self._calculate_overall_quality(all_metrics)
        all_metrics['overall_quality'] = overall_quality

        self.quality_reports.append(all_metrics)
        self._print_report(all_metrics)

        return all_metrics

    def _basic_checks(self, tensor: torch.Tensor) -> Dict:
        """Базовые проверки тензора"""
        checks = {}

        # Проверка формы
        checks['shape'] = tensor.shape
        checks['shape_correct'] = tensor.shape == self.expected_shape

        # Проверка типа данных
        checks['dtype'] = tensor.dtype
        checks['dtype_float'] = tensor.dtype in [torch.float32, torch.float16, torch.float64]

        # Проверка устройства
        checks['device'] = tensor.device

        # Проверка на NaN и Inf
        checks['has_nan'] = torch.isnan(tensor).any().item()
        checks['has_inf'] = torch.isinf(tensor).any().item()

        # Размер в памяти (в МБ)
        checks['memory_mb'] = tensor.numel() * tensor.element_size() / (1024 * 1024)

        logger.info(f"Форма тензора: {checks['shape']} {'✓' if checks['shape_correct'] else '✗'}")
        logger.info(f"Тип данных: {checks['dtype']} {'✓' if checks['dtype_float'] else '✗'}")
        logger.info(f"Устройство: {checks['device']}")
        logger.info(f"Размер в памяти: {checks['memory_mb']:.2f} МБ")
        logger.info(f"NaN значения: {'Есть ✗' if checks['has_nan'] else 'Нет ✓'}")
        logger.info(f"Inf значения: {'Есть ✗' if checks['has_inf'] else 'Нет ✓'}")

        return checks

    def _statistical_analysis(self, tensor: torch.Tensor) -> Dict:
        """
        Статистический анализ нормализованного тензора
        """
        stats = {}

        # Приводим тензор к float32 для всех вычислений
        if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
            tensor_float = tensor.float()
            logger.info(f"Тензор приведен к float32 для статистических вычислений (исходный тип: {tensor.dtype})")
        else:
            tensor_float = tensor

        # Анализ для канала (для model это один канал)
        if len(tensor_float.shape) == 5:  # [batch, channels, depth, height, width]
            channel_data = tensor_float[0, 0]  # первый batch, первый канал
        else:
            channel_data = tensor_float[0] if len(tensor_float.shape) >= 4 else tensor_float

        stats = self._analyze_single_channel(channel_data)

        logger.info("Статистический анализ:")
        logger.info(f"Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                    f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        return stats

    def _analyze_single_channel(self, data: torch.Tensor) -> Dict:
        """Анализ одного канала"""
        stats = {}

        # Основная статистика
        stats['mean'] = float(data.mean())
        stats['std'] = float(data.std())
        stats['min'] = float(data.min())
        stats['max'] = float(data.max())
        stats['median'] = float(data.median())

        # Квантили
        try:
            quantiles = torch.quantile(data.flatten(),
                                       torch.tensor([0.05, 0.25, 0.75, 0.95], dtype=data.dtype))
            stats['q05'] = float(quantiles[0])
            stats['q25'] = float(quantiles[1])
            stats['q75'] = float(quantiles[2])
            stats['q95'] = float(quantiles[3])
        except Exception:
            stats['q05'] = stats['q25'] = stats['q75'] = stats['q95'] = 0.0

        # Дополнительные метрики
        stats['range'] = stats['max'] - stats['min']
        stats['iqr'] = stats['q75'] - stats['q25']  # Межквартильный размах

        # Критерии для нормализованных данных ImageNet
        # После нормализации: (HU - mean) / std, где mean=0.485, std=0.229
        # Это дает диапазон примерно от -2 до +2
        stats['proper_normalization'] = (stats['min'] >= -3.0) and (stats['max'] <= 3.0)
        stats['mean_in_range'] = -1.0 <= stats['mean'] <= 1.0

        # Коэффициент вариации
        stats['cv'] = stats['std'] / abs(stats['mean']) if abs(stats['mean']) > 0.01 else float('inf')

        # Дополнительная диагностика для нормализованных данных
        self._analyze_normalized_distribution(data, stats)

        return stats

    def _analyze_normalized_distribution(self, data: torch.Tensor, metrics: Dict):
        """Дополнительный анализ для нормализованных данных"""

        # Анализ использования динамического диапазона
        used_range = metrics['max'] - metrics['min']
        metrics['range_utilization'] = used_range

        # Проверка на "обрезание" значений (clipping)
        edge_threshold = 0.001  # 0.1% пикселей
        near_min = (data <= (metrics['min'] + 0.1)).float().mean().item()
        near_max = (data >= (metrics['max'] - 0.1)).float().mean().item()

        metrics['clipping_at_extremes'] = (near_min > edge_threshold) or (near_max > edge_threshold)

        # Анализ распределения по диапазонам для нормализованных данных
        very_negative = (data <= -1.5).float().mean().item()  # Очень темные области
        negative = ((data > -1.5) & (data <= 0)).float().mean().item()  # Темные области
        neutral = ((data > 0) & (data <= 1.5)).float().mean().item()  # Средние значения
        positive = (data > 1.5).float().mean().item()  # Яркие области

        metrics['very_negative_ratio'] = very_negative
        metrics['negative_ratio'] = negative
        metrics['neutral_ratio'] = neutral
        metrics['positive_ratio'] = positive

    def _artifact_detection(self, tensor: torch.Tensor) -> Dict:
        """Обнаружение артефактов и аномалий"""
        checks = {}

        if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
            data = tensor[0, 0]  # первый batch, первый канал
        else:
            data = tensor[0] if len(tensor.shape) >= 4 else tensor

        # Проверка на константные срезы
        slice_stds = torch.std(data, dim=(1, 2))
        checks['constant_slices'] = (slice_stds < 1e-6).sum().item()
        checks['constant_slices_ratio'] = checks['constant_slices'] / data.shape[0]

        # Проверка на выбросы (значения > 4 sigma)
        mean_val = torch.mean(data)
        std_val = torch.std(data)
        outliers = torch.abs(data - mean_val) > 4 * std_val
        checks['outliers_count'] = outliers.sum().item()
        checks['outliers_ratio'] = checks['outliers_count'] / data.numel()

        # Проверка однородности по срезам
        slice_means = torch.mean(data, dim=(1, 2))
        checks['slice_mean_std'] = torch.std(slice_means).item()
        checks['uniform_slices'] = checks['slice_mean_std'] < std_val * 0.3

        # Проверка на резкие переходы между срезами
        if data.shape[0] > 1:
            slice_diffs = torch.diff(slice_means)
            checks['max_slice_diff'] = torch.max(torch.abs(slice_diffs)).item()
            checks['smooth_transitions'] = checks['max_slice_diff'] < 3 * checks['slice_mean_std']
        else:
            checks['max_slice_diff'] = 0
            checks['smooth_transitions'] = True

        logger.info("Анализ артефактов:")
        logger.info(f"Константные срезы: {checks['constant_slices_ratio']:.1%}")
        logger.info(f"Выбросы: {checks['outliers_ratio']:.1%}")
        logger.info(f"Плавные переходы: {'✓' if checks['smooth_transitions'] else '✗'}")

        return checks

    def _spatial_analysis(self, tensor: torch.Tensor) -> Dict:
        """Анализ пространственного распределения"""
        metrics = {}

        if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
            data = tensor[0, 0]  # первый batch, первый канал
        else:
            data = tensor[0] if len(tensor.shape) >= 4 else tensor

        # Анализ центральных и периферийных областей
        d, h, w = data.shape
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Центральная область (25% от каждого измерения)
        d_margin, h_margin, w_margin = d // 8, h // 8, w // 8
        center_region = data[
            center_d - d_margin:center_d + d_margin,
            center_h - h_margin:center_h + h_margin,
            center_w - w_margin:center_w + w_margin
        ]

        metrics['center_mean'] = torch.mean(center_region).item()
        metrics['center_std'] = torch.std(center_region).item()

        # Периферийная область (углы)
        corner_size = min(d // 4, h // 4, w // 4)
        corners = [
            data[:corner_size, :corner_size, :corner_size],
            data[:corner_size, :corner_size, -corner_size:],
            data[:corner_size, -corner_size:, :corner_size],
            data[:corner_size, -corner_size:, -corner_size:],
        ]

        corner_means = [torch.mean(corner).item() for corner in corners]
        metrics['corner_mean'] = np.mean(corner_means)
        metrics['corner_std'] = np.std(corner_means)

        # Контрастность (разница между центром и углами)
        metrics['center_corner_contrast'] = abs(metrics['center_mean'] - metrics['corner_mean'])

        # Проверка на адекватность контраста для КТ
        metrics['adequate_contrast'] = metrics['center_corner_contrast'] > 0.05  # Для нормализованных данных

        logger.info("Пространственный анализ:")
        logger.info(f"Центр: {metrics['center_mean']:.4f}, Углы: {metrics['corner_mean']:.4f}")
        logger.info(f"Контраст: {metrics['center_corner_contrast']:.4f}")

        return metrics

    def _model_readiness_check(self, tensor: torch.Tensor) -> Dict:
        """Проверка готовности для подачи в model"""
        checks = {}

        # Проверка совместимости с model
        checks['shape_correct'] = tensor.shape == self.expected_shape
        checks['batch_size_ok'] = tensor.shape[0] >= 1
        checks['spatial_dims_ok'] = len(tensor.shape) == 5

        # Проверка минимальных размеров (для model это 128x128x128)
        if len(tensor.shape) == 5:
            checks['min_size_ok'] = all(s >= 32 for s in tensor.shape[2:])  # Минимальный размер для 3D CNN
        else:
            checks['min_size_ok'] = all(s >= 32 for s in tensor.shape[1:])

        # Проверка нормализации (ImageNet нормализация)
        mean_abs = torch.mean(torch.abs(tensor)).item()
        checks['likely_normalized'] = mean_abs < 5  # Грубая оценка нормализации

        # Проверка на градиенты (для обучения)
        checks['requires_grad'] = tensor.requires_grad

        # Проверка памяти для модели (примерная оценка)
        model_memory_estimate = tensor.numel() * 4 * 5 / (1024 ** 3)  # Примерно 5x для forward pass
        checks['memory_feasible'] = model_memory_estimate < 4  # < 4GB для model

        logger.info("Готовность для model:")
        logger.info(f"Правильная форма: {'✓' if checks['shape_correct'] else '✗'}")
        logger.info(f"Размер batch: {'✓' if checks['batch_size_ok'] else '✗'}")
        logger.info(f"Пространственные размеры: {'✓' if checks['spatial_dims_ok'] else '✗'}")
        logger.info(f"Минимальный размер: {'✓' if checks['min_size_ok'] else '✗'}")
        logger.info(f"Вероятно нормализован: {'✓' if checks['likely_normalized'] else '✗'}")

        return checks

    def _calculate_overall_quality(self, metrics: Dict) -> Dict:
        """Расчет общей оценки качества"""
        score = 0
        max_score = 0
        issues = []

        # Базовые проверки (вес: 30%)
        basic = metrics['basic_checks']
        if basic['shape_correct']:
            score += 8
        else:
            issues.append("Неправильная форма тензора")

        if basic['dtype_float']:
            score += 3
        else:
            issues.append("Неподходящий тип данных")

        if not basic['has_nan']:
            score += 4
        else:
            issues.append("Содержит NaN значения")

        if not basic['has_inf']:
            score += 3
        else:
            issues.append("Содержит Inf значения")

        max_score += 18

        # Статистические проверки (вес: 25%)
        stats = metrics['statistical_metrics']
        if stats.get('proper_normalization', False):
            score += 6
        else:
            issues.append("Проблемы с нормализацией")

        if stats.get('mean_in_range', False):
            score += 4
        else:
            issues.append("Среднее значение вне ожидаемого диапазона")

        max_score += 10

        # Проверки артефактов (вес: 20%)
        artifacts = metrics['artifact_checks']
        if artifacts['constant_slices_ratio'] < 0.05:
            score += 3
        else:
            issues.append("Слишком много константных срезов")

        if artifacts['outliers_ratio'] < 0.005:
            score += 3
        else:
            issues.append("Слишком много выбросов")

        if artifacts['smooth_transitions']:
            score += 2
        else:
            issues.append("Резкие переходы между срезами")

        max_score += 8

        # Пространственный анализ (вес: 15%)
        spatial = metrics['spatial_analysis']
        if spatial['adequate_contrast']:
            score += 5
        else:
            issues.append("Недостаточный контраст")

        max_score += 5

        # Готовность модели (вес: 10%)
        model = metrics['model_readiness']
        if model['shape_correct'] and model['spatial_dims_ok']:
            score += 3
        else:
            issues.append("Проблемы с размерностями для модели")

        if model['memory_feasible']:
            score += 2
        else:
            issues.append("Возможные проблемы с памятью")

        max_score += 5

        # Расчет финального скора
        quality_score = (score / max_score) * 100

        if quality_score >= 90:
            quality_level = "ОТЛИЧНО"
        elif quality_score >= 75:
            quality_level = "ХОРОШО"
        elif quality_score >= 60:
            quality_level = "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            quality_level = "ПЛОХО"

        return {
            'score': quality_score,
            'level': quality_level,
            'issues': issues,
            'ready_for_training': quality_score >= 70
        }

    def _print_report(self, metrics: Dict):
        """Печать итогового отчета"""
        logger.info("=" * 50)
        logger.info("ИТОГОВЫЙ ОТЧЕТ О КАЧЕСТВЕ")
        logger.info("=" * 50)

        overall = metrics['overall_quality']
        logger.info(f"Общая оценка: {overall['score']:.1f}/100 ({overall['level']})")
        logger.info(f"Готов для обучения: {'✓ ДА' if overall['ready_for_training'] else '✗ НЕТ'}")

        if overall['issues']:
            logger.info(f"Выявленные проблемы:")
            for i, issue in enumerate(overall['issues'], 1):
                logger.info(f"{i}. {issue}")
        else:
            logger.info("✓ Проблем не обнаружено!")

        logger.info("=" * 50)

    def visualize_tensor(self, tensor: torch.Tensor,
                         slice_indices: Optional[List[int]] = None,
                         save_path: Optional[str] = None):
        """Визуализация срезов тензора для model"""

        # Извлекаем данные для визуализации
        if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
            data = tensor[0, 0].cpu().numpy()  # первый batch, первый канал
        elif len(tensor.shape) == 4:  # [batch, depth, height, width]
            data = tensor[0].cpu().numpy()
        else:
            data = tensor.cpu().numpy()

        if slice_indices is None:
            # Выбираем несколько срезов для визуализации
            depth = data.shape[0]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

        fig, axes = plt.subplots(1, len(slice_indices), figsize=(15, 5))
        if len(slice_indices) == 1:
            axes = [axes]

        for i, slice_idx in enumerate(slice_indices):
            im = axes[i].imshow(data[slice_idx], cmap='gray')
            axes[i].set_title(f'Срез {slice_idx}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def batch_assessment(self, tensors: List[torch.Tensor],
                         tensor_names: Optional[List[str]] = None,
                         save_report: bool = True,
                         report_path: str = "quality_assessment_report.json") -> List[Dict]:
        """Оценка качества нескольких тензоров"""
        if tensor_names is None:
            tensor_names = [f"tensor_{i}" for i in range(len(tensors))]

        results = []

        with tqdm(total=len(tensors), desc="Оценка качества тензоров") as pbar:
            for tensor, name in zip(tensors, tensor_names):
                try:
                    result = self.assess_tensor_quality(tensor, name)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка оценки тензора {name}: {e}")
                finally:
                    pbar.update(1)

        # Сводная статистика
        if results:
            scores = [r['overall_quality']['score'] for r in results]
            ready_count = sum(r['overall_quality']['ready_for_training'] for r in results)

            logger.info(f"\n{'=' * 50}")
            logger.info("СВОДНАЯ СТАТИСТИКА ПО БАТЧУ")
            logger.info(f"{'=' * 50}")
            logger.info(f"Количество тензоров: {len(results)}")
            logger.info(f"Средняя оценка: {np.mean(scores):.1f}")
            logger.info(f"Готовых для обучения: {ready_count}/{len(results)}")

            # Сохраняем отчет если требуется
            if save_report:
                self._save_batch_report(results, report_path)

        return results

    def _save_batch_report(self, results: List[Dict], report_path: str):
        """Сохранение отчета в JSON файл"""
        # Конвертируем тензоры в сериализуемые типы
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Отчет сохранен в: {report_path}")

    def get_summary_statistics(self) -> Dict:
        """Получение сводной статистики по всем оцененным тензорам"""
        if not self.quality_reports:
            return {}

        scores = [report['overall_quality']['score'] for report in self.quality_reports]
        ready_count = sum(1 for report in self.quality_reports
                          if report['overall_quality']['ready_for_training'])

        stats = {
            'total_tensors': len(self.quality_reports),
            'ready_for_training': ready_count,
            'not_ready': len(self.quality_reports) - ready_count,
            'average_score': float(np.mean(scores)) if scores else 0,
            'median_score': float(np.median(scores)) if scores else 0,
            'min_score': float(np.min(scores)) if scores else 0,
            'max_score': float(np.max(scores)) if scores else 0,
            'std_score': float(np.std(scores)) if scores else 0
        }

        return stats


def main():
    """Основная функция для командной строки"""
    import argparse

    parser = argparse.ArgumentParser(description="Оценка качества тензоров для model")
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к файлу или директории с тензорами")
    parser.add_argument("--expected-shape", type=int, nargs=5,
                        default=[1, 1, 128, 128, 128],
                        help="Ожидаемая форма тензора (default: 1 1 128 128 128)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Максимальное количество файлов для обработки")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Количество параллельных процессов (default: 4)")
    parser.add_argument("--save-report", action="store_true",
                        help="Сохранить отчет в JSON файл")
    parser.add_argument("--report-path", type=str, default="quality_report.json",
                        help="Путь к файлу отчета (default: quality_report.json)")
    parser.add_argument("--visualize", action="store_true",
                        help="Показать визуализацию тензоров")

    args = parser.parse_args()

    # Создаем ассессор
    assessor = CTTensorQualityAssessment(expected_shape=tuple(args.expected_shape))
    input_path = Path(args.input)

    if input_path.is_file():
        # Обработка одного файла
        try:
            tensor = assessor.load_tensor_from_file(str(input_path))
            result = assessor.assess_tensor_quality(tensor, input_path.stem)

            if args.visualize:
                assessor.visualize_tensor(tensor)

        except Exception as e:
            logger.error(f"Ошибка обработки файла {input_path}: {e}")

    elif input_path.is_dir():
        # Обработка директории
        try:
            tensors_and_names = assessor.load_tensors_from_directory(
                str(input_path),
                max_files=args.max_files,
                num_workers=args.num_workers
            )

            if tensors_and_names:
                tensors = [t for t, n in tensors_and_names]
                names = [n for t, n in tensors_and_names]

                results = assessor.batch_assessment(
                    tensors, names,
                    save_report=args.save_report,
                    report_path=args.report_path
                )

                # Показываем сводную статистику
                summary = assessor.get_summary_statistics()
                logger.info("Сводная статистика:")
                for key, value in summary.items():
                    logger.info(f"{key}: {value}")

                if args.visualize and tensors:
                    # Визуализируем первый тензор
                    assessor.visualize_tensor(tensors[0])

            else:
                logger.warning("Тензоры не найдены в указанной директории")

        except Exception as e:
            logger.error(f"Ошибка обработки директории {input_path}: {e}")

    else:
        logger.error(f"Указанный путь не существует: {input_path}")


if __name__ == "__main__":
    main()