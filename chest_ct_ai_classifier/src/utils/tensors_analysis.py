import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import os
import glob
from pathlib import Path

warnings.filterwarnings('ignore')


class CTTensorQualityAssessment:
    """
    Класс для оценки качества нормализованных тензоров КТ снимков грудной клетки
    перед подачей в 3D EfficientNet модель
    """

    def __init__(self, expected_shape: Tuple[int, int, int, int, int] = (1, 3, 128, 160, 160)):
        """
        Args:
            expected_shape: Ожидаемая форма тензора (batch, channels, depth, height, width)
        """
        self.expected_shape = expected_shape
        self.quality_metrics = {}
        self.supported_formats = ['.pt', '.pth', '.npy', '.npz']

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
                                    max_files: Optional[int] = None) -> List[Tuple[torch.Tensor, str]]:
        """
        Загрузка всех тензоров из директории

        Args:
            directory_path: Путь к директории
            pattern: Паттерн для поиска файлов (например, "*.pt" или "scan_*")
            max_files: Максимальное количество файлов для загрузки

        Returns:
            Список кортежей (тензор, имя_файла) в исходных типах данных
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

        print(f"Найдено {len(all_files)} файлов для загрузки")

        tensors_and_names = []

        for file_path in all_files:
            try:
                tensor = self.load_tensor_from_file(file_path)
                tensors_and_names.append((tensor, file_path.stem))
            except Exception as e:
                print(f"Ошибка загрузки {file_path}: {e}")
                continue

        print(f"Успешно загружено {len(tensors_and_names)} тензоров")
        return tensors_and_names

    def _resize_tensor(self, tensor: torch.Tensor,
                       target_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Изменение размера тензора до целевой формы

        Args:
            tensor: Исходный тензор
            target_shape: Целевая форма (batch, channels, depth, height, width)

        Returns:
            Тензор измененного размера
        """
        import torch.nn.functional as F

        current_shape = tensor.shape
        target_batch, target_channels, target_depth, target_height, target_width = target_shape

        print(f"Изменение размера с {current_shape} на {target_shape}")

        # Если форма уже совпадает, возвращаем как есть
        if current_shape == target_shape:
            return tensor

        # Обрабатываем случай с 4D тензором (без каналов)
        if len(current_shape) == 4:
            # Добавляем размерность каналов
            tensor = tensor.unsqueeze(1)  # (batch, 1, depth, height, width)
            current_shape = tensor.shape

        # Интерполяция пространственных измерений
        if current_shape[2:] != target_shape[2:]:
            # Используем trilinear интерполяцию для 3D данных
            # Сначала меняем порядок размерностей для F.interpolate
            tensor = tensor.permute(0, 1, 3, 4, 2)  # (batch, channels, height, width, depth)

            tensor = F.interpolate(
                tensor,
                size=(target_height, target_width, target_depth),
                mode='trilinear',
                align_corners=False
            )

            # Возвращаем правильный порядок
            tensor = tensor.permute(0, 1, 4, 2, 3)  # (batch, channels, depth, height, width)

        # Обрезаем или дополняем batch dimension
        if current_shape[0] != target_batch:
            if current_shape[0] > target_batch:
                tensor = tensor[:target_batch]
            else:
                # Дублируем последний элемент
                padding_needed = target_batch - current_shape[0]
                last_tensor = tensor[-1:].repeat(padding_needed, 1, 1, 1, 1)
                tensor = torch.cat([tensor, last_tensor], dim=0)

        # Обрезаем или дополняем channels dimension
        if current_shape[1] != target_channels:
            if current_shape[1] > target_channels:
                tensor = tensor[:, :target_channels]
            else:
                # Дублируем последний канал
                padding_needed = target_channels - current_shape[1]
                last_channel = tensor[:, -1:].repeat(1, padding_needed, 1, 1, 1)
                tensor = torch.cat([tensor, last_channel], dim=1)

        return tensor

    def assess_tensor_quality(self, tensor: torch.Tensor,
                              tensor_name: str = "CT_tensor") -> Dict:
        """
        Полная оценка качества нормализованного тензора

        Args:
            tensor: Входной нормализованный тензор КТ снимка
            tensor_name: Название тензора для отчета

        Returns:
            Словарь с метриками качества
        """
        print(f"=== Оценка качества нормализованного тензора: {tensor_name} ===\n")

        # Базовые проверки
        basic_checks = self._basic_checks(tensor)

        # Статистический анализ для всех каналов
        statistical_metrics = self._statistical_analysis_multi_channel(tensor)

        # Проверка на артефакты и аномалии для всех каналов
        artifact_checks = self._artifact_detection_multi_channel(tensor)

        # Анализ пространственного распределения для всех каналов
        spatial_analysis = self._spatial_analysis_multi_channel(tensor)

        # Сравнение между каналами
        channel_comparison = self._channel_comparison_analysis(tensor)

        # Проверка готовности для модели
        model_readiness = self._model_readiness_check(tensor)

        # Объединение всех метрик
        all_metrics = {
            'tensor_name': tensor_name,
            'basic_checks': basic_checks,
            'statistical_metrics': statistical_metrics,
            'artifact_checks': artifact_checks,
            'spatial_analysis': spatial_analysis,
            'channel_comparison': channel_comparison,
            'model_readiness': model_readiness
        }

        # Общая оценка качества
        overall_quality = self._calculate_overall_quality(all_metrics)
        all_metrics['overall_quality'] = overall_quality

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

        print(f"Форма тензора: {checks['shape']} {'✓' if checks['shape_correct'] else '✗'}")
        print(f"Тип данных: {checks['dtype']} {'✓' if checks['dtype_float'] else '✗'}")
        print(f"Устройство: {checks['device']}")
        print(f"Размер в памяти: {checks['memory_mb']:.2f} МБ")
        print(f"NaN значения: {'Есть ✗' if checks['has_nan'] else 'Нет ✓'}")
        print(f"Inf значения: {'Есть ✗' if checks['has_inf'] else 'Нет ✓'}\n")

        return checks

    def _statistical_analysis_multi_channel(self, tensor: torch.Tensor) -> Dict:
        """
        Статистический анализ нормализованного тензора для всех каналов
        """
        stats = {}

        # Приводим тензор к float32 для всех вычислений
        if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
            tensor_float = tensor.float()
            print(f"Тензор приведен к float32 для статистических вычислений (исходный тип: {tensor.dtype})")
        else:
            tensor_float = tensor

        # Анализ для каждого канала
        channel_stats = {}
        num_channels = tensor_float.shape[1] if len(tensor_float.shape) == 5 else 1

        for channel_idx in range(num_channels):
            if len(tensor_float.shape) == 5:  # [batch, channels, depth, height, width]
                channel_data = tensor_float[0, channel_idx]  # первый batch, текущий канал
            else:
                channel_data = tensor_float[0] if len(tensor_float.shape) >= 4 else tensor_float

            channel_stats[f'channel_{channel_idx}'] = self._analyze_single_channel(channel_data, channel_idx)

        stats['channels'] = channel_stats

        # Общая статистика по всем каналам
        all_channel_means = [channel_stats[f'channel_{i}']['mean'] for i in range(num_channels)]
        all_channel_stds = [channel_stats[f'channel_{i}']['std'] for i in range(num_channels)]

        stats['mean_across_channels'] = float(np.mean(all_channel_means))
        stats['std_across_channels'] = float(np.mean(all_channel_stds))
        stats['channel_variance'] = float(np.var(all_channel_means)) if len(all_channel_means) > 1 else 0.0

        print("Статистический анализ по каналам:")
        for i in range(num_channels):
            ch_stats = channel_stats[f'channel_{i}']
            print(f"Канал {i}: Mean={ch_stats['mean']:.4f}, Std={ch_stats['std']:.4f}, "
                  f"Min={ch_stats['min']:.4f}, Max={ch_stats['max']:.4f}")
        print()

        return stats

    def _analyze_single_channel(self, data: torch.Tensor, channel_idx: int) -> Dict:
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
        except Exception as e:
            stats['q05'] = stats['q25'] = stats['q75'] = stats['q95'] = 0.0

        # Дополнительные метрики
        stats['range'] = stats['max'] - stats['min']
        stats['iqr'] = stats['q75'] - stats['q25']  # Межквартильный размах

        # Критерии для нормализованных данных [0,1]
        stats['range_appropriate'] = (0 <= stats['min'] <= 0.1) and (0.9 <= stats['max'] <= 1.0)
        stats['has_negative'] = stats['min'] < -0.01  # Небольшая погрешность
        stats['proper_normalization'] = (stats['min'] >= -0.01) and (stats['max'] <= 1.01)

        # Для нормализованных данных ожидаем среднее около 0.3-0.7
        stats['mean_in_range'] = 0.1 <= stats['mean'] <= 0.9

        # Коэффициент вариации для нормализованных данных
        stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] > 0.01 else float('inf')

        # Дополнительная диагностика для нормализованных данных
        self._analyze_normalized_distribution_single(data, stats)

        return stats

    def _analyze_normalized_distribution_single(self, data: torch.Tensor, metrics: Dict):
        """Дополнительный анализ для нормализованных данных одного канала"""

        # Анализ использования динамического диапазона
        used_range = metrics['max'] - metrics['min']
        metrics['range_utilization'] = used_range

        # Проверка на "обрезание" значений
        edge_threshold = 0.01  # 1% пикселей
        near_zero = (data <= 0.05).float().mean().item()
        near_one = (data >= 0.95).float().mean().item()

        metrics['clipping_at_zero'] = near_zero > edge_threshold
        metrics['clipping_at_one'] = near_one > edge_threshold

        # Анализ распределения по тканевым диапазонам (примерные значения после нормализации)
        air_region = (data <= 0.1).float().mean().item()  # Воздух (обычно темные области)
        soft_tissue = ((data > 0.2) & (data < 0.8)).float().mean().item()  # Мягкие ткани
        bright_regions = (data >= 0.8).float().mean().item()  # Яркие области (кость/контраст)

        metrics['air_ratio'] = air_region
        metrics['soft_tissue_ratio'] = soft_tissue
        metrics['bright_ratio'] = bright_regions

    def _artifact_detection_multi_channel(self, tensor: torch.Tensor) -> Dict:
        """Обнаружение артефактов и аномалий для всех каналов"""
        checks = {}

        num_channels = tensor.shape[1] if len(tensor.shape) == 5 else 1
        channel_checks = {}

        for channel_idx in range(num_channels):
            if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
                data = tensor[0, channel_idx]  # первый batch, текущий канал
            else:
                data = tensor[0] if len(tensor.shape) >= 4 else tensor

            channel_checks[f'channel_{channel_idx}'] = self._artifact_detection_single_channel(data)

        checks['channels'] = channel_checks

        # Общая статистика
        all_constant_slices = [channel_checks[f'channel_{i}']['constant_slices_ratio'] for i in range(num_channels)]
        checks['max_constant_slices_ratio'] = max(all_constant_slices) if all_constant_slices else 0
        checks['avg_constant_slices_ratio'] = np.mean(all_constant_slices) if all_constant_slices else 0

        print("Анализ артефактов по каналам:")
        for i in range(num_channels):
            ch_checks = channel_checks[f'channel_{i}']
            print(f"Канал {i}: Константные срезы={ch_checks['constant_slices_ratio']:.1%}, "
                  f"Выбросы={ch_checks['outliers_ratio']:.1%}, "
                  f"Плавные переходы={'✓' if ch_checks['smooth_transitions'] else '✗'}")
        print()

        return checks

    def _artifact_detection_single_channel(self, data: torch.Tensor) -> Dict:
        """Обнаружение артефактов для одного канала"""
        checks = {}

        # Проверка на константные срезы (могут указывать на проблемы)
        slice_stds = torch.std(data, dim=(1, 2))
        checks['constant_slices'] = (slice_stds < 1e-6).sum().item()
        checks['constant_slices_ratio'] = checks['constant_slices'] / data.shape[0]

        # Проверка на выбросы (значения > 3 sigma)
        mean_val = torch.mean(data)
        std_val = torch.std(data)
        outliers = torch.abs(data - mean_val) > 3 * std_val
        checks['outliers_count'] = outliers.sum().item()
        checks['outliers_ratio'] = checks['outliers_count'] / data.numel()

        # Проверка однородности по срезам
        slice_means = torch.mean(data, dim=(1, 2))
        checks['slice_mean_std'] = torch.std(slice_means).item()
        checks['uniform_slices'] = checks['slice_mean_std'] < std_val * 0.5

        # Проверка на резкие переходы между срезами
        if data.shape[0] > 1:
            slice_diffs = torch.diff(slice_means)
            checks['max_slice_diff'] = torch.max(torch.abs(slice_diffs)).item()
            checks['smooth_transitions'] = checks['max_slice_diff'] < 2 * checks['slice_mean_std']
        else:
            checks['max_slice_diff'] = 0
            checks['smooth_transitions'] = True

        return checks

    def _spatial_analysis_multi_channel(self, tensor: torch.Tensor) -> Dict:
        """Анализ пространственного распределения для всех каналов"""
        metrics = {}

        num_channels = tensor.shape[1] if len(tensor.shape) == 5 else 1
        channel_metrics = {}

        for channel_idx in range(num_channels):
            if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
                data = tensor[0, channel_idx]  # первый batch, текущий канал
            else:
                data = tensor[0] if len(tensor.shape) >= 4 else tensor

            channel_metrics[f'channel_{channel_idx}'] = self._spatial_analysis_single_channel(data)

        metrics['channels'] = channel_metrics

        # Общая статистика
        all_contrasts = [channel_metrics[f'channel_{i}']['center_corner_contrast'] for i in range(num_channels)]
        metrics['avg_contrast'] = np.mean(all_contrasts) if all_contrasts else 0
        metrics['max_contrast'] = max(all_contrasts) if all_contrasts else 0

        print("Пространственный анализ по каналам:")
        for i in range(num_channels):
            ch_metrics = channel_metrics[f'channel_{i}']
            print(f"Канал {i}: Центр={ch_metrics['center_mean']:.4f}, "
                  f"Углы={ch_metrics['corner_mean']:.4f}, "
                  f"Контраст={ch_metrics['center_corner_contrast']:.4f}")
        print()

        return metrics

    def _spatial_analysis_single_channel(self, data: torch.Tensor) -> Dict:
        """Анализ пространственного распределения для одного канала"""
        metrics = {}

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
        metrics['adequate_contrast'] = metrics['center_corner_contrast'] > 0.1  # Для нормализованных данных

        return metrics

    def _channel_comparison_analysis(self, tensor: torch.Tensor) -> Dict:
        """Сравнение характеристик между каналами"""
        comparison = {}

        if len(tensor.shape) != 5 or tensor.shape[1] < 2:
            comparison['channels_consistent'] = True
            comparison['channel_differences'] = {}
            comparison['max_channel_diff'] = 0
            return comparison

        num_channels = tensor.shape[1]
        channel_means = []
        channel_stds = []

        # Собираем статистику по каждому каналу
        for channel_idx in range(num_channels):
            data = tensor[0, channel_idx]
            channel_means.append(data.mean().item())
            channel_stds.append(data.std().item())

        # Анализ различий между каналами
        comparison['channel_means'] = channel_means
        comparison['channel_stds'] = channel_stds

        # Максимальная разница между средними значениями каналов
        comparison['max_mean_diff'] = max(channel_means) - min(channel_means) if channel_means else 0
        comparison['max_std_diff'] = max(channel_stds) - min(channel_stds) if channel_stds else 0

        # Проверка на согласованность каналов
        # Если разница между каналами слишком велика, это может указывать на проблемы
        #comparison['channels_consistent'] = comparison['max_mean_diff'] < 0.3  # Порог для нормализованных данных
        # Для разных окон несогласованность - это нормально
        comparison['channels_consistent'] = True  # Всегда считаем согласованными
        comparison['channels_different'] = comparison['max_mean_diff'] > 0.1  # Проверяем, что каналы разные

        # Детальный анализ пар каналов
        differences = {}
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                diff = abs(channel_means[i] - channel_means[j])
                differences[f'channels_{i}_{j}'] = diff

        comparison['channel_differences'] = differences
        comparison['max_channel_diff'] = max(differences.values()) if differences else 0

        print("Сравнение между каналами:")
        print(f"Средние значения каналов: {[f'{m:.4f}' for m in channel_means]}")
        print(f"Стандартные отклонения: {[f'{s:.4f}' for s in channel_stds]}")
        print(f"Максимальная разница средних: {comparison['max_mean_diff']:.4f}")
        print(f"Каналы согласованы: {'✓' if comparison['channels_consistent'] else '✗'}")
        print()

        return comparison

    def _model_readiness_check(self, tensor: torch.Tensor) -> Dict:
        """Проверка готовности для подачи в модель"""
        checks = {}

        # Проверка совместимости с EfficientNet-3D
        checks['batch_size_ok'] = tensor.shape[0] >= 1
        checks['spatial_dims_ok'] = len(tensor.shape) in [4, 5]

        # Проверка минимальных размеров
        if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
            checks['min_size_ok'] = all(s >= 32 for s in tensor.shape[2:])  # Минимальный размер для CNN
        else:  # [batch, depth, height, width]
            checks['min_size_ok'] = all(s >= 32 for s in tensor.shape[1:])

        # Проверка нормализации (приблизительная)
        mean_abs = torch.mean(torch.abs(tensor)).item()
        checks['likely_normalized'] = mean_abs < 10  # Грубая оценка нормализации

        # Проверка на градиенты (для обучения)
        checks['requires_grad'] = tensor.requires_grad

        # Проверка памяти для модели (примерная оценка)
        model_memory_estimate = tensor.numel() * 4 * 10 / (1024 ** 3)  # Примерно 10x для forward pass
        checks['memory_feasible'] = model_memory_estimate < 8  # < 8GB

        print("Готовность для модели:")
        print(f"Размер batch: {'✓' if checks['batch_size_ok'] else '✗'}")
        print(f"Пространственные размеры: {'✓' if checks['spatial_dims_ok'] else '✗'}")
        print(f"Минимальный размер: {'✓' if checks['min_size_ok'] else '✗'}")
        print(f"Вероятно нормализован: {'✓' if checks['likely_normalized'] else '✗'}")
        print(f"Требует градиенты: {'✓' if checks['requires_grad'] else 'Нет'}")
        print(f"Память приемлема: {'✓' if checks['memory_feasible'] else '✗'}")
        print(f"Примерная память модели: {model_memory_estimate:.1f} ГБ\n")

        return checks

    def _calculate_overall_quality(self, metrics: Dict) -> Dict:
        """Расчет общей оценки качества"""
        score = 0
        max_score = 0
        issues = []

        # Базовые проверки (вес: 25%)
        basic = metrics['basic_checks']
        if basic['shape_correct']:
            score += 5
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

        max_score += 15

        # Статистические проверки (вес: 20%)
        stats = metrics['statistical_metrics']
        # Проверяем статистику для всех каналов
        channel_stats = stats.get('channels', {})
        for channel_key, ch_stats in channel_stats.items():
            if ch_stats.get('proper_normalization', False):
                score += 2
            else:
                issues.append(f"Проблемы с нормализацией в {channel_key}")

            if ch_stats.get('mean_in_range', False):
                score += 1
            else:
                issues.append(f"Среднее значение вне ожидаемого диапазона в {channel_key}")

        max_score += 10

        # Проверки артефактов (вес: 20%)
        artifacts = metrics['artifact_checks']
        channel_artifacts = artifacts.get('channels', {})
        for channel_key, ch_artifacts in channel_artifacts.items():
            if ch_artifacts['constant_slices_ratio'] < 0.1:
                score += 1
            else:
                issues.append(f"Слишком много константных срезов в {channel_key}")

            if ch_artifacts['outliers_ratio'] < 0.01:
                score += 1
            else:
                issues.append(f"Слишком много выбросов в {channel_key}")

            if ch_artifacts['smooth_transitions']:
                score += 1
            else:
                issues.append(f"Резкие переходы между срезами в {channel_key}")

        max_score += 10

        # Пространственный анализ (вес: 15%)
        spatial = metrics['spatial_analysis']
        channel_spatial = spatial.get('channels', {})
        for channel_key, ch_spatial in channel_spatial.items():
            if ch_spatial['adequate_contrast']:
                score += 2
            else:
                issues.append(f"Недостаточный контраст в {channel_key}")

        max_score += 10

        # Сравнение между каналами (вес: 10%)
        channel_comparison = metrics.get('channel_comparison', {})
        if channel_comparison.get('channels_different', False):
            score += 5  # Разные каналы - это хорошо!
            print("✓ Каналы демонстрируют разные представления данных")
        else:
            issues.append("Каналы слишком похожи - отсутствует разнообразие представлений")

        max_score += 5

        # Готовность модели (вес: 10%)
        model = metrics['model_readiness']
        if model['spatial_dims_ok'] and model['min_size_ok']:
            score += 2
        else:
            issues.append("Проблемы с размерностями для модели")

        if model['memory_feasible']:
            score += 1
        else:
            issues.append("Возможные проблемы с памятью")

        max_score += 3

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
        print("=" * 50)
        print("ИТОГОВЫЙ ОТЧЕТ О КАЧЕСТВЕ")
        print("=" * 50)


        overall = metrics['overall_quality']
        print(f"Общая оценка: {overall['score']:.1f}/100 ({overall['level']})")
        print(f"Готов для обучения: {'✓ ДА' if overall['ready_for_training'] else '✗ НЕТ'}")

        if overall['issues']:
            print(f"\nВыявленные проблемы:")
            for i, issue in enumerate(overall['issues'], 1):
                print(f"{i}. {issue}")
        else:
            print(f"\n✓ Проблем не обнаружено!")

        print("=" * 50)

    def visualize_tensor(self, tensor: torch.Tensor,
                         slice_indices: Optional[List[int]] = None,
                         channel_index: int = 0,
                         save_path: Optional[str] = None):
        """Визуализация срезов тензора"""

        # Извлекаем данные для визуализации
        if len(tensor.shape) == 5:  # [batch, channels, depth, height, width]
            data = tensor[0, channel_index].cpu().numpy()  # первый batch, указанный канал
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

    def visualize_all_channels(self, tensor: torch.Tensor,
                               slice_indices: Optional[List[int]] = None,
                               save_path: Optional[str] = None):
        """Визуализация всех каналов тензора"""

        if len(tensor.shape) != 5:
            print("Для визуализации всех каналов тензор должен иметь 5D форму")
            return

        num_channels = tensor.shape[1]

        if slice_indices is None:
            depth = tensor.shape[2]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

        fig, axes = plt.subplots(num_channels, len(slice_indices),
                                 figsize=(5 * len(slice_indices), 4 * num_channels))

        if len(slice_indices) == 1 and num_channels == 1:
            axes = [[axes]]
        elif len(slice_indices) == 1:
            axes = [[ax] for ax in axes]
        elif num_channels == 1:
            axes = [axes]

        for channel_idx in range(num_channels):
            for i, slice_idx in enumerate(slice_indices):
                data = tensor[0, channel_idx, slice_idx].cpu().numpy()
                im = axes[channel_idx][i].imshow(data, cmap='gray')
                axes[channel_idx][i].set_title(f'Канал {channel_idx}, Срез {slice_idx}')
                axes[channel_idx][i].axis('off')
                plt.colorbar(im, ax=axes[channel_idx][i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def batch_assessment(self, tensors: List[torch.Tensor],
                         tensor_names: Optional[List[str]] = None) -> List[Dict]:
        """Оценка качества нескольких тензоров"""
        if tensor_names is None:
            tensor_names = [f"tensor_{i}" for i in range(len(tensors))]

        results = []
        for tensor, name in zip(tensors, tensor_names):
            result = self.assess_tensor_quality(tensor, name)
            results.append(result)

        # Сводная статистика
        scores = [r['overall_quality']['score'] for r in results]
        print(f"\n{'=' * 50}")
        print("СВОДНАЯ СТАТИСТИКА ПО БАТЧУ")
        print(f"{'=' * 50}")
        print(f"Количество тензоров: {len(results)}")
        print(f"Средняя оценка: {np.mean(scores):.1f}")
        print(
            f"Готовых для обучения: {sum(r['overall_quality']['ready_for_training'] for r in results)}/{len(results)}")

        return results


# Примеры использования с загрузкой файлов
def example_usage_with_files():
    """Примеры использования с загрузкой тензоров из файлов"""

    assessor = CTTensorQualityAssessment(expected_shape=(1, 3, 128, 160, 160))

    # Пример 1: Загрузка одного тензора из файла
    print("=== Пример 1: Загрузка одного файла ===")
    try:
        # Замените на реальный путь к вашему файлу
        tensor_path = "../data/data_tensors/studies/01-0.pt"

        tensor = assessor.load_tensor_from_file(tensor_path)

        quality_report = assessor.assess_tensor_quality(tensor, "loaded_tensor")

        # Визуализация всех каналов
        assessor.visualize_all_channels(tensor)

    except FileNotFoundError:
        print("Файл не найден. Укажите правильный путь в tensor_path")

    # Пример 2: Загрузка всех тензоров из директории
    print("\n=== Пример 2: Загрузка из директории ===")
    try:
        # Замените на реальный путь к директории с тензорами
        directory_path = "../data/data_tensors/studies/"

        tensors_and_names = assessor.load_tensors_from_directory(
            directory_path,
            pattern="*",  # Все файлы
            max_files=5  # Ограничиваем количество для примера
        )

        # Пакетная оценка
        tensors = [tensor for tensor, name in tensors_and_names]
        names = [name for tensor, name in tensors_and_names]

        results = assessor.batch_assessment(tensors, names)

    except FileNotFoundError:
        print("Директория не найдена. Укажите правильный путь в directory_path")


def quick_assessment_from_path(file_or_directory_path: str):
    """
    Быстрая оценка тензора(ов) по пути к файлу или директории

    Args:
        file_or_directory_path: Путь к файлу с тензором или директории с тензорами
    """
    assessor = CTTensorQualityAssessment(expected_shape=(1, 3, 128, 160, 160))
    path = Path(file_or_directory_path)

    if path.is_file():
        # Загружаем один файл
        try:
            tensor = assessor.load_tensor_from_file(str(path))
            assessor.assess_tensor_quality(tensor, path.stem)
            # Визуализация всех каналов для лучшего понимания
            assessor.visualize_all_channels(tensor)

        except Exception as e:
            print(f"Ошибка обработки файла {path}: {e}")

    elif path.is_dir():
        # Загружаем все файлы из директории
        try:
            tensors_and_names = assessor.load_tensors_from_directory(str(path))

            if tensors_and_names:
                tensors = [t for t, n in tensors_and_names]
                names = [n for t, n in tensors_and_names]
                assessor.batch_assessment(tensors, names)
            else:
                print("Тензоры не найдены в указанной директории")

        except Exception as e:
            print(f"Ошибка обработки директории {path}: {e}")

    else:
        print(f"Указанный путь не существует или недоступен: {file_or_directory_path}")


if __name__ == "__main__":
    # Пример использования
    # example_usage_with_files()
    pass