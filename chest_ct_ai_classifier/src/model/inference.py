import time

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path
from omegaconf import OmegaConf
import warnings
import os
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Заменяем относительные импорты
from .model_generator import generate_model
from .lightning_module import MedicalClassificationModel


# Добавляем Captum
try:
    from captum.attr import IntegratedGradients, Saliency, LayerGradCam, LayerAttribution
    from captum.attr import visualization as viz

    # Устанавливаем non-GUI backend для matplotlib (для macOS совместимости)
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️ Captum не установлен. Установите: pip install captum")

warnings.filterwarnings("ignore", category=UserWarning)


class MedicalModelInference:
    """
    Класс для инференса обученной медицинской классификационной модели.

    Принимает 3D тензор размером 256x256x256 и возвращает предсказание.
    """

    def __init__(self,
                 weights_path: str,
                 model_config: 'ModelConfig',  # Изменили тип аннотации
                 device: Optional[str] = None,
                 use_half_precision: bool = False):
        """
        Инициализация inference модуля.

        Args:
            weights_path: путь к чекпоинту модели
            model_config: конфигурация модели (ModelConfig)
            device: устройство ('cuda' или 'cpu'), если None - автоматически
            use_half_precision: использовать ли half precision (float16) для ускорения
        """
        self.weights_path = Path(weights_path)

        # Преобразуем ModelConfig в OmegaConf
        if isinstance(model_config, dict):
            self.model_config = OmegaConf.create(model_config)
        elif hasattr(model_config, '__dict__'):  # Это ModelConfig (dataclass)
            # Преобразуем dataclass в словарь, затем в OmegaConf
            config_dict = {}
            for field_name in dir(model_config):
                if not field_name.startswith('_'):
                    try:
                        value = getattr(model_config, field_name)
                        if not callable(value):
                            config_dict[field_name] = value
                    except:
                        continue
            self.model_config = OmegaConf.create(config_dict)
        else:
            self.model_config = model_config if isinstance(model_config, OmegaConf) else OmegaConf.create(model_config)

        self.use_half_precision = use_half_precision

        # Определение устройства
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Создание модели
        self.model = self._load_model()
        self.model.eval()

        # Установка half precision если нужно
        if self.use_half_precision and self.device == 'cuda':
            self.model.half()
            self._half_precision = True
        else:
            self._half_precision = False

        print(f"✅ Inference модуль инициализирован на устройстве: {self.device}")
        print(f"📊 Ожидаемый входной размер: 1, 1, 256, 256, 256")
        print(f"🎯 Количество классов: {self.model_config.n_seg_classes}")

    def _load_model(self):
        """Загрузка и инициализация модели с устойчивой подгонкой ключей state_dict."""
        from argparse import Namespace
        from collections import OrderedDict

        print("=" * 60)
        print("🔧 Создание модели для инференса...")

        # Преобразуем конфигурацию в namespace
        config_dict = OmegaConf.to_container(self.model_config)
        config_dict['gpu_id'] = [0] if torch.cuda.is_available() else []
        config_dict['phase'] = 'test'  # ВАЖНО: режим теста
        config_dict['no_cuda'] = (self.device == 'cpu')

        args = Namespace(**config_dict)

        # Генерируем пустую модель целевой архитектуры
        model, _ = generate_model(args)
        print("=" * 60)

        # Загружаем чекпоинт (CPU-совместимо)
        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=('cpu' if self.device == 'cpu' else self.device),
                weights_only=False
            )
            print(f"📁 Загружен checkpoint. Доступные ключи: {list(checkpoint.keys())}")
        except Exception as e:
            print(f"❌ Ошибка загрузки checkpoint: {e}")
            raise

        # Извлечение state_dict из разных возможных форматов
        state_dict = None

        # Попытка 1: PyTorch Lightning (.ckpt)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("🔧 Обнаружен формат PyTorch Lightning (.ckpt)")

        # Попытка 2: Стандартный PyTorch с обёрткой
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("🔧 Обнаружен формат model_state_dict (.pth)")

        # Попытка 3: Прямой state_dict (.pth) - чистые веса
        elif isinstance(checkpoint, dict):
            # Проверяем, есть ли тензоры напрямую в checkpoint
            tensor_keys = [k for k, v in checkpoint.items() if isinstance(v, torch.Tensor)]

            if len(tensor_keys) > 0:
                # Это прямой state_dict
                state_dict = checkpoint
                print("🔧 Обнаружен прямой state_dict (.pth)")
            else:
                # Возможно, это словарь с метаданными
                # Ищем любой ключ, который может содержать веса
                possible_keys = ['model', 'net', 'network', 'weights', 'parameters']
                for key in possible_keys:
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        state_dict = checkpoint[key]
                        print(f"🔧 Обнаружен state_dict в ключе '{key}'")
                        break

        # Попытка 4: Если это вообще не словарь (редкий случай)
        else:
            print("⚠️ Checkpoint не является словарем, пробуем использовать напрямую")
            state_dict = checkpoint

        # Финальная проверка
        if state_dict is None or (isinstance(state_dict, dict) and len(state_dict) == 0):
            print("❌ Не удалось найти state_dict в checkpoint")
            print(
                f"📋 Доступные ключи в checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
            raise ValueError("Неизвестный формат checkpoint. Убедитесь, что файл содержит веса модели.")

        # Получаем ключи целевой модели для сравнения
        target_keys = set(model.state_dict().keys())
        sample_target_key = list(target_keys)[0] if target_keys else ""
        sample_source_key = list(state_dict.keys())[0] if state_dict else ""

        print(f"🔍 Пример ключа в модели: {sample_target_key}")
        print(f"🔍 Пример ключа в checkpoint: {sample_source_key}")

        # Определяем, нужно ли добавлять или удалять префикс
        needs_module_prefix = any(k.startswith('module.') for k in target_keys)
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

        print(f"🔧 Модель требует 'module.' префикс: {needs_module_prefix}")
        print(f"🔧 Checkpoint имеет 'module.' префикс: {has_module_prefix}")

        # Обработка префиксов
        new_state_dict = OrderedDict()
        skipped_keys = []

        for k, v in state_dict.items():
            # Пропускаем служебные параметры (расширенный список)
            skip_patterns = [
                'loss_fn', 'criterion', 'optimizer', 'scheduler',
                'class_weights', 'loss_weights', 'weight_decay',
                'learning_rate', 'momentum', 'best_score'
            ]

            if any(pattern in k for pattern in skip_patterns):
                skipped_keys.append(k)
                continue

            new_key = k

            # Убираем лишние префиксы
            for prefix in ['model.module.', 'model.', 'net.', 'encoder.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break

            # Специальная обработка для 'module.'
            if new_key.startswith('module.'):
                new_key = new_key[7:]  # убираем 'module.'

            # Добавляем 'module.' если модель требует, а его нет
            if needs_module_prefix and not new_key.startswith('module.'):
                new_key = 'module.' + new_key
            # Убираем 'module.' если модель не требует, а он есть
            elif not needs_module_prefix and new_key.startswith('module.'):
                new_key = new_key[7:]

            new_state_dict[new_key] = v

        print(f"🔑 Обработано ключей: {len(new_state_dict)}")
        if skipped_keys:
            print(
                f"⏭️  Пропущено служебных ключей: {len(skipped_keys)} ({', '.join(skipped_keys[:3])}{'...' if len(skipped_keys) > 3 else ''})")
        if new_state_dict:
            print(f"🔍 Пример преобразованного ключа: {list(new_state_dict.keys())[0]}")

        # Загрузка state_dict в модель с обработкой ошибок
        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

            # Считаем успешно загруженные ключи
            loaded_keys = len(new_state_dict) - len(unexpected_keys)
            total_model_keys = len(model.state_dict())

            if missing_keys:
                print(
                    f"⚠️ Отсутствующие ключи ({len(missing_keys)}): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
            if unexpected_keys:
                print(
                    f"⚠️ Неожиданные ключи ({len(unexpected_keys)}): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")

            # Успешная загрузка
            if not missing_keys and not unexpected_keys:
                print("✅ Все ключи загружены успешно!")
            elif loaded_keys >= total_model_keys * 0.95:  # Загружено >= 95%
                print(
                    f"✅ Загружено {loaded_keys}/{total_model_keys} ключей ({loaded_keys / total_model_keys * 100:.1f}%) - достаточно для работы!")
            else:
                print(
                    f"🔄 Загружено {loaded_keys}/{total_model_keys} ключей ({loaded_keys / total_model_keys * 100:.1f}%)")

        except Exception as e:
            print(f"❌ Ошибка при strict загрузке: {e}")
            print("🔄 Пробуем загрузить только совпадающие ключи...")

            # Загружаем только совпадающие ключи по размеру
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in new_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(
                f"🔄 Загружено {len(filtered_dict)} из {len(new_state_dict)} ключей ({len(filtered_dict) / len(model_dict) * 100:.1f}%)")

        # Убираем DataParallel wrapper если он есть
        if hasattr(model, 'module'):
            print("🔧 Убираем DataParallel wrapper...")
            model = model.module

        # Перемещаем модель на нужное устройство
        model = model.to(self.device)
        model.eval()

        print("✅ Модель успешно загружена и готова к инференсу!")
        print("=" * 60)

        return model

    def _validate_input_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Валидация и предобработка входного тензора."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(tensor)}")

        if tensor.dim() != 5:
            raise ValueError(f"Ожидается 5D тензор (B, C, D, H, W), получено {tensor.shape}")

        batch_size, channels, depth, height, width = tensor.shape

        expected_shape = (1, 1, 256, 256, 256)
        if (channels, depth, height, width) != expected_shape[1:]:
            raise ValueError(
                f"Ожидается тензор формы {expected_shape}, "
                f"получено {tensor.shape}. "
                f"Ожидаемые размеры: channels=1, depth=256, height=256, width=256"
            )

        tensor = tensor.to(self.device)

        if self._half_precision:
            tensor = tensor.half()
        else:
            tensor = tensor.float()

        return tensor

    def _free_memory(self):
        """Очистка памяти GPU и сбор мусора."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Union[float, int, np.ndarray]]:
        """Выполняет предсказание на одном тензоре."""
        with torch.no_grad():
            input_tensor = self._validate_input_tensor(input_tensor)

            if self._half_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_tensor)
            else:
                logits = self.model(input_tensor)

            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()

            logits_np = logits.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

            # Очищаем память после предсказания
            self._free_memory()

            return {
                'prediction': int(predicted_class),
                'probabilities': probabilities_np.squeeze(),
                'confidence': float(confidence),
                'logits': logits_np.squeeze()
            }

    def explain_prediction(self, input_tensor: torch.Tensor, target_class: Optional[int] = 1,
                           method: str = "saliency", visualize: bool = True,
                           threshold: float = 0.1, alpha: float = 0.7, save_png: bool = True,
                           output_path: Optional[str] = None):
        """
        Объясняет предсказание с помощью Captum.

        Args:
            input_tensor: тензор формы (1, 1, 256, 256, 256)
            target_class: класс для объяснения (если None, используется предсказанный)
            method: метод объяснения ('integrated_gradients', 'saliency')
            visualize: сохранять ли визуализацию (по умолчанию True для macOS совместимости)
            threshold: порог для маскирования слабых атрибутов (0-1)
            alpha: прозрачность наложения тепловой карты
            save_png: сохранять ли PNG файл (по умолчанию True)
            output_path: путь для сохранения изображения (если None, используется автогенерация)

        Returns:
            dict: {'attributions': тензор атрибутов, 'image_path': путь к сохраненному изображению}
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum не установлен. Установите: pip install captum")

        try:
            # Валидация и подготовка входного тензора
            original_tensor = input_tensor.clone()
            input_tensor = self._validate_input_tensor(input_tensor)

            # Получаем предсказание
            if target_class is None:
                prediction_result = self.predict(original_tensor)
                target_class = prediction_result['prediction']

            print(f"🎯 Объяснение для класса {target_class} с методом {method}")

            # Подготавливаем тензор для атрибуции
            input_tensor = input_tensor.detach()
            input_tensor.requires_grad_(False)

            # Очищаем память перед вычислением атрибутов
            self._free_memory()

            # Выбираем метод с оптимизацией памяти
            if method == "integrated_gradients":
                explainer = IntegratedGradients(self.model)
                # Используем меньше шагов для экономии памяти
                with torch.no_grad():
                    attributions = explainer.attribute(input_tensor, target=target_class, n_steps=2)
            elif method == "saliency":
                explainer = Saliency(self.model)
                with torch.no_grad():
                    attributions = explainer.attribute(input_tensor, target=target_class, abs=False)
            else:
                raise ValueError(f"Неподдерживаемый метод: {method}")

            result = {'attributions': attributions}

            if visualize or save_png:
                saved_image_path = self._visualize_3d_attributions_enhanced(
                    attributions, original_tensor,
                    title=f"Attributions ({method})",
                    threshold=threshold,
                    alpha=alpha,
                    save_to_file=True,
                    output_path=output_path
                )
                result['image_path'] = saved_image_path
                print(f"✅ Визуализация сохранена: {saved_image_path}")

            return result

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ Недостаточно памяти GPU для объяснения. Попробуйте:")
                print("   - Использовать method='saliency'")
                print("   - Увеличить threshold (например, 0.3)")
                print("   - Обработать модель на CPU")
                self._free_memory()
                return None
            else:
                raise e

    def _visualize_3d_attributions_enhanced(self, attributions, original_tensor,
                                            title="Attributions", threshold=0.1, alpha=0.7, 
                                            save_to_file=False, output_path=None):
        """
        Улучшенная визуализация атрибутов с маскированием нулей и наложением.

        Args:
            attributions: тензор атрибутов
            original_tensor: исходный тензор
            title: заголовок визуализации
            threshold: порог для маскирования
            alpha: прозрачность наложения
            save_to_file: сохранять ли в файл (по умолчанию False для обратной совместимости)
            output_path: путь для сохранения (если None, генерируется автоматически)

        Returns:
            str: путь к сохраненному файлу (если save_to_file=True), иначе None
        """
        attr_np = attributions.squeeze().detach().cpu().numpy()
        original_np = original_tensor.squeeze().detach().cpu().numpy()

        # Нормализуем исходное изображение для визуализации
        original_normalized = (original_np - original_np.min()) / (original_np.max() - original_np.min())

        # Создаем маску для значимых атрибутов
        attr_abs = np.abs(attr_np)
        attr_max = np.max(attr_abs)

        if attr_max > 0:
            # Нормализуем и применяем порог
            attr_normalized = attr_abs / attr_max
            mask = attr_normalized > threshold
        else:
            mask = np.zeros_like(attr_abs, dtype=bool)

        mid_slice = attr_np.shape[0] // 2

        # Создаем кастомную colormap с прозрачностью для низких значений
        colors = plt.cm.hot(np.linspace(0, 1, 256))
        colors[0] = [0, 0, 0, 0]  # Полностью прозрачный для минимального значения
        transparent_hot = LinearSegmentedColormap.from_list('transparent_hot', colors)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Визуализация для трех плоскостей
        planes = [
            ('Axial (Z)', mid_slice, lambda x: x[mid_slice, :, :]),
            ('Coronal (Y)', mid_slice, lambda x: x[:, mid_slice, :]),
            ('Sagittal (X)', mid_slice, lambda x: x[:, :, mid_slice])
        ]

        for i, (plane_name, slice_idx, slice_fn) in enumerate(planes):
            # Исходное изображение
            orig_slice = slice_fn(original_normalized)
            axes[0, i].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"{plane_name} - Исходное")
            axes[0, i].axis('off')

            # Тепловая карта атрибутов с наложением
            attr_slice = slice_fn(attr_np)
            mask_slice = slice_fn(mask)

            # Применяем маску - оставляем только значимые атрибуты
            masked_attr = np.ma.masked_where(~mask_slice, attr_slice)

            im = axes[1, i].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
            im2 = axes[1, i].imshow(masked_attr, cmap=transparent_hot,
                                    alpha=alpha, vmin=attr_np.min(), vmax=attr_np.max())

            axes[1, i].set_title(f"{plane_name} - Атрибуты (порог: {threshold})")
            axes[1, i].axis('off')

            # Добавляем colorbar для атрибутов
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

        plt.suptitle(f"{title}\n(показаны только атрибуты > {threshold:.1%} от максимума)", fontsize=14)
        plt.tight_layout()

        # Дополнительная информация
        total_voxels = np.prod(attr_np.shape)
        significant_voxels = np.sum(mask)
        print(f"📊 Значимые воксели: {significant_voxels}/{total_voxels} ({significant_voxels / total_voxels:.1%})")
        print(f"📈 Максимальный атрибут: {attr_np.max():.4f}, Минимальный: {attr_np.min():.4f}")

        if save_to_file:
            if output_path is None:
                # Генерируем уникальное имя файла
                output_path = f"attributions_{int(time.time())}_{hash(str(time.time())) % 1000:03d}.png"

            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Закрываем figure для освобождения памяти
            return output_path
        else:
            # Для обратной совместимости - показываем только если явно не требуется сохранение
            # Внимание: на macOS это может вызвать ошибку!
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️ Не удалось показать изображение: {e}")
                print("💡 Используйте save_to_file=True для сохранения в файл")
            finally:
                plt.close(fig)
            return None

    def _visualize_3d_attributions(self, attributions, title="Attributions"):
        """Стандартная визуализация атрибутов по осям (для обратной совместимости)."""
        attr_np = attributions.squeeze().detach().cpu().numpy()

        mid_slice = attr_np.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Axial (Z)
        im1 = axes[0].imshow(attr_np[mid_slice, :, :], cmap='hot', vmin=attr_np.min(), vmax=attr_np.max())
        axes[0].set_title("Axial (Z)")
        plt.colorbar(im1, ax=axes[0])

        # Coronal (Y)
        im2 = axes[1].imshow(attr_np[:, mid_slice, :], cmap='hot', vmin=attr_np.min(), vmax=attr_np.max())
        axes[1].set_title("Coronal (Y)")
        plt.colorbar(im2, ax=axes[1])

        # Sagittal (X)
        im3 = axes[2].imshow(attr_np[:, :, mid_slice], cmap='hot', vmin=attr_np.min(), vmax=attr_np.max())
        axes[2].set_title("Sagittal (X)")
        plt.colorbar(im3, ax=axes[2])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def predict_with_explanation(self, input_tensor: torch.Tensor, method: str = "saliency",
                                 threshold: float = 0.1, alpha: float = 0.7, 
                                 save_png: bool = True, output_path: Optional[str] = None):
        """
        Выполняет предсказание и объяснение за раз.

        Args:
            input_tensor: входной тензор
            method: метод объяснения
            threshold: порог для маскирования
            alpha: прозрачность
            save_png: сохранять ли изображение
            output_path: путь для сохранения

        Returns:
            dict: {'prediction': ..., 'explanation': ...}
        """
        prediction = self.predict(input_tensor)
        explanation = self.explain_prediction(
            input_tensor,
            target_class=prediction['prediction'],
            method=method,
            threshold=threshold,
            alpha=alpha,
            save_png=save_png,
            output_path=output_path
        )
        return {
            'prediction': prediction,
            'explanation': explanation
        }

    # Остальные методы остаются без изменений
    def predict_batch(self, batch_tensor: torch.Tensor) -> List[Dict[str, Union[float, int, np.ndarray]]]:
        """Выполняет предсказание на батче тензоров."""
        with torch.no_grad():
            batch_size = batch_tensor.shape[0]
            batch_tensor = self._validate_input_tensor(batch_tensor)

            if self._half_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_tensor)
            else:
                logits = self.model(batch_tensor)

            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

            logits_np = logits.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

            results = []
            for i in range(batch_size):
                results.append({
                    'prediction': int(predicted_classes[i].item()),
                    'probabilities': probabilities_np[i],
                    'confidence': float(confidences[i].item()),
                    'logits': logits_np[i]
                })

            return results

    def predict_probability(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Возвращает только вероятности классов."""
        result = self.predict(input_tensor)
        return result['probabilities']

    def predict_class(self, input_tensor: torch.Tensor) -> int:
        """Возвращает только предсказанный класс."""
        result = self.predict(input_tensor)
        return result['prediction']


# Пример использования
if __name__ == "__main__":
    from config import ModelConfig

    # Пример использования
    config = ModelConfig()
    config.input_D = 256
    config.input_H = 256
    config.input_W = 256
    config.n_seg_classes = 2  # пример для бинарной классификации

    # Создаем inference модуль
    inference = MedicalModelInference(
        weights_path="model\\outputs\\weights\\best-epoch=42-val_f1=0.7650-val_auroc=0.8675.ckpt",
        model_config=config
    )

    # Тестовый тензор 256x256x256
    test_tensor = torch.randn(1, 1, 256, 256, 256)

    # Получаем предсказание
    result = inference.predict(test_tensor)

    print(f"Предсказанный класс: {result['prediction']}")
    print(f"Уверенность: {result['confidence']:.4f}")

    # Объяснение с улучшенной визуализацией
    try:
        explanation = inference.explain_prediction(
            test_tensor,
            method="saliency",
            visualize=True,
            threshold=0.2,  # Настройте порог по необходимости
            alpha=0.6  # Настройте прозрачность
        )

        # Или за раз:
        full_result = inference.predict_with_explanation(
            test_tensor,
            method="saliency",
            threshold=0.2,
            alpha=0.6
        )
        print(f"Полный результат: {full_result['prediction']['prediction']}")

    except Exception as e:
        print(f"❌ Ошибка при объяснении: {e}")
        print("🔄 Пробуем с более высоким порогом...")

        # Повторная попытка с более высоким порогом
        explanation = inference.explain_prediction(
            test_tensor,
            method="saliency",
            visualize=True,
            threshold=0.3,  # Более высокий порог
            alpha=0.5
        )