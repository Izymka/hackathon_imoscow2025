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
try:
    from .model_generator import generate_model
    from .lightning_module import MedicalClassificationModel
except ImportError:
    # Если относительный импорт не работает, используем абсолютный
    from model_generator import generate_model
    from lightning_module import MedicalClassificationModel


# Добавляем Captum
try:
    from captum.attr import IntegratedGradients, Saliency, LayerGradCam, LayerAttribution
    from captum.attr import visualization as viz
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
        config_dict = OmegaConf.to_container(self.model_config)
        config_dict['gpu_id'] = [0] if torch.cuda.is_available() else []
        config_dict['phase'] = 'test'
        config_dict['no_cuda'] = (self.device == 'cpu')

        from argparse import Namespace
        args = Namespace(**config_dict)

        # Генерируем пустую модель целевой архитектуры
        model, _ = generate_model(args)

        # Загружаем чекпоинт (CPU-совместимо)
        checkpoint = torch.load(self.weights_path, map_location=('cpu' if self.device == 'cpu' else self.device), weights_only=False)

        # Извлекаем исходный state_dict из разных возможных форматов
        raw_sd = None
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            raw_sd = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            # Если в словаре много тензоров — это, вероятно, сам state_dict
            tensor_like = [k for k, v in checkpoint.items() if isinstance(v, torch.Tensor)]
            if len(tensor_like) > 0:
                raw_sd = checkpoint
        # Если формат неизвестен, пробуем напрямую как state_dict
        if raw_sd is None:
            raw_sd = checkpoint if isinstance(checkpoint, dict) else {}

        target_sd = model.state_dict()

        # Нормализуем ключи: убираем префиксы 'model.module.', 'module.', 'model.'
        def normalize_key(k: str) -> str:
            for pref in ('model.module.', 'module.', 'model.'):
                if k.startswith(pref):
                    k = k[len(pref):]
            return k

        cleaned_sd = {}
        matched, total = 0, 0
        for k, v in raw_sd.items():
            total += 1
            nk = normalize_key(k)
            if nk in target_sd and target_sd[nk].shape == v.shape:
                cleaned_sd[nk] = v
                matched += 1

        # Загружаем максимально совместимую подмножину весов
        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        try:
            print(f"🔑 Загрузка весов: сопоставлено {matched}/{total}; пропущено {len(missing)}; лишних {len(unexpected)}")
        except Exception:
            pass

        # Убираем DataParallel, если есть
        if hasattr(model, 'module'):
            model = model.module

        # Перемещаем модель на нужное устройство
        model = model.to(self.device)
        model.eval()
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

    def explain_prediction(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                           method: str = "saliency", visualize: bool = True,
                           threshold: float = 0.1, alpha: float = 0.7):
        """
        Объясняет предсказание с помощью Captum.

        Args:
            input_tensor: тензор формы (1, 1, 256, 256, 256)
            target_class: класс для объяснения (если None, используется предсказанный)
            method: метод объяснения ('integrated_gradients', 'saliency')
            visualize: показывать ли визуализацию
            threshold: порог для маскирования слабых атрибутов (0-1)
            alpha: прозрачность наложения тепловой карты

        Returns:
            attributions: объяснение (тензор с тем же размером, что и вход)
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum не установлен. Установите: pip install captum")

        try:
            # Валидация и подготовка входного тензора
            original_tensor = input_tensor.clone()
            input_tensor = self._validate_input_tensor(input_tensor)

            # Получаем предсказание
            prediction_result = self.predict(original_tensor)
            if target_class is None:
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
                    attributions = explainer.attribute(input_tensor, target=target_class, n_steps=15)
            elif method == "saliency":
                explainer = Saliency(self.model)
                with torch.no_grad():
                    attributions = explainer.attribute(input_tensor, target=target_class, abs=False)
            else:
                raise ValueError(f"Неподдерживаемый метод: {method}")

            if visualize:
                self._visualize_3d_attributions_enhanced(
                    attributions, original_tensor,
                    title=f"Attributions ({method})",
                    threshold=threshold,
                    alpha=alpha
                )

            return attributions

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
                                            title="Attributions", threshold=0.1, alpha=0.7):
        """Улучшенная визуализация атрибутов с маскированием нулей и наложением."""
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
        plt.show()

        # Дополнительная информация
        total_voxels = np.prod(attr_np.shape)
        significant_voxels = np.sum(mask)
        print(f"📊 Значимые воксели: {significant_voxels}/{total_voxels} ({significant_voxels / total_voxels:.1%})")
        print(f"📈 Максимальный атрибут: {attr_np.max():.4f}, Минимальный: {attr_np.min():.4f}")

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
                                 threshold: float = 0.1, alpha: float = 0.7):
        """
        Выполняет предсказание и объяснение за раз.

        Returns:
            dict: {'prediction': ..., 'explanation': ...}
        """
        prediction = self.predict(input_tensor)
        explanation = self.explain_prediction(
            input_tensor,
            target_class=prediction['prediction'],
            method=method,
            threshold=threshold,
            alpha=alpha
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