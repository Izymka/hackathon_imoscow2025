# inference.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path
from omegaconf import OmegaConf
from .model_generator import generate_model
from .lightning_module import MedicalClassificationModel
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MedicalModelInference:
    """
    Класс для инференса обученной медицинской классификационной модели.

    Принимает 3D тензор размером 256x256x256 и возвращает предсказание.
    """

    def __init__(self,
                 weights_path: str,
                 model_config,
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

        # Сохраняем оригинальный конфиг ModelConfig
        self.model_config = model_config
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
        print(
            f"📊 Ожидаемый входной размер: 1, 1, {self.model_config.input_D}, {self.model_config.input_H}, {self.model_config.input_W}")
        print(f"🎯 Количество классов: {self.model_config.n_seg_classes}")

    def _load_model(self):
        """Загрузка и инициализация модели."""
        # Преобразуем ModelConfig в OmegaConf
        config_dict = {
            'model': self.model_config.model,
            'model_depth': self.model_config.model_depth,
            'resnet_shortcut': self.model_config.resnet_shortcut,
            'input_W': self.model_config.input_W,
            'input_H': self.model_config.input_H,
            'input_D': self.model_config.input_D,
            'n_seg_classes': self.model_config.n_seg_classes,
            'pretrain_path': self.model_config.pretrain_path if self.model_config.use_pretrained else None,
            'no_cuda': (self.device == 'cpu'),
            'gpu_id': [0] if torch.cuda.is_available() else [],
            'phase': 'test'
        }

        omega_config = OmegaConf.create(config_dict)

        from argparse import Namespace
        args = Namespace(**config_dict)

        # Генерируем модель
        model, _ = generate_model(args)

        # Загружаем веса
        if self.device == 'cpu':
            checkpoint = torch.load(self.weights_path, map_location='cpu', weights_only=True)
        else:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)

        # Проверяем, является ли чекпоинт Lightning модулем или обычной моделью
        if 'state_dict' in checkpoint:
            # Это Lightning чекпоинт
            lightning_model = MedicalClassificationModel.load_from_checkpoint(
                self.weights_path,
                model=model,
                learning_rate=self.model_config.learning_rate,
                num_classes=self.model_config.n_seg_classes
            )
            model = lightning_model.model
        else:
            # Это обычный state_dict
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

        # Перемещаем модель на нужное устройство
        model = model.to(self.device)

        return model

    def _validate_input_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Валидация и предобработка входного тензора.

        Args:
            tensor: входной тензор

        Returns:
            torch.Tensor: валидированный тензор
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(tensor)}")

        if tensor.dim() != 5:
            raise ValueError(f"Ожидается 5D тензор (B, C, D, H, W), получено {tensor.shape}")

        batch_size, channels, depth, height, width = tensor.shape

        # Проверяем размеры
        expected_shape = (1, 1, self.model_config.input_D, self.model_config.input_H, self.model_config.input_W)
        if (channels, depth, height, width) != expected_shape[1:]:
            raise ValueError(
                f"Ожидается тензор формы {expected_shape}, "
                f"получено {tensor.shape}. "
                f"Ожидаемые размеры: channels=1, depth={self.model_config.input_D}, "
                f"height={self.model_config.input_H}, width={self.model_config.input_W}"
            )

        # Перемещаем на нужное устройство и тип данных
        tensor = tensor.to(self.device)

        if self._half_precision:
            tensor = tensor.half()
        else:
            tensor = tensor.float()

        return tensor

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Выполняет предсказание на одном тензоре.

        Args:
            input_tensor: тензор формы (1, 1, input_D, input_H, input_W)

        Returns:
            Dict с результатами предсказания:
            - prediction: класс предсказания (int)
            - probabilities: вероятности всех классов (numpy array)
            - confidence: уверенность в предсказании (float)
            - logits: логиты (numpy array)
        """
        with torch.no_grad():
            # Валидация входа
            input_tensor = self._validate_input_tensor(input_tensor)

            # Выполняем предсказание
            if self._half_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_tensor)
            else:
                logits = self.model(input_tensor)

            # Применяем softmax для получения вероятностей
            probabilities = F.softmax(logits, dim=1)

            # Получаем предсказанный класс
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()

            # Конвертируем в numpy для возврата
            logits_np = logits.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

            return {
                'prediction': int(predicted_class),
                'probabilities': probabilities_np.squeeze(),
                'confidence': float(confidence),
                'logits': logits_np.squeeze()
            }

    def predict_batch(self, batch_tensor: torch.Tensor) -> List[Dict[str, Union[float, int, np.ndarray]]]:
        """
        Выполняет предсказание на батче тензоров.

        Args:
            batch_tensor: тензор формы (B, 1, input_D, input_H, input_W)

        Returns:
            List[Dict] - результаты для каждого элемента в батче
        """
        with torch.no_grad():
            # Проверяем размер батча
            batch_size = batch_tensor.shape[0]

            # Валидация входа
            batch_tensor = self._validate_input_tensor(batch_tensor)

            # Выполняем предсказание
            if self._half_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_tensor)
            else:
                logits = self.model(batch_tensor)

            # Применяем softmax
            probabilities = F.softmax(logits, dim=1)

            # Получаем предсказания для всего батча
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

            # Конвертируем в numpy
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
        """
        Возвращает только вероятности классов.

        Args:
            input_tensor: тензор формы (1, 1, input_D, input_H, input_W)

        Returns:
            np.ndarray: вероятности всех классов
        """
        result = self.predict(input_tensor)
        return result['probabilities']

    def predict_class(self, input_tensor: torch.Tensor) -> int:
        """
        Возвращает только предсказанный класс.

        Args:
            input_tensor: тензор формы (1, 1, input_D, input_H, input_W)

        Returns:
            int: предсказанный класс
        """
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
        weights_path="/model/outputs/weights/best-epoch=00-val_f1=0.6222-val_auroc=0.6858.ckpt",

        model_config=config
    )

    # Тестовый тензор 256x256x256
    test_tensor = torch.randn(1, 1, 256, 256, 256)

    # Получаем предсказание
    result = inference.predict(test_tensor)

    print(f"Предсказанный класс: {result['prediction']}")
    print(f"Уверенность: {result['confidence']:.4f}")
    print(f"Вероятности классов: {result['probabilities']}")

    # Пример батчевого предсказания
    batch_tensor = torch.randn(3, 1, 256, 256, 256)
    batch_results = inference.predict_batch(batch_tensor)

    for i, res in enumerate(batch_results):
        print(f"Образец {i}: класс={res['prediction']}, уверенность={res['confidence']:.4f}")