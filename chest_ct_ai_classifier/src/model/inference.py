import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, List, Optional
from pathlib import Path
import argparse

# Изменяем импорты для работы как модуля, так и как standalone скрипта
try:
    from .config import ModelConfig
    # from .model_generator import generate_model  # Убрали прямой импорт
except ImportError:
    from config import ModelConfig
    # from model_generator import generate_model  # Убрали прямой импорт

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn

# Исправляем импорты для работы как модуля, так и как standalone скрипта
try:
    from .models import resnet
except ImportError:
    try:
        from models import resnet
    except ImportError:
        import models.resnet as resnet

def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        # Исправленная загрузка для CPU/GPU совместимости
        if opt.no_cuda or not torch.cuda.is_available():
            pretrain = torch.load(opt.pretrain_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            pretrain = torch.load(opt.pretrain_path, weights_only=True)

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()


class MedicalModelInference:
    """
    Класс для инференса обученной медицинской модели в продакшене.
    
    Поддерживает:
    - Загрузку обученной модели
    - Предсказание для одного образца
    - Пакетные предсказания
    - Возврат вероятностей и классов
    - GPU/CPU совместимость
    """
    
    def __init__(self, 
                 weights_path: str, 
                 model_config: Optional[ModelConfig] = None,
                 device: Optional[str] = None):
        """
        Инициализация inference класса.
        
        Args:
            weights_path: Путь к файлу весов модели (.pth)
            model_config: Конфигурация модели (если None, используется по умолчанию)
            device: Устройство для вычислений ('cuda', 'cpu', или None для автовыбора)
        """
        self.weights_path = Path(weights_path)
        self.config = model_config or ModelConfig()
        
        # Определение устройства
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Используем устройство: {self.device}")
        
        # Загрузка модели
        self.model = self._load_model()
        self.model.eval()
        
        # Названия классов
        self.class_names = ['Здоров', 'Болен']  # Измените под свои классы
        
        print("✅ Модель загружена и готова для предсказаний!")
        
        # Добавляем проверку работоспособности модели
        self._test_model()
    
    def _test_model(self):
        """Проверяем, что модель действительно работает."""
        try:
            # Создаем тестовый тензор
            test_input = torch.randn(1, 1, self.config.input_D, 
                                   self.config.input_H, self.config.input_W).to(self.device)
            
            # Пробуем получить вывод модели
            with torch.no_grad():
                test_output = self.model(test_input)
                
            print(f"🧪 Тест модели прошел успешно. Форма вывода: {test_output.shape}")
            print(f"🧪 Пример вывода: {test_output[0].cpu().numpy()}")
            
        except Exception as e:
            print(f"⚠️ Ошибка при тестировании модели: {e}")
            raise
    
    def _load_model(self):
        """Загрузка модели с весами."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Файл весов не найден: {self.weights_path}")
        
        print(f"📝 Конфигурация модели:")
        print(f"   - Тип модели: {self.config.model}")
        print(f"   - Глубина: {self.config.model_depth}")
        print(f"   - Классы: {self.config.n_seg_classes}")
        print(f"   - Размер входа: {self.config.input_D}x{self.config.input_H}x{self.config.input_W}")
        
        # Создание конфигурации для генерации модели
        cfg_dict = {
            'model': self.config.model,
            'model_depth': self.config.model_depth,
            'n_seg_classes': self.config.n_seg_classes,
            'no_cuda': self.device.type == 'cpu',
            'input_W': self.config.input_W,
            'input_H': self.config.input_H,
            'input_D': self.config.input_D,
            'resnet_shortcut': self.config.resnet_shortcut,
            'gpu_id': [] if self.device.type == 'cpu' else [0],
            'phase': 'test',
            'pin_memory': False,
            'pretrain_path': '',
            'new_layer_names': self.config.new_layer_names
        }
        
        cfg_namespace = argparse.Namespace(**cfg_dict)
        
        # Генерация модели
        print("🏗️ Создаем архитектуру модели...")
        model, _ = generate_model(cfg_namespace)
        
        # Загрузка весов
        print(f"📂 Загрузка весов из: {self.weights_path}")
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"❌ Ошибка загрузки весов: {e}")
            # Пробуем загрузить без weights_only для старых версий PyTorch
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device)
            except Exception as e2:
                print(f"❌ Критическая ошибка загрузки весов: {e2}")
                raise
        
        # Обработка различных форматов весов
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"📊 Найдено {len(state_dict)} параметров в checkpoint")
        
        # Очистка префиксов в ключах (если модель была обучена с DataParallel)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # убираем 'module.'
            elif k.startswith('model.'):
                name = k[6:]  # убираем 'model.'
            else:
                name = k
            new_state_dict[name] = v
        
        # Проверяем соответствие ключей модели и checkpoint
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"⚠️ Отсутствующие ключи в checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Неожиданные ключи в checkpoint: {unexpected_keys}")
        
        # Загружаем веса с проверкой
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Веса успешно загружены")
        except Exception as e:
            print(f"❌ Ошибка загрузки весов в модель: {e}")
            raise
        
        model = model.to(self.device)
        
        return model
    
    def _validate_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Валидация входного тензора."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Вход должен быть torch.Tensor")
        
        expected_shape = (self.config.input_D, self.config.input_H, self.config.input_W)
        
        # Если тензор имеет форму [1, 1, D, H, W] - это правильно
        if len(tensor.shape) == 5 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
            actual_shape = tensor.shape[2:]
            if actual_shape != expected_shape:
                raise ValueError(f"Неправильный размер тензора: {actual_shape}, ожидался: {expected_shape}")
        
        # Если тензор имеет форму [1, D, H, W] - добавляем channel dimension
        elif len(tensor.shape) == 4 and tensor.shape[0] == 1:
            actual_shape = tensor.shape[1:]
            if actual_shape != expected_shape:
                raise ValueError(f"Неправильный размер тензора: {actual_shape}, ожидался: {expected_shape}")
            tensor = tensor.unsqueeze(1)  # [1, 1, D, H, W]
        
        # Если тензор имеет форму [D, H, W] - добавляем batch и channel dimensions
        elif len(tensor.shape) == 3:
            if tensor.shape != expected_shape:
                raise ValueError(f"Неправильный размер тензора: {tensor.shape}, ожидался: {expected_shape}")
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        else:
            raise ValueError(f"Неподдерживаемая форма тензора: {tensor.shape}")
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Union[int, float, List[float]]]:
        """
        Предсказание для одного образца.
        
        Args:
            input_tensor: Входной тензор [1, 1, 128, 128, 128] или [128, 128, 128]
            
        Returns:
            dict: {
                'predicted_class': int,
                'predicted_class_name': str,
                'confidence': float,
                'probabilities': List[float]
            }
        """
        # Валидация и подготовка входа
        tensor = self._validate_input(input_tensor)
        
        print(f"🔍 Входной тензор после валидации: {tensor.shape}")
        print(f"🔍 Статистика входного тензора: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        
        # Предсказание
        outputs = self.model(tensor)
        print(f"🔍 Сырой вывод модели: {outputs}")
        print(f"🔍 Форма вывода: {outputs.shape}")
        
        probabilities = F.softmax(outputs, dim=1)
        print(f"🔍 Вероятности после softmax: {probabilities}")
        
        # Извлечение результатов
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        probs_list = probabilities[0].cpu().numpy().tolist()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probs_list
        }
        
        return result

def debug_inference():
    """Функция для отладки inference."""
    print("🔧 Режим отладки inference...")
    
    # Проверяем путь к весам
    weights_path = "model/outputs/weights/best_weights.pth"
    if not Path(weights_path).exists():
        print(f"❌ Файл весов не найден: {weights_path}")
        # Попробуем найти альтернативные пути
        possible_paths = [
            "model/outputs/checkpoints/best_weights.pth",
            "outputs/weights/best_weights.pth",
            "outputs/checkpoints/best_weights.pth"
        ]
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ Найден альтернативный путь: {path}")
                weights_path = path
                break
        else:
            print("❌ Веса не найдены ни по одному из возможных путей")
            return
    
    # Создаем inference
    try:
        inference = MedicalModelInference(weights_path)
        
        # Создаем тестовый тензор
        pneumotorax = torch.randn(128, 128, 128)  # Ваш формат данных
        
        # Делаем предсказание
        result = inference.predict(pneumotorax)
        
        print(f"✅ Результат предсказания: {result}")
        
    except Exception as e:
        print(f"❌ Ошибка в debug_inference: {e}")
        import traceback
        traceback.print_exc()


def read_tensor(path: Path) -> torch.Tensor:
    """Чтение тензора из PyTorch файла (.pt, .pth)"""
    try:
        tensor = torch.load(path, weights_only=False, map_location='cpu')

        # Если это словарь с 'state_dict' или 'tensor', извлекаем тензор
        if isinstance(tensor, dict):
            if 'tensor' in tensor:
                tensor = tensor['tensor']
            elif 'state_dict' in tensor:
                # Это чекпоинт модели, возвращаем как есть или извлекаем тензор
                raise ValueError(
                    "Файл содержит state_dict модели, а не тензор. Используйте load_state_dict для модели.")
            else:
                # Если словарь содержит тензор, извлекаем первый
                for key, value in tensor.items():
                    if isinstance(value, torch.Tensor):
                        return value
                raise ValueError("Не найден тензор в словаре")

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Файл не содержит тензор, а содержит: {type(tensor)}")

        return tensor

    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении PyTorch файла: {e}")

if __name__ == "__main__":
    #debug_inference()
    test_path = Path("../data/test_ideal/norma_anon.pt")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found at path: {test_path}")
    pneumonia = read_tensor(test_path)
    inference = MedicalModelInference("outputs/weights/best_weights.pth")
    predict_result = inference.predict(pneumonia)
    print(predict_result)