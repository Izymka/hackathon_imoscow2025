# production_example.py
"""
Пример использования модели в продакшене.
Показывает различные сценарии использования inference класса.
"""

import torch
import numpy as np
from pathlib import Path
import time
from inference import MedicalModelInference
from config import ModelConfig


class ProductionPipeline:
    """Пример production pipeline для медицинской классификации."""
    
    def __init__(self, weights_path: str, config: ModelConfig = None):
        """
        Инициализация production pipeline.
        
        Args:
            weights_path: Путь к файлу весов модели
            config: Конфигурация модели
        """
        self.inference = MedicalModelInference(weights_path, config)
        self.processing_times = []
        
    def preprocess_data(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Предобработка входных данных.
        
        Args:
            raw_data: Сырые данные (например, из DICOM файла)
            
        Returns:
            torch.Tensor: Предобработанный тензор
        """
        # Здесь может быть ваша логика предобработки:
        # - нормализация
        # - изменение размера
        # - фильтрация шумов и т.д.
        
        # Простой пример нормализации
        data = raw_data.astype(np.float32)
        data = (data - data.mean()) / data.std()
        
        # Убеждаемся, что размер правильный
        if data.shape != (128, 128, 128):
            # Здесь может быть resize логика
            print(f"⚠️  Предупреждение: размер данных {data.shape} не соответствует ожидаемому (128, 128, 128)")
        
        # Преобразуем в тензор с правильными размерностями
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        return tensor
    
    def process_single_case(self, raw_data: np.ndarray, case_id: str = None) -> dict:
        """
        Обработка одного случая.
        
        Args:
            raw_data: Сырые медицинские данные
            case_id: Идентификатор случая
            
        Returns:
            dict: Результат анализа
        """
        start_time = time.time()
        
        # Предобработка
        tensor = self.preprocess_data(raw_data)
        
        # Предсказание
        prediction = self.inference.predict(tensor)
        
        # Постобработка результатов
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        result = {
            'case_id': case_id,
            'diagnosis': prediction['predicted_class_name'],
            'confidence': prediction['confidence'],
            'probabilities': {
                'healthy': prediction['probabilities'][0],
                'pathology': prediction['probabilities'][1]
            },
            'processing_time_seconds': processing_time,
            'recommendation': self._generate_recommendation(prediction)
        }
        
        return result
    
    def process_batch_cases(self, raw_data_list: list, case_ids: list = None) -> list:
        """
        Пакетная обработка множества случаев.
        
        Args:
            raw_data_list: Список сырых данных
            case_ids: Список идентификаторов случаев
            
        Returns:
            list: Список результатов анализа
        """
        if case_ids is None:
            case_ids = [f"case_{i+1}" for i in range(len(raw_data_list))]
        
        start_time = time.time()
        
        # Предобработка всех случаев
        tensors = []
        for raw_data in raw_data_list:
            tensor = self.preprocess_data(raw_data)
            tensors.append(tensor.squeeze(0))  # убираем batch dimension для объединения
        
        # Объединяем в один батч
        batch_tensor = torch.stack(tensors)  # [N, 1, D, H, W]
        
        # Пакетное предсказание
        predictions = self.inference.predict_batch(batch_tensor)
        
        # Постобработка результатов
        total_time = time.time() - start_time
        avg_time_per_case = total_time / len(raw_data_list)
        
        results = []
        for i, (case_id, prediction) in enumerate(zip(case_ids, predictions)):
            result = {
                'case_id': case_id,
                'diagnosis': prediction['predicted_class_name'],
                'confidence': prediction['confidence'],
                'probabilities': {
                    'healthy': prediction['probabilities'][0],
                    'pathology': prediction['probabilities'][1]
                },
                'processing_time_seconds': avg_time_per_case,
                'recommendation': self._generate_recommendation(prediction)
            }
            results.append(result)
        
        return results
    
    def _generate_recommendation(self, prediction: dict) -> str:
        """
        Генерация медицинской рекомендации на основе предсказания.
        
        Args:
            prediction: Результат предсказания модели
            
        Returns:
            str: Текстовая рекомендация
        """
        confidence = prediction['confidence']
        predicted_class = prediction['predicted_class']
        
        if predicted_class == 0:  # Здоров
            if confidence > 0.9:
                return "Высокая вероятность отсутствия патологии. Рекомендуется стандартное наблюдение."
            elif confidence > 0.7:
                return "Вероятность отсутствия патологии. Рекомендуется контрольное обследование через 6 месяцев."
            else:
                return "Неопределенный результат. Рекомендуется дополнительное обследование и консультация специалиста."
        else:  # Болен
            if confidence > 0.9:
                return "Высокая вероятность патологии. Требуется срочная консультация специалиста и дополнительные исследования."
            elif confidence > 0.7:
                return "Вероятность патологии. Рекомендуется консультация специалиста в ближайшее время."
            else:
                return "Подозрение на патологию. Рекомендуется дополнительное обследование и консультация специалиста."
    
    def get_performance_stats(self) -> dict:
        """Статистика производительности."""
        if not self.processing_times:
            return {"message": "Статистика недоступна - не было выполнено ни одного предсказания"}
        
        return {
            "total_processed": len(self.processing_times),
            "average_time_seconds": np.mean(self.processing_times),
            "min_time_seconds": np.min(self.processing_times),
            "max_time_seconds": np.max(self.processing_times),
            "cases_per_minute": 60 / np.mean(self.processing_times)
        }


def main():
    """Демонстрация использования в продакшене."""
    print("🏥 Демонстрация Production Pipeline для медицинской классификации")
    print("=" * 70)
    
    # Пути к модели (измените под свои пути)
    weights_path = "model/outputs/checkpoints/best_weights.pth"
    
    try:
        # Инициализация pipeline
        pipeline = ProductionPipeline(weights_path)
        
        print("✅ Pipeline инициализирован успешно!")
        
        # Демонстрация 1: Обработка одного случая
        print("\n📋 Демонстрация 1: Обработка одного случая")
        print("-" * 50)
        
        # Создаем тестовые данные (в реальности это будут ваши медицинские данные)
        test_data = np.random.randn(128, 128, 128).astype(np.float32)
        
        result = pipeline.process_single_case(test_data, case_id="PATIENT_001")
        
        print(f"🏥 Случай: {result['case_id']}")
        print(f"🔬 Диагноз: {result['diagnosis']}")
        print(f"📊 Уверенность: {result['confidence']:.2%}")
        print(f"⏱️ Время обработки: {result['processing_time_seconds']:.3f} сек")
        print(f"💡 Рекомендация: {result['recommendation']}")
        
        # Демонстрация 2: Пакетная обработка
        print("\n📦 Демонстрация 2: Пакетная обработка")
        print("-" * 50)
        
        # Создаем несколько тестовых случаев
        batch_data = [np.random.randn(128, 128, 128).astype(np.float32) for _ in range(3)]
        case_ids = ["PATIENT_002", "PATIENT_003", "PATIENT_004"]
        
        batch_results = pipeline.process_batch_cases(batch_data, case_ids)
        
        for result in batch_results:
            print(f"🏥 {result['case_id']}: {result['diagnosis']} "
                  f"({result['confidence']:.2%} уверенности)")
        
        # Статистика производительности
        print("\n📈 Статистика производительности")
        print("-" * 50)
        
        stats = pipeline.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Информация о модели
        print("\n🤖 Информация о модели")
        print("-" * 50)
        
        model_info = pipeline.inference.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
    
    except FileNotFoundError:
        print("❌ Файл весов не найден!")
        print(f"💡 Убедитесь, что файл существует: {weights_path}")
        print("💡 Обучите модель сначала, запустив: python main.py")
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
