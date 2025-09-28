# model/test_model.py
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from evaluate_model import evaluate_model, plot_roc_curve
from datasets.medical_tensors import MedicalTensorDataset
from monai.metrics import ConfusionMatrixMetric
from model_generator import generate_model
import argparse
import numpy as np


def create_test_config():
    """Создание тестовой конфигурации"""
    config = argparse.Namespace()
    config.model = 'resnet'
    config.model_depth = 10  # измени при необходимости
    config.n_seg_classes = 2  # количество классов
    config.no_cuda = not torch.cuda.is_available()
    config.input_D = 128
    config.input_H = 128
    config.input_W = 128
    config.resnet_shortcut = 'A'
    config.gpu_id = [] if config.no_cuda else [0]
    config.phase = 'test'
    config.pin_memory = not config.no_cuda
    config.pretrain_path = ''
    config.new_layer_names = ['fc']
    
    return config


def find_best_checkpoint(checkpoint_dirs):
    """Поиск лучшего чекпоинта"""
    checkpoint_path = None
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            # Ищем файлы с расширением .pth или .pth.tar
            checkpoints = [f for f in os.listdir(checkpoint_dir) 
                         if f.endswith('.pth') or f.endswith('.pth.tar')]
            if checkpoints:
                checkpoints.sort()
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Найден чекпоинт: {checkpoint_path}")
                break
    
    return checkpoint_path


def load_model_checkpoint(checkpoint_path, config, device):
    """Загрузка модели из чекпоинта"""
    print("Загрузка модели...")
    
    # Создание модели
    model, _ = generate_model(config)
    
    # Загрузка чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Обработка различных форматов чекпоинтов
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Обработка префиксов в ключах (если модель обучалась с DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # убираем 'module.'
        else:
            name = k
        # Пропускаем всё, что относится к loss_fn
        if not name.startswith('loss_fn'):
            new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def plot_confusion_matrix_detailed(cm, class_names, output_dir):
    """Построение детальной матрицы путаницы"""
    plt.figure(figsize=(10, 8))
    
    # Нормализованная матрица путаницы
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем subplot'ы
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Количество случаев'})
    ax1.set_title('Матрица путаницы (абсолютные значения)')
    ax1.set_ylabel('Истинные метки')
    ax1.set_xlabel('Предсказанные метки')
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Доля от общего количества'})
    ax2.set_title('Матрица путаницы (нормализованная)')
    ax2.set_ylabel('Истинные метки')
    ax2.set_xlabel('Предсказанные метки')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def print_detailed_metrics(results, class_names):
    """Печать детальных метрик"""
    print(f"\n{'=' * 60}")
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"{'=' * 60}")
    
    # Основные метрики
    print(f"📊 Общая точность: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"📈 Средняя потеря: {results['loss']:.4f}")
    
    # Анализ матрицы путаницы
    cm = results['confusion_matrix']
    print(f"\n🔍 Анализ матрицы путаницы:")
    print(f"Матрица путаницы:")
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            print(f"  {true_class} → {pred_class}: {cm[i, j]}")
    
    # Специфичность и чувствительность для бинарной классификации
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"\n🩺 Клинические метрики:")
        print(f"  Чувствительность (Sensitivity/Recall): {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
        print(f"  Специфичность (Specificity): {specificity:.4f} ({specificity * 100:.2f}%)")
        print(f"  Положительная прогностическая ценность (PPV): {ppv:.4f} ({ppv * 100:.2f}%)")
        print(f"  Отрицательная прогностическая ценность (NPV): {npv:.4f} ({npv * 100:.2f}%)")
    
    # Детальный отчет по классам
    print(f"\n📋 Подробный отчет по классам:")
    print(classification_report(results['labels'], results['predictions'], 
                               target_names=class_names))


def main():
    # Настройки устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Используем устройство: {device}")
    
    # Пути для поиска чекпоинтов (расширенный список)
    checkpoint_dirs = [
        'model/outputs/weights',
        'model/outputs/checkpoints',    # новый путь из config.py
        'trails/models/resnet_10',      # старый путь
        'outputs/checkpoints',          # возможный путь
        'checkpoints',                  # еще один возможный путь
    ]
    
    # Поиск чекпоинта
    checkpoint_path = find_best_checkpoint(checkpoint_dirs)
    
    if not checkpoint_path:
        print("❌ Чекпоинт не найден!")
        print("📁 Проверьте следующие директории:")
        for dir_path in checkpoint_dirs:
            print(f"   - {dir_path}")
        return
    
    # Создание конфигурации
    config = create_test_config()
    
    # Загрузка модели
    model = load_model_checkpoint(checkpoint_path, config, device)
    print("✅ Модель успешно загружена!")
    
    # Пути к тестовым данным (расширенный поиск)
    test_data_paths = [
        ('data/test/tensors', 'data/test/labels.csv'),
        ('data/tensors', 'data/test_labels.csv'),
        ('../data/tensors', '../data/test_labels.csv'),
        ('data/processed', 'data/processed/labels.csv'),
    ]
    
    # Поиск тестовых данных
    test_data_root = None
    test_labels_path = None
    
    for data_root, labels_path in test_data_paths:
        if os.path.exists(data_root) and os.path.exists(labels_path):
            test_data_root = data_root
            test_labels_path = labels_path
            break
    
    if not test_data_root:
        print("❌ Тестовые данные не найдены!")
        print("📁 Проверьте следующие пути:")
        for data_root, labels_path in test_data_paths:
            print(f"   - Данные: {data_root}, Метки: {labels_path}")
        return
    
    print(f"📂 Используем тестовые данные:")
    print(f"   - Данные: {test_data_root}")
    print(f"   - Метки: {test_labels_path}")
    
    # Создание тестового датасета
    print("📊 Создание тестового датасета...")
    test_dataset = MedicalTensorDataset(
        data_root=test_data_root,
        img_list=test_labels_path,
        sets=config
    )
    
    print(f"📈 Размер тестового датасета: {len(test_dataset)} образцов")
    
    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Создание директории для результатов
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Оценка модели
    print("🔄 Запуск тестирования...")
    class_names = ['Здоров', 'Болен']  # измени под свои классы
    
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_results=True,
        output_dir=output_dir
    )
    
    # Печать детальных метрик
    print_detailed_metrics(results, class_names)
    
    # Построение детальной матрицы путаницы
    plot_confusion_matrix_detailed(results['confusion_matrix'], class_names, output_dir)
    
    # Построение ROC-кривой (если доступно)
    if 'probabilities' in results:
        plot_roc_curve(results['labels'], results['probabilities'], class_names, output_dir)
    
    print(f"\n💾 Все результаты сохранены в директории: {output_dir}")
    print("📋 Созданные файлы:")
    print("   - evaluation_metrics.txt - текстовые метрики")
    print("   - confusion_matrix.png - матрица путаницы")
    print("   - detailed_confusion_matrix.png - детальная матрица путаницы")
    print("   - predictions.csv - предсказания для каждого образца")
    if 'probabilities' in results:
        print("   - roc_curve.png - ROC-кривая")
    
    print(f"\n🎯 Итоговый результат: {results['accuracy']*100:.2f}% точности")


if __name__ == "__main__":
    main()