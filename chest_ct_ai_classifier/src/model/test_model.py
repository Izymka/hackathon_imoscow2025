import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from model_evaluate import evaluate_model, plot_roc_curve
from datasets.medical_tensors import MedicalTensorDataset
from model_generator import generate_model
from config import ModelConfig
import numpy as np
from collections import OrderedDict


def find_best_weights():
    """Поиск лучших весов (.ckpt или .pth)"""
    # Сначала ищем .ckpt файлы
    checkpoint_path = 'model/outputs/weights/best-epoch=17-val_f1=0.8682-val_recall=0.8615-val_specificity=0.8644--val_auroc=0.9168.ckpt'

    if os.path.exists(checkpoint_path):
        print(f"✅ Найден чекпоинт: {checkpoint_path}")
        return checkpoint_path, 'ckpt'

    # Ищем .pth файлы
    pth_paths = [
        'model/outputs/weights/best_model.pth',
        'model/outputs/weights/model.pth',
        'model/weights/best_model.pth',
        'weights/best_model.pth',
        'best_model.pth'
    ]

    for pth_path in pth_paths:
        if os.path.exists(pth_path):
            print(f"✅ Найден .pth файл: {pth_path}")
            return pth_path, 'pth'

    # Альтернативные пути для .ckpt
    checkpoint_dir = 'model/outputs/checkpoints'
    if os.path.exists(checkpoint_dir):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_files[0])
            print(f"🔄 Используем найденный .ckpt: {checkpoint_path}")
            return checkpoint_path, 'ckpt'

    # Альтернативные пути для .pth
    weights_dirs = ['model/outputs/weights', 'model/weights', 'weights', '.']
    for weights_dir in weights_dirs:
        if os.path.exists(weights_dir):
            pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
            if pth_files:
                pth_files.sort(key=lambda x: os.path.getctime(os.path.join(weights_dir, x)), reverse=True)
                pth_path = os.path.join(weights_dir, pth_files[0])
                print(f"🔄 Используем найденный .pth: {pth_path}")
                return pth_path, 'pth'

    print(f"❌ Файлы весов (.ckpt или .pth) не найдены")
    return None, None


def load_model_weights(weights_path, weights_type, cfg, device):
    """Загрузка модели из .ckpt или .pth файла"""
    print("=" * 60)
    print("🔧 Создание модели для тестирования...")

    # Преобразуем ModelConfig в namespace
    class OptNamespace:
        pass

    opt = OptNamespace()
    for key, value in cfg.__dict__.items():
        setattr(opt, key, value)

    # Важно: phase='test' для правильного режима
    opt.phase = 'test'
    opt.gpu_id = [0] if not cfg.no_cuda and torch.cuda.is_available() else []

    # Создаем модель ТОЧНО ТАК ЖЕ, как в train.py
    model, _ = generate_model(opt)
    print("=" * 60)

    # Загрузка файла весов
    try:
        if weights_type == 'pth':
            # Для .pth файлов используем weights_only=True для безопасности
            weights = torch.load(weights_path, map_location=device, weights_only=True)
            print(f"📁 Загружен .pth файл. Тип данных: {type(weights)}")

            # .pth файлы обычно содержат прямой state_dict
            if isinstance(weights, dict):
                if 'state_dict' in weights:
                    state_dict = weights['state_dict']
                    print("🔧 Обнаружен формат с ключом 'state_dict'")
                elif 'model_state_dict' in weights:
                    state_dict = weights['model_state_dict']
                    print("🔧 Обнаружен формат с ключом 'model_state_dict'")
                else:
                    state_dict = weights
                    print("🔧 Обнаружен прямой state_dict в .pth")
            else:
                print("❌ Неожиданный формат .pth файла")
                return None
        else:
            # Для .ckpt файлов (PyTorch Lightning)
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            print(f"📁 Загружен .ckpt файл. Доступные ключи: {list(checkpoint.keys())}")

            # Извлечение state_dict из checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("🔧 Обнаружен формат PyTorch Lightning (state_dict)")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("🔧 Обнаружен формат model_state_dict")
            else:
                state_dict = checkpoint
                print("🔧 Обнаружен прямой state_dict")

    except Exception as e:
        print(f"❌ Ошибка загрузки файла весов: {e}")
        return None

    # Обработка префиксов
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Убираем все возможные префиксы
        new_key = k
        if k.startswith('model.'):
            new_key = k[6:]
        elif k.startswith('module.'):
            new_key = k[7:]
        elif k.startswith('net.'):
            new_key = k[4:]
        elif k.startswith('encoder.'):
            new_key = k[8:]

        # Пропускаем служебные параметры
        if not any(skip in new_key for skip in ['loss_fn', 'criterion', 'optimizer', 'scheduler']):
            new_state_dict[new_key] = v

    # Загрузка state_dict в модель
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(
                f"⚠️ Отсутствующие ключи ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(
                f"⚠️ Неожиданные ключи ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        if not missing_keys and not unexpected_keys:
            print("✅ Все ключи загружены успешно!")

    except Exception as e:
        print(f"❌ Ошибка загрузки state_dict: {e}")
        # Загружаем только совпадающие ключи
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in new_state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"🔄 Загружено {len(filtered_dict)} из {len(new_state_dict)} ключей")

    model = model.to(device)
    model.eval()

    print(f"✅ Модель успешно загружена из {weights_type.upper()} файла!")
    return model


def plot_confusion_matrix_detailed(cm, class_names, output_dir):
    """Построение детальной матрицы путаницы"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    plt.close()


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

    # Поиск файлов весов
    weights_path, weights_type = find_best_weights()
    if not weights_path:
        print("❌ Не найдены файлы весов (.ckpt или .pth) для тестирования")
        return

    # 👉 ИСПОЛЬЗУЕМ ТОТ ЖЕ ModelConfig, что и в train.py
    print("=" * 60)
    print("⚙️  ЗАГРУЗКА КОНФИГУРАЦИИ")
    print("=" * 60)
    cfg = ModelConfig()
    print(cfg.get_training_summary())
    print("=" * 60)

    # Загрузка модели с правильной конфигурацией
    model = load_model_weights(weights_path, weights_type, cfg, device)
    if model is None:
        return

    # Пути к тестовым данным
    test_data_root = 'data/test/'
    test_labels_path = 'data/test/labels.csv'

    # Проверка существования путей
    if not os.path.exists(test_data_root):
        print(f"❌ Тестовые данные не найдены: {test_data_root}")
        return

    if not os.path.exists(test_labels_path):
        print(f"❌ Файл с метками не найден: {test_labels_path}")
        return

    print(f"📂 Используем тестовые данные:")
    print(f"   - Данные: {test_data_root}")
    print(f"   - Метки: {test_labels_path}")

    # Создание namespace для датасета - ТОЧНО КАК В TRAIN.PY
    class OptNamespace:
        pass

    opt = OptNamespace()
    for key, value in cfg.__dict__.items():
        setattr(opt, key, value)
    opt.phase = 'test'  # 👈 ВАЖНО: режим теста
    opt.pin_memory = cfg.pin_memory

    # Создание тестового датасета
    print("📊 Создание тестового датасета...")
    test_dataset = MedicalTensorDataset(
        data_root=test_data_root,
        img_list=test_labels_path,
        sets=opt
    )

    print(f"📈 Размер тестового датасета: {len(test_dataset)} образцов")

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Для стабильности при тестировании
    )

    # Создание директории для результатов
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)

    # Оценка модели
    print("=" * 60)
    print("🔄 Запуск тестирования...")
    print("=" * 60)

    class_names = ['Норма', 'Патология']

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

    # Построение ROC-кривой
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

    print(f"\n🎯 Итоговый результат: {results['accuracy'] * 100:.2f}% точности")
    print("=" * 60)


if __name__ == "__main__":
    main()