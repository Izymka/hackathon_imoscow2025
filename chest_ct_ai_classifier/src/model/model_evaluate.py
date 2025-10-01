import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, \
    average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from monai.metrics import compute_roc_auc
from monai.metrics import ConfusionMatrixMetric
import os
from tqdm import tqdm
import json


def evaluate_model(model, test_loader, device, class_names=None, save_results=True, output_dir='results'):
    """
    Оценка качества модели с подробными метриками для медицинских задач

    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        device: устройство (cuda/cpu)
        class_names: список имен классов
        save_results: сохранять ли результаты
        output_dir: директория для сохранения результатов
    """

    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Автоматическое определение устройства, если не указано
    if device is None:
        device = next(model.parameters()).device

    # Обработка DataParallel моделей
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model = model.to(device)

    # Переводим модель в режим оценки
    model.eval()

    # Списки для хранения результатов
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_filenames = []

    # Критерий для вычисления loss
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    print("Начинаем оценку модели...")

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Оценка")):
            data, targets = data.to(device), targets.to(device)

            # Прогноз
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Вероятности и предсказания
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # Сохраняем результаты
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # Очистка памяти для GPU
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Преобразуем в numpy массивы
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Вычисляем метрики
    accuracy = np.mean(all_predictions == all_labels)
    avg_loss = total_loss / len(test_loader)

    # Классификационный отчет
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(all_labels)))]

    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_predictions)

    # Дополнительные метрики для бинарной классификации
    additional_metrics = {}
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()

        # Базовые метрики
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Чувствительность (Recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Специфичность
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Точность (Precision)

        # Производные метрики
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

        # Отношения правдоподобия
        plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')  # Positive Likelihood Ratio
        nlr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')  # Negative Likelihood Ratio

        # AUC-ROC
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])

        # Average Precision
        avg_precision = average_precision_score(all_labels, all_probabilities[:, 1])

        additional_metrics = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'plr': plr,
            'nlr': nlr,
            'auc_roc': auc_score,
            'average_precision': avg_precision,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }

    # Вывод результатов
    print(f"\n{'=' * 60}")
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"📊 Общая точность (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"📈 Средний Loss: {avg_loss:.4f}")

    if len(class_names) == 2:
        print(f"\n🩺 МЕДИЦИНСКИЕ МЕТРИКИ (бинарная классификация):")
        print(f"  • Чувствительность (Sensitivity/Recall): {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
        print(f"  • Специфичность (Specificity): {specificity:.4f} ({specificity * 100:.2f}%)")
        print(f"  • Точность (Precision/PPV): {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  • F1-Score: {f1_score:.4f} ({f1_score * 100:.2f}%)")
        print(f"  • NPV: {npv:.4f} ({npv * 100:.2f}%)")
        print(f"  • AUC-ROC: {auc_score:.4f}")
        print(f"  • Average Precision: {avg_precision:.4f}")
        print(f"\n📋 КОНТИНГЕНТНАЯ ТАБЛИЦА:")
        print(f"  • True Positive (TP): {tp}")
        print(f"  • True Negative (TN): {tn}")
        print(f"  • False Positive (FP): {fp}")
        print(f"  • False Negative (FN): {fn}")

    # Подробный классификационный отчет
    print(f"\n📋 Подробный отчет по классам:")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))

    # Сохранение результатов
    if save_results:
        save_evaluation_results(
            all_labels, all_predictions, all_probabilities,
            cm, accuracy, avg_loss, class_names, output_dir, additional_metrics
        )

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'additional_metrics': additional_metrics
    }


def save_evaluation_results(labels, predictions, probabilities, cm, accuracy, loss, class_names, output_dir,
                            additional_metrics=None):
    """Сохранение результатов оценки"""

    # Сохранение метрик в JSON файл
    metrics_dict = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

    if additional_metrics:
        # Конвертируем numpy значения в float для JSON
        for key, value in additional_metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                metrics_dict[key] = float(value)
            else:
                metrics_dict[key] = value

    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    # Сохранение метрик в текстовый файл
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ\n")
        f.write("=" * 60 + "\n")
        f.write(f"Точность (Accuracy): {accuracy:.4f}\n")
        f.write(f"Средний Loss: {loss:.4f}\n\n")

        if additional_metrics:
            f.write("МЕДИЦИНСКИЕ МЕТРИКИ:\n")
            f.write(f"  Чувствительность (Sensitivity): {additional_metrics.get('sensitivity', 0):.4f}\n")
            f.write(f"  Специфичность (Specificity): {additional_metrics.get('specificity', 0):.4f}\n")
            f.write(f"  Точность (Precision): {additional_metrics.get('precision', 0):.4f}\n")
            f.write(f"  F1-Score: {additional_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"  NPV: {additional_metrics.get('npv', 0):.4f}\n")
            f.write(f"  AUC-ROC: {additional_metrics.get('auc_roc', 0):.4f}\n")
            f.write(f"  Average Precision: {additional_metrics.get('average_precision', 0):.4f}\n\n")

        f.write("Классификационный отчет:\n")
        f.write(classification_report(labels, predictions, target_names=class_names))

    # Сохранение матрицы ошибок как изображение
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Количество случаев'})
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Сохранение предсказаний
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'true_label_name': [class_names[i] for i in labels],
        'predicted_label_name': [class_names[i] for i in predictions]
    })

    # Добавляем вероятности для каждого класса
    for i, class_name in enumerate(class_names):
        results_df[f'probability_{class_name}'] = probabilities[:, i]

    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, encoding='utf-8')

    print(f"\n💾 Результаты сохранены в директорию: {output_dir}")


def plot_roc_curve(labels, probabilities, class_names, output_dir='results'):
    """Построение ROC-кривой и Precision-Recall кривой"""

    plt.figure(figsize=(15, 6))

    # ROC-кривая
    plt.subplot(1, 2, 1)

    if len(class_names) == 2:
        # Бинарная классификация
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        auc_score = roc_auc_score(labels, probabilities[:, 1])
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {auc_score:.4f})')

        # Precision-Recall кривая
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        avg_precision = average_precision_score(labels, probabilities[:, 1])
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR Curve (AP = {avg_precision:.4f})')

    else:
        # Многоклассовая классификация
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

        # ROC-кривые для каждого класса
        plt.subplot(1, 2, 1)
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            try:
                fpr, tpr, _ = roc_curve(labels == i, probabilities[:, i])
                auc_score = roc_auc_score(labels == i, probabilities[:, i])
                plt.plot(fpr, tpr, color=color, lw=2,
                         label=f'{class_name} (AUC = {auc_score:.4f})')
            except Exception as e:
                print(f"Ошибка при построении ROC для класса {class_name}: {e}")
                continue

        # Precision-Recall кривые для каждого класса
        plt.subplot(1, 2, 2)
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            try:
                precision, recall, _ = precision_recall_curve(labels == i, probabilities[:, i])
                avg_precision = average_precision_score(labels == i, probabilities[:, i])
                plt.plot(recall, precision, color=color, lw=2,
                         label=f'{class_name} (AP = {avg_precision:.4f})')
            except Exception as e:
                print(f"Ошибка при построении PR для класса {class_name}: {e}")
                continue

    # Настройка ROC subplot
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Настройка Precision-Recall subplot
    plt.subplot(1, 2, 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_confidence_intervals(metrics, labels, predictions, n_bootstrap=1000):
    """
    Расчет доверительных интервалов методом бутстрэп

    Args:
        metrics: словарь с метриками
        labels: истинные метки
        predictions: предсказания
        n_bootstrap: количество бутстрэп выборок
    """
    n_samples = len(labels)
    bootstrapped_metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }

    for _ in range(n_bootstrap):
        # Создаем бутстрэп выборку
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_labels = labels[indices]
        boot_preds = predictions[indices]

        # Вычисляем метрики для бутстрэп выборки
        bootstrapped_metrics['accuracy'].append(np.mean(boot_labels == boot_preds))

        if len(np.unique(labels)) == 2:
            cm = confusion_matrix(boot_labels, boot_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                bootstrapped_metrics['sensitivity'].append(sensitivity)
                bootstrapped_metrics['specificity'].append(specificity)

    # Вычисляем доверительные интервалы (95%)
    ci_metrics = {}
    for metric_name, values in bootstrapped_metrics.items():
        if values:
            ci_metrics[metric_name] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }

    return ci_metrics


def quick_evaluate(model_path, test_dataset, device='cpu', batch_size=1, class_names=None):
    """
    Быстрая оценка модели

    Args:
        model_path: путь к сохраненной модели
        test_dataset: тестовый датасет
        device: устройство
        batch_size: размер батча
        class_names: список имен классов
    """

    # Загрузка модели
    checkpoint = torch.load(model_path, map_location=device)

    # Определяем, что загружать - модель или state_dict
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'state_dict' in checkpoint:
        # Нужно создать модель соответствующей архитектуры
        # Это место для вашей кастомной логики создания модели
        model = None  # Замените на вашу модель
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint  # Предполагаем, что загружен сам model

    # Создание DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if class_names is None:
        class_names = ['Normal', 'Abnormal']

    # Оценка
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    return results


# Пример использования для Jupyter ноутбука
def demo_usage():
    """Демонстрация использования функций"""
    print("🔬 Модуль оценки модели загружен")
    print("Доступные функции:")
    print("  • evaluate_model() - полная оценка модели")
    print("  • plot_roc_curve() - построение ROC и PR кривых")
    print("  • quick_evaluate() - быстрая оценка")
    print("  • calculate_confidence_intervals() - доверительные интервалы")

    print("\n📝 Пример использования в Jupyter ноутбуке:")
    print("""
# 1. Импорт функций
from evaluate_model import evaluate_model, plot_roc_curve

# 2. Загрузка модели и данных
# model = ... ваша загруженная модель
# test_loader = ... ваш DataLoader

# 3. Запуск оценки
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    class_names=['Здоров', 'Болен'],
    save_results=True,
    output_dir='evaluation_results'
)

# 4. Построение графиков
plot_roc_curve(
    results['labels'],
    results['probabilities'],
    ['Здоров', 'Болен'],
    output_dir='evaluation_results'
)
    """)


# Условный импорт для примера (убрать проблемные относительные импорты)
if __name__ == "__main__":
    demo_usage()

    # Если хотите протестировать с реальными данными, раскомментируйте и адаптируйте:
    """
    try:
        # Пример с условным импортом
        from datasets.medical_tensors import MedicalTensorDataset

        # Ваш тестовый код здесь
        print("Тестирование с реальными данными...")

    except ImportError as e:
        print(f"Не удалось импортировать модули для тестирования: {e}")
        print("Это нормально - основные функции модуля работают корректно")
    """