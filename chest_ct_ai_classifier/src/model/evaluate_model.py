import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from monai.metrics import compute_roc_auc, compute_confusion_matrix
import os
from tqdm import tqdm


def evaluate_model(model, test_loader, device, class_names=None, save_results=True, output_dir='results'):
    """
    Оценка качества модели с подробными метриками

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

            # Если нужно сохранить имена файлов (для отладки)
            # all_filenames.extend([f"batch_{batch_idx}_sample_{i}" for i in range(len(targets))])

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

    print(f"\n{'=' * 50}")
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print(f"{'=' * 50}")
    print(f"Средний Loss: {avg_loss:.4f}")
    print(f"Точность (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Подробный классификационный отчет
    print(f"\nПодробный отчет по классам:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nМатрица ошибок:")
    print(cm)

    # AUC-ROC (для бинарной классификации)
    if len(class_names) == 2:
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        print(f"\nAUC-ROC: {auc_score:.4f}")
    else:
        # Для многоклассовой классификации
        try:
            auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            print(f"\nAUC-ROC (multiclass): {auc_score:.4f}")
        except:
            print("\nAUC-ROC: не удалось вычислить для многоклассовой задачи")

    # Сохранение результатов
    if save_results:
        save_evaluation_results(
            all_labels, all_predictions, all_probabilities,
            cm, accuracy, avg_loss, class_names, output_dir
        )

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm
    }


def save_evaluation_results(labels, predictions, probabilities, cm, accuracy, loss, class_names, output_dir):
    """Сохранение результатов оценки"""

    # Сохранение метрик в текстовый файл
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Точность (Accuracy): {accuracy:.4f}\n")
        f.write(f"Средний Loss: {loss:.4f}\n\n")

        f.write("Классификационный отчет:\n")
        f.write(classification_report(labels, predictions, target_names=class_names))

    # Сохранение матрицы ошибок как изображение
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
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

    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"\nРезультаты сохранены в директорию: {output_dir}")


def plot_roc_curve(labels, probabilities, class_names, output_dir='results'):
    """Построение ROC-кривой"""

    plt.figure(figsize=(8, 6))

    if len(class_names) == 2:
        # Бинарная классификация
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        auc_score = roc_auc_score(labels, probabilities[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    else:
        # Многоклассовая классификация
        for i, class_name in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(labels == i, probabilities[:, i])
                auc_score = roc_auc_score(labels == i, probabilities[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.4f})')
            except:
                continue

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Функция для быстрой оценки
def quick_evaluate(model_path, test_dataset, device='cpu', batch_size=1):
    """
    Быстрая оценка модели

    Args:
        model_path: путь к сохраненной модели
        test_dataset: тестовый датасет
        device: устройство
        batch_size: размер батча
    """

    # Загрузка модели
    checkpoint = torch.load(model_path, map_location=device)

    # Создание DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Оценка
    results = evaluate_model(
        model=checkpoint['model'] if 'model' in checkpoint else checkpoint,
        test_loader=test_loader,
        device=device,
        class_names=['Normal', 'Abnormal']  # Измени под свои классы
    )

    return results


# Пример использования
if __name__ == "__main__":
    # Пример использования функции
    """
    # Загрузка модели
    model = ... # твоя модель
    model.load_state_dict(torch.load('path/to/checkpoint.pth')['state_dict'])

    # Создание тестового датасета
    test_dataset = MedicalTensorDataset(
        data_root='../data/tensors',
        img_list='../data/test_labels.csv',  # отдельный файл для теста
        sets=None  # или нужные параметры
    )

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Оценка
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=['Здоров', 'Болен']  # измени под свои классы
    )
    """
    pass