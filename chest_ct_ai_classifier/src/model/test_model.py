# model/test_model.py
import torch
import os
from torch.utils.data import DataLoader
from evaluate_model import evaluate_model
from datasets.medical_tensors import MedicalTensorDataset
from setting import parse_opts
from model import generate_model


def main():
    # Настройки устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    # Загрузка последнего чекпоинта
    checkpoint_dirs = ['trails/models/resnet_10']  # измени при необходимости
    checkpoint_path = None

    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
            if checkpoints:
                # Берем последний чекпоинт
                checkpoints.sort()
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Найден чекпоинт: {checkpoint_path}")
                break

    if not checkpoint_path:
        print("Чекпоинт не найден!")
        return

    # Загрузка чекпоинта
    print("Загрузка модели...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Настройки модели
    sets = parse_opts()
    sets.model = 'resnet'
    sets.model_depth = 10  # измени при необходимости
    sets.n_seg_classes = 2  # количество классов
    sets.no_cuda = not torch.cuda.is_available()
    sets.input_D = 128
    sets.input_H = 160
    sets.input_W = 160

    # Создание модели
    print("Создание модели...")
    model, _ = generate_model(sets)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Создание тестового датасета
    print("Создание тестового датасета...")
    test_dataset = MedicalTensorDataset(
        data_root='../data/tensors',
        img_list='../data/test_labels.csv',  # убедись, что файл существует
        sets=sets
    )

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Оценка модели
    print("Оценка модели...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=['Здоров', 'Болен'],  # измени под свои классы
        save_results=True,
        output_dir='../evaluation_results'
    )

    print(f"\n{'=' * 50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'=' * 50}")
    print(f"Точность: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Loss: {results['loss']:.4f}")


if __name__ == "__main__":
    main()