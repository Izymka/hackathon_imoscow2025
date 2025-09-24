import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


def split_dataset(data_dir='data', train_ratio=0.7, random_state=42):
    """
    Разделяет датасет на тренировочную и тестовую выборки с сохранением структуры

    Args:
        data_dir: путь к директории с данными
        train_ratio: доля тренировочных данных (0.7 = 70%)
        random_state: seed для воспроизводимости
    """

    # Создаем необходимые директории
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    tensors_dir = os.path.join(data_dir, 'processed')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    

    # Читаем метки
    labels_path = Path(data_dir) / "processed" / "labels.csv"
    df = pd.read_csv(labels_path)
    print(df.columns)
    # Преобразуем label в int для корректной стратификации
    df['label'] = df['label'].astype(int)

    # Разделение с стратификацией
    train_df, test_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=random_state,
        stratify=df['label']
    )

    # Копируем тренировочные файлы
    for _, row in train_df.iterrows():
        src_path = os.path.join(tensors_dir, row['filename'])
        dst_path = os.path.join(train_dir, row['filename'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: File {src_path} not found")

    # Копируем тестовые файлы
    for _, row in test_df.iterrows():
        src_path = os.path.join(tensors_dir, row['filename'])
        dst_path = os.path.join(test_dir, row['filename'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: File {src_path} not found")

    # Сохраняем метки для тренировочной выборки
    train_labels_path = os.path.join(train_dir, 'labels.csv')
    train_df.to_csv(train_labels_path, index=False)

    # Сохраняем метки для тестовой выборки
    test_labels_path = os.path.join(test_dir, 'labels.csv')
    test_df.to_csv(test_labels_path, index=False)

    # Выводим информацию о разделении
    print(f"Dataset split completed!")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class distribution:")
    print(train_df['label'].value_counts().sort_index())
    print(f"Test class distribution:")
    print(test_df['label'].value_counts().sort_index())

    return train_df, test_df


# Использование функции
if __name__ == "__main__":
    train_df, test_df = split_dataset()