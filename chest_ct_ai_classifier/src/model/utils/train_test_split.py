import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


def split_dataset(data_dir='data', train_ratio=0.7, random_state=654321):
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

    # Проверяем существование исходной директории
    if not os.path.exists(tensors_dir):
        raise FileNotFoundError(f"Source directory {tensors_dir} does not exist")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Читаем метки
    labels_path = Path(data_dir) / "processed" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file {labels_path} not found")

    df = pd.read_csv(labels_path)
    print(f"Found columns: {df.columns.tolist()}")
    print(f"Total samples: {len(df)}")

    # Проверяем наличие необходимых колонок
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'label' columns")

    # Преобразуем label в int для корректной стратификации
    df['label'] = df['label'].astype(int)

    # Проверяем, что все файлы существуют
    missing_files = []
    for filename in df['filename']:
        if not os.path.exists(os.path.join(tensors_dir, filename)):
            missing_files.append(filename)

    if missing_files:
        print(f"Warning: {len(missing_files)} files not found:")
        for f in missing_files[:5]:  # Показываем только первые 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")

        # Удаляем отсутствующие файлы из датафрейма
        df = df[~df['filename'].isin(missing_files)]
        print(f"Continuing with {len(df)} valid samples")

    # Разделение с стратификацией
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=df['label']
        )
    except ValueError as e:
        print(f"Stratification failed: {e}")
        print("Falling back to random split without stratification")
        train_df, test_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            random_state=random_state
        )

    # Функция для безопасного копирования файлов
    def copy_files_safely(df, target_dir, split_name):
        copied = 0
        failed = 0

        for _, row in df.iterrows():
            src_path = os.path.join(tensors_dir, row['filename'])
            dst_path = os.path.join(target_dir, row['filename'])

            try:
                if os.path.exists(src_path):
                    # Создаем поддиректории если нужно
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                else:
                    print(f"Warning: File {src_path} not found")
                    failed += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                failed += 1

        print(f"{split_name}: copied {copied} files, failed {failed}")
        return copied, failed

    # Копируем файлы
    print("\nCopying files...")
    train_copied, train_failed = copy_files_safely(train_df, train_dir, "Train")
    test_copied, test_failed = copy_files_safely(test_df, test_dir, "Test")

    # Удаляем записи о файлах, которые не удалось скопировать
    if train_failed > 0:
        # Проверяем какие файлы действительно скопированы
        train_df = train_df[train_df['filename'].apply(
            lambda x: os.path.exists(os.path.join(train_dir, x))
        )]

    if test_failed > 0:
        test_df = test_df[test_df['filename'].apply(
            lambda x: os.path.exists(os.path.join(test_dir, x))
        )]

    # Сохраняем метки для тренировочной выборки
    train_labels_path = os.path.join(train_dir, 'labels.csv')
    train_df.to_csv(train_labels_path, index=False)

    # Сохраняем метки для тестовой выборки
    test_labels_path = os.path.join(test_dir, 'labels.csv')
    test_df.to_csv(test_labels_path, index=False)

    # Выводим информацию о разделении
    print(f"\n{'=' * 50}")
    print(f"Dataset split completed!")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train ratio: {len(train_df) / (len(train_df) + len(test_df)):.2%}")

    print(f"\nTrain class distribution:")
    train_dist = train_df['label'].value_counts().sort_index()
    for label, count in train_dist.items():
        print(f"  Class {label}: {count} ({count / len(train_df):.1%})")

    print(f"\nTest class distribution:")
    test_dist = test_df['label'].value_counts().sort_index()
    for label, count in test_dist.items():
        print(f"  Class {label}: {count} ({count / len(test_df):.1%})")

    # Проверяем сохранность пропорций
    print(f"\nClass balance check:")
    for label in sorted(df['label'].unique()):
        orig_ratio = (df['label'] == label).mean()
        train_ratio_actual = (train_df['label'] == label).mean()
        test_ratio_actual = (test_df['label'] == label).mean()
        print(
            f"  Class {label}: Original {orig_ratio:.1%}, Train {train_ratio_actual:.1%}, Test {test_ratio_actual:.1%}")

    return train_df, test_df


# Использование функции
if __name__ == "__main__":
    try:
        train_df, test_df = split_dataset()
        print("\n✅ Dataset split completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise