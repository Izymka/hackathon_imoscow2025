import torch
import pandas as pd
import os
from torch.utils.data import Dataset


class MedicalTensorDataset(Dataset):
    def __init__(self, data_root, img_list, sets):
        """
        data_root: путь к папке с тензорами
        img_list: путь к CSV файлу с именами файлов и метками
        """
        self.data_root = data_root
        self.labels_df = pd.read_csv(img_list)  # CSV с колонками: filename, label
        self.sets = sets

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        filename = row['filename']
        label = int(row['label'])

        # Загружаем тензор
        tensor_path = os.path.join(self.data_root, filename)
        tensor = torch.load(tensor_path)

        # Твои тензоры имеют формат [1, 3, 128, 160, 160]
        # Но DataLoader добавит batch dimension, поэтому убираем существующий batch dimension
        if tensor.dim() == 5 and tensor.shape[0] == 1:
            # [1, 3, 128, 160, 160] -> [3, 128, 160, 160]
            tensor = tensor.squeeze(0)
        elif tensor.dim() != 4 or tensor.shape[0] != 3:
            raise ValueError(f"Ожидается тензор [3, 128, 160, 160], получено: {tensor.shape}")

        # Теперь DataLoader добавит batch dimension: [3, 128, 160, 160] -> [1, 3, 128, 160, 160]

        # Создаем метку
        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor