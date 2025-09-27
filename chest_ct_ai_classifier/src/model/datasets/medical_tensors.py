# datasets/medical_tensors.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.data import MetaTensor


class MedicalTensorDataset(Dataset):
    def __init__(self, data_root, img_list, sets, transform=None):
        """
        data_root: путь к папке с тензорами (.pt файлами)
        img_list: путь к CSV файлу с колонками 'filename' и 'label'
        sets: объект конфигурации (опционально)
        transform: MONAI-трансформации, которые работают с torch.Tensor
        """
        self.data_root = data_root
        self.labels_df = pd.read_csv(img_list)
        self.sets = sets
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        # Получаем имя файла и метку
        row = self.labels_df.iloc[index]
        filename = row['filename']
        label = int(row['label'])

        # Полный путь к файлу тензора
        tensor_path = os.path.join(self.data_root, filename)

        # === Безопасная загрузка тензора с MetaTensor ===
        with torch.serialization.safe_globals([MetaTensor]):
            tensor = torch.load(tensor_path, weights_only=False)

        # === Конвертация MetaTensor в обычный torch.Tensor ===
        if isinstance(tensor, MetaTensor):
            tensor = tensor.as_tensor()

        # === Проверка и коррекция формы ===
        if tensor.dim() != 5 or tensor.shape != (1, 1, 256, 256, 256):
            raise ValueError(
                f"Неподдерживаемая форма тензора для индекса {index}: "
                f"получено {tensor.shape}, ожидается (1, 1, 128, 128, 128). "
                f"Проверьте файл: {tensor_path}"
            )

        # Убираем внешнюю размерность батча
        tensor = tensor.squeeze(0)  # [1, 128, 128, 128]

        # === Применяем аугментации, если заданы ===
        if self.transform:
            tensor = self.transform(tensor)

        # Создаем тензор метки
        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor