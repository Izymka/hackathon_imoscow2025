
"""
inference_3d_4chanel.py

Скрипт для инференса 3D ResNet модели на 4-канальных тензорах 128x128x128.
Использует веса из checkpoint файла и подготовленные тензоры.
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import sys


sys.path.append('model/outputs/weights')  # путь к модели
from models import resnet


def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """Dataset для инференса - загружает подготовленные .pt файлы"""

    def __init__(self, tensor_paths):
        self.tensor_paths = tensor_paths

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        tensor = torch.load(tensor_path, weights_only=True)

        # Убедимся, что тензор имеет правильную форму [1, 4, 128, 128, 128]
        if tensor.shape != (1, 4, 128, 128, 128):
            raise ValueError(f"Неправильная форма тензора: {tensor.shape}, ожидается (1, 4, 128, 128, 128)")

        return tensor, str(tensor_path)


def adapt_first_conv_layer_to_4ch(model, pretrained_state_dict=None):
    """
    Адаптирует первый свёрточный слой модели под 4-канальный вход.
    """
    device = next(model.parameters()).device

    # Загружаем предобученные веса, если есть
    if pretrained_state_dict is not None:
        # Убираем 'module.' если нужно
        cleaned_state_dict = {}
        for k, v in pretrained_state_dict.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[new_k] = v
        # Загружаем с strict=False, так как conv1 не совпадает
        model.load_state_dict(cleaned_state_dict, strict=False)

    # Получаем оригинальный вес conv1 (должен быть [C_out, 1, K, K, K])
    original_weight = model.conv1.weight.data  # [64, 1, 7, 7, 7] для ResNet-18/34
    assert original_weight.shape[1] == 1, f"Ожидался 1 входной канал, получено {original_weight.shape[1]}"

    # Создаём новый вес: повторяем и делим на 4 для сохранения масштаба
    new_weight = original_weight.repeat(1, 4, 1, 1, 1) / 4.0  # [64, 4, 7, 7, 7]

    # Создаём новый conv1 слой
    new_conv1 = torch.nn.Conv3d(
        in_channels=4,
        out_channels=original_weight.shape[0],
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=model.conv1.bias is not None
    )
    new_conv1.weight.data = new_weight
    if new_conv1.bias is not None:
        new_conv1.bias.data = model.conv1.bias.data.clone()

    # Заменяем в модели
    model.conv1 = new_conv1.to(device)
    print(f"✅ Заменён conv1: {original_weight.shape} → {new_weight.shape}")
    return model


def create_model(model_depth, n_classes, checkpoint_path=None):
    """Создает модель и загружает веса из checkpoint"""

    # Маппинг глубин модели
    model_functions = {
        10: resnet.resnet10,
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        200: resnet.resnet200
    }

    if model_depth not in model_functions:
        raise ValueError(f"Model depth {model_depth} not supported. Available: {list(model_functions.keys())}")

    # Создаем модель с 1 каналом (как в MedicalNet)
    model = model_functions[model_depth](
        sample_input_W=128,
        sample_input_H=128,
        sample_input_D=128,
        shortcut_type='A',  # или 'B' в зависимости от конфигурации
        no_cuda=False,
        num_seg_classes=2
    )

    # Загружаем checkpoint
    if checkpoint_path:
        print(f'📥 Загрузка checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Проверяем структуру checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # предполагаем, что веса напрямую

        # Загружаем веса
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Загружено {len(state_dict)} параметров из checkpoint")

    # Адаптируем первый слой под 4 канала
    model = adapt_first_conv_layer_to_4ch(model)

    return model


def run_inference(model, dataloader, device, output_dir, save_probabilities=True, save_logits=True):
    """Запускает инференс для всей выборки"""

    model.eval()
    results = []

    print(f"🚀 Запуск инференса на {len(dataloader.dataset)} образцах...")

    with torch.no_grad():
        for batch_idx, (batch_tensors, tensor_paths) in enumerate(tqdm(dataloader, desc="Инференс")):
            batch_tensors = batch_tensors.to(device)

            # Прямой проход
            outputs = model(batch_tensors)

            # Применяем softmax для получения вероятностей (если задача классификации)
            probabilities = F.softmax(outputs, dim=1)

            # Сохраняем результаты для каждого образца в батче
            for i in range(len(tensor_paths)):
                tensor_path = Path(tensor_paths[i])
                patient_name = tensor_path.stem

                result = {
                    'patient_name': patient_name,
                    'tensor_path': str(tensor_path),
                    'logits': outputs[i].cpu().numpy(),
                    'probabilities': probabilities[i].cpu().numpy()
                }

                # Сохраняем результаты
                if save_logits:
                    logits_path = output_dir / f"{patient_name}_logits.pt"
                    torch.save(outputs[i].cpu(), logits_path)
                    result['logits_path'] = str(logits_path)

                if save_probabilities:
                    probs_path = output_dir / f"{patient_name}_probs.pt"
                    torch.save(probabilities[i].cpu(), probs_path)
                    result['probs_path'] = str(probs_path)

                # Добавляем предсказанный класс
                predicted_class = torch.argmax(probabilities[i]).item()
                result['predicted_class'] = predicted_class
                result['confidence'] = probabilities[i][predicted_class].item()

                # Добавляем все вероятности
                result['all_probabilities'] = probabilities[i].cpu().numpy().tolist()

                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Инференс 3D ResNet модели")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к checkpoint файлу (.pth, .pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Входная директория с .pt тензорами или путь к одному тензору")
    parser.add_argument("--output", type=str, required=True,
                        help="Директория для сохранения результатов")
    parser.add_argument("--model-depth", type=int, default=18,
                        choices=[10, 18, 34, 50, 101, 152, 200],
                        help="Глубина ResNet модели")
    parser.add_argument("--num-classes", type=int, required=True,
                        help="Количество классов для классификации")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Размер батча (обычно 1 для 3D данных)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Устройство для инференса")
    parser.add_argument("--no-probabilities", action="store_true",
                        help="Не сохранять вероятности")
    parser.add_argument("--no-logits", action="store_true",
                        help="Не сохранять логиты")
    parser.add_argument("--verbose", action="store_true",
                        help="Подробный вывод")
    parser.add_argument("--log-file", type=str, default="logs/inference.log",
                        help="Файл лога")

    args = parser.parse_args()

    # Настройка логирования
    logger = setup_logging(Path(args.log_file))

    # Определение устройства
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ Используется GPU")
    else:
        device = torch.device("cpu")
        logger.info("✅ Используется CPU")

    # Создание выходной директории
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Поиск тензоров
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix == '.pt':
        tensor_paths = [input_path]
    elif input_path.is_dir():
        tensor_paths = list(input_path.glob("*.pt"))
        if not tensor_paths:
            raise ValueError(f"Не найдено .pt файлов в {input_path}")
    else:
        raise ValueError(f"Входной путь не существует или не является .pt файлом или директорией: {input_path}")

    logger.info(f"Найдено {len(tensor_paths)} тензоров для инференса")

    # Создание датасета и даталоадера
    dataset = InferenceDataset(tensor_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Создание и загрузка модели
    logger.info(f"Создание модели ResNet-{args.model_depth} с {args.num_classes} классами")
    model = create_model(args.model_depth, args.num_classes, args.checkpoint)
    model = model.to(device)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Параметров: {total_params:,}, обучаемых: {trainable_params:,}")

    # Запуск инференса
    results = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        save_probabilities=not args.no_probabilities,
        save_logits=not args.no_logits
    )

    # Сохранение общего отчета
    report = {
        'config': {
            'checkpoint': args.checkpoint,
            'input_dir': str(input_path),
            'output_dir': str(output_dir),
            'model_depth': args.model_depth,
            'num_classes': args.num_classes,
            'batch_size': args.batch_size,
            'device': str(device),
            'total_samples': len(results)
        },
        'results': results
    }

    report_path = output_dir / "inference_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Подсчет статистики по классам
    class_counts = {}
    total_confidence = 0
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        total_confidence += result['confidence']

    avg_confidence = total_confidence / len(results) if results else 0

    logger.info(f"\n{'=' * 60}")
    logger.info("ИНФЕРЕНС ЗАВЕРШЕН")
    logger.info(f"Обработано образцов: {len(results)}")
    logger.info(f"Средняя уверенность: {avg_confidence:.4f}")
    logger.info("Распределение по классам:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        logger.info(f"  Класс {class_id}: {count} ({100 * count / len(results):.1f}%)")
    logger.info(f"Результаты сохранены в: {output_dir}")
    logger.info(f"Отчет: {report_path}")
    logger.info(f"{'=' * 60}")

    # Сохранение краткого отчета
    summary_path = output_dir / "inference_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Инференс завершен: {len(results)} образцов\n")
        f.write(f"Средняя уверенность: {avg_confidence:.4f}\n")
        f.write("Распределение по классам:\n")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            f.write(f"  Класс {class_id}: {count} ({100 * count / len(results):.1f}%)\n")

    logger.info(f"Краткий отчет сохранен: {summary_path}")
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()