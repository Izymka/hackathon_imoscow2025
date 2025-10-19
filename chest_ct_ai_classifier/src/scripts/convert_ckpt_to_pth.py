import os
import re
from pathlib import Path
from datetime import datetime

import click
import torch

def extract_epoch_and_f1(filename: str) -> tuple[int, float] | None:
    """Извлекает номер эпохи и значение val_f1 из имени файла."""
    match = re.search(r'epoch=(\d+).*val_f1=([\d.]+)', filename)
    if match:
        epoch = int(match.group(1))
        val_f1 = float(match.group(2))
        return epoch, val_f1
    return None


def get_file_modification_time(filepath: str) -> str:
    """Получает время модификации файла в формате YYYY-MM-DD_HH-MM."""
    mtime = os.path.getmtime(filepath)
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime('%Y-%m-%d_%H-%M')


def create_output_filename(epoch: int, val_f1: float, mod_time: str) -> str:
    """Создает новое имя файла в формате best-epoch=XX-val_f1=X.XXXX-YYYY-MM-DD_HH-MM.ckpt.pth."""
    return f"best-epoch={epoch:02d}-val_f1={val_f1:.4f}-{mod_time}.ckpt.pth"


def convert_single_checkpoint(input_path: str, output_path: str):
    """Конвертирует один чекпоинт."""
    # Загружаем чекпоинт
    ckpt = torch.load(input_path, map_location="cpu")

    # Если чекпоинт сделан PyTorch Lightning, веса лежат внутри key "state_dict"
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # иногда там уже только веса

    # Иногда Lightning добавляет префикс "model.", его можно убрать:
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    state_dict = {
        "state_dict": state_dict
    }
    # Сохраняем в .pth
    torch.save(state_dict, output_path)


@click.group()
def cli():
    """Утилита для конвертации чекпоинтов PyTorch Lightning в .pth формат."""
    pass


@cli.command()
@click.argument('input_ckpt', type=click.Path(exists=True))
@click.argument('output_pth', type=click.Path())
def convert_checkpoint(input_ckpt: str, output_pth: str):
    """Конвертирует один чекпоинт из .ckpt в .pth формат."""
    convert_single_checkpoint(input_ckpt, output_pth)
    click.echo(f"✓ Конвертация завершена: {output_pth}")


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True))
def convert_batch(input_dir: str, output_dir: str):
    """Массовая конвертация чекпоинтов из input_dir в output_dir.

    Извлекает из имени файла номер эпохи и val_f1, добавляет время модификации.
    Пропускает уже существующие файлы.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Создаем выходную директорию, если её нет
    output_path.mkdir(parents=True, exist_ok=True)

    # Находим все .ckpt файлы
    ckpt_files = list(input_path.glob("*.ckpt"))

    if not ckpt_files:
        click.echo(f"⚠ В директории {input_dir} не найдено .ckpt файлов")
        return

    click.echo(f"Найдено {len(ckpt_files)} файлов для конвертации")

    converted = 0
    skipped = 0
    errors = 0

    for ckpt_file in ckpt_files:
        try:
            # Извлекаем данные из имени файла
            result = extract_epoch_and_f1(ckpt_file.name)
            if result is None:
                click.echo(f"⚠ Пропускаем {ckpt_file.name}: не удалось извлечь epoch и val_f1")
                skipped += 1
                continue

            epoch, val_f1 = result

            # Получаем время модификации
            mod_time = get_file_modification_time(str(ckpt_file))

            # Формируем новое имя файла
            output_filename = create_output_filename(epoch, val_f1, mod_time)
            output_file = output_path / output_filename

            # Проверяем, существует ли уже целевой файл
            if output_file.exists():
                click.echo(f"⊘ Пропускаем {ckpt_file.name}: {output_filename} уже существует")
                skipped += 1
                continue

            # Конвертируем
            click.echo(f"→ Конвертация {ckpt_file.name} -> {output_filename}")
            convert_single_checkpoint(str(ckpt_file), str(output_file))
            converted += 1

        except Exception as e:
            click.echo(f"✗ Ошибка при обработке {ckpt_file.name}: {e}")
            errors += 1

    click.echo(f"\n{'='*60}")
    click.echo(f"Конвертировано: {converted}")
    click.echo(f"Пропущено: {skipped}")
    click.echo(f"Ошибок: {errors}")
    click.echo(f"{'='*60}")


if __name__ == '__main__':
    cli()
