import click
import torch


@click.command()
@click.argument('input_ckpt', type=click.Path(exists=True))
@click.argument('output_pth', type=click.Path())
def convert_checkpoint(input_ckpt: str, output_pth: str):
    # Загружаем чекпоинт
    ckpt = torch.load(input_ckpt, map_location="cpu")

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
    torch.save(state_dict, output_pth)


if __name__ == '__main__':
    convert_checkpoint()
