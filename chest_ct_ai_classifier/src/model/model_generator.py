import torch
from torch import nn
from models import resnet


def adapt_first_conv_layer_to_4ch(model, pretrained_state_dict=None):
    """
    Адаптирует первый свёрточный слой модели под 4-канальный вход.

    Args:
        model: исходная модель с 1-канальным conv1
        pretrained_state_dict: словарь весов MedicalNet (опционально)

    Returns:
        модель с conv1 на 4 канала
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
    new_conv1 = nn.Conv3d(
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


def adapt_model_for_input_size_and_channels(model, input_size, model_depth, n_seg_classes, n_input_channels=4):
    """
    Адаптирует модель под новый размер входа И количество каналов.
    """
    print(f"🔧 Адаптация модели для входа {n_input_channels}x{input_size}...")

    device = next(model.parameters()).device

    # === 1. Адаптация первого слоя под 4 канала ===
    if n_input_channels != 1:
        model = adapt_first_conv_layer_to_4ch(model)  # веса уже загружены ранее

    # === 2. Заморозка всех параметров ===
    for param in model.parameters():
        param.requires_grad = False

    # === 3. Вычисление размера FC слоя ===
    with torch.no_grad():
        dummy_input = torch.randn(1, n_input_channels, input_size[2], input_size[1], input_size[0]).to(device)

        # Собираем свёрточную часть
        if hasattr(model, 'module'):
            conv_seq = nn.Sequential(
                model.module.conv1,
                model.module.bn1,
                model.module.relu,
                model.module.maxpool,
                model.module.layer1,
                model.module.layer2,
                model.module.layer3,
                model.module.layer4,
                model.module.avgpool
            )
        else:
            conv_seq = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool
            )
        conv_seq = conv_seq.to(device)
        conv_out = conv_seq(dummy_input)
        flattened_size = conv_out.view(conv_out.size(0), -1).size(1)

    print(f"📊 Размер FC входа: {flattened_size}")

    # === 4. Замена FC слоя ===
    if hasattr(model, 'module'):
        old_fc = model.module.fc
        model.module.fc = nn.Linear(flattened_size, n_seg_classes).to(device)
        new_fc = model.module.fc
    else:
        old_fc = model.fc
        model.fc = nn.Linear(flattened_size, n_seg_classes).to(device)
        new_fc = model.fc

    # Перемещаем новый FC слой на то же устройство
    if hasattr(model, 'module'):
        model.module.fc = model.module.fc.to(device)
    else:
        model.fc = model.fc.to(device)

    print(f"🔄 Заменен FC слой: {old_fc.in_features} → {flattened_size} входов, {n_seg_classes} выходов")

    # Инициализация нового слоя
    if isinstance(new_fc, nn.Linear):
        nn.init.xavier_uniform_(new_fc.weight)
        if new_fc.bias is not None:
            nn.init.zeros_(new_fc.bias)

    print(f"🔄 FC слой: {old_fc.in_features} → {flattened_size} входов, {n_seg_classes} выходов")

    # === 5. Разморозка нужных слоёв ===
    layers_to_unfreeze = ['layer3', 'layer4', 'fc']
    print("🔥 Размораживание слоёв:", layers_to_unfreeze)

    def unfreeze_module(module):
        for name, child in module.named_children():
            if name in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = True
            elif name == 'fc':
                for param in child.parameters():
                    param.requires_grad = True

    if hasattr(model, 'module'):
        unfreeze_module(model.module)
    else:
        unfreeze_module(model)

    # Собираем обучаемые параметры
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"📈 Обучаемых параметров: {sum(p.numel() for p in trainable_params)}")

    return model, trainable_params


def generate_model(opt):
    """
    Генерирует модель с поддержкой 4-канального входа и transfer learning от MedicalNet.
    """
    assert opt.model in ['resnet']
    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

    # === 1. Создание базовой модели ===
    model_functions = {
        10: resnet.resnet10,
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        200: resnet.resnet200
    }

    # Создаём модель с 1 каналом (MedicalNet стандарт)
    model = model_functions[opt.model_depth](
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes
    )

    # === 2. Настройка устройства ===
    if not opt.no_cuda and torch.cuda.is_available():
        model = model.cuda()
        if len(opt.gpu_id) > 1:
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
        else:
            # Используем один GPU
            torch.cuda.set_device(opt.gpu_id[0])
    # Иначе остаётся на CPU

    # === 3. Загрузка предобученных весов ===
    pretrained_state_dict = None
    if opt.phase != 'test' and opt.pretrain_path:
        print(f'📥 Загрузка предобученной модели: {opt.pretrain_path}')
        map_location = torch.device('cpu') if opt.no_cuda or not torch.cuda.is_available() else None
        checkpoint = torch.load(opt.pretrain_path, map_location=map_location, weights_only=True)

        # MedicalNet сохраняет веса в 'state_dict'
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint  # на случай если веса сохранены напрямую

    # === 4. Адаптация под 4 канала и размер входа ===
    current_input_size = (opt.input_W, opt.input_H, opt.input_D)
    n_input_channels = getattr(opt, 'n_input_channels', 4)  # по умолчанию 4

    # Всегда адаптируем, так как мы используем 4 канала
    model, trainable_params = adapt_model_for_input_size_and_channels(
        model=model,
        input_size=current_input_size,
        model_depth=opt.model_depth,
        n_seg_classes=opt.n_seg_classes,
        n_input_channels=n_input_channels
    )

    return model, trainable_params