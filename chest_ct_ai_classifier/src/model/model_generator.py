import torch
from torch import nn

# Заменяем относительный импорт на абсолютный
try:
    from .models import resnet
except ImportError:
    # Если относительный импорт не работает, используем абсолютный
    from models import resnet


def adapt_model_for_input_size(model, input_size, model_depth, n_seg_classes):
    """
    Адаптирует модель для нового размера входа путем замены последнего слоя.
    """
    print(f"🔧 Адаптация модели для входа размером {input_size}...")

    # Заморозка всех параметров
    print("❄️ Замораживание всех параметров...")
    for param in model.parameters():
        param.requires_grad = False

    # Определение устройства модели
    device = next(model.parameters()).device
    print(f"🎯 Устройство модели: {device}")

    # Вычисление нового размера полносвязного слоя
    with torch.no_grad():
        # Создаем dummy_input на том же устройстве, что и модель
        dummy_input = torch.randn(1, 1, input_size[2], input_size[1], input_size[0]).to(device)

        # Извлекаем сверточную часть модели
        if hasattr(model, 'module'):
            # DataParallel случай
            conv_features = nn.Sequential(
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
            conv_features = nn.Sequential(
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

        # Перемещаем Sequential на то же устройство
        conv_features = conv_features.to(device)

        # Вычисляем размер после сверток
        conv_output = conv_features(dummy_input)
        flattened_size = conv_output.view(conv_output.size(0), -1).size(1)

    print(f"📊 Новый размер входа FC слоя: {flattened_size}")

    # Замена FC слоя
    if hasattr(model, 'module'):
        old_fc = model.module.fc
        model.module.fc = nn.Linear(flattened_size, n_seg_classes)
        new_fc = model.module.fc
        # Перемещаем новый FC слой на то же устройство
        model.module.fc = model.module.fc.to(device)
    else:
        old_fc = model.fc
        model.fc = nn.Linear(flattened_size, n_seg_classes)
        new_fc = model.fc
        # Перемещаем новый FC слой на то же устройство
        model.fc = model.fc.to(device)

    print(f"🔄 Заменен FC слой: {old_fc.in_features} → {flattened_size} входов, {n_seg_classes} выходов")

    # Инициализация нового слоя
    if isinstance(new_fc, nn.Linear):
        nn.init.xavier_uniform_(new_fc.weight)
        if new_fc.bias is not None:
            nn.init.zeros_(new_fc.bias)

    # Размораживание только FC слоя
    print("🔥 Размораживание FC слоя для обучения...")
    for param in new_fc.parameters():
        param.requires_grad = True

    # Размораживаем layer3 и layer4 (ВЫНЕСЕНО ИЗ ЦИКЛА!)
    print("🔥 Размораживание layer3 и layer4...")
    if hasattr(model, 'module'):
        for p in model.module.layer3.parameters():
            p.requires_grad = True
        for p in model.module.layer4.parameters():
            p.requires_grad = True
    else:
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True

    # Собираем все обучаемые параметры
    trainable_params = []
    base_params = []

    # FC слой - новые параметры
    trainable_params.extend(list(new_fc.parameters()))

    # layer3 и layer4 - базовые параметры
    if hasattr(model, 'module'):
        base_params.extend(list(model.module.layer3.parameters()))
        base_params.extend(list(model.module.layer4.parameters()))
    else:
        base_params.extend(list(model.layer3.parameters()))
        base_params.extend(list(model.layer4.parameters()))

    print(f"✅ Обучаемых параметров: {sum(p.numel() for p in trainable_params + base_params):,}")
    print(f"   - FC слой: {sum(p.numel() for p in trainable_params):,}")
    print(f"   - layer3+layer4: {sum(p.numel() for p in base_params):,}")

    # Возвращаем в том же формате, что и стандартный путь
    parameters = {
        'base_parameters': base_params,
        'new_parameters': trainable_params
    }

    return model, parameters


def generate_model(opt):
    assert opt.model in ['resnet', 'hybrid_resnet_transformer']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        # Создание модели с новыми размерами
        model_functions = {
            10: resnet.resnet10,
            18: resnet.resnet18,
            34: resnet.resnet34,
            50: resnet.resnet50,
            101: resnet.resnet101,
            152: resnet.resnet152,
            200: resnet.resnet200
        }

        model = model_functions[opt.model_depth](
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_seg_classes=opt.n_seg_classes,
            use_cbam=getattr(opt, 'use_cbam', False)
        )

    elif opt.model == 'hybrid_resnet_transformer':
        print(f"🤖 Создание гибридной ResNet-Transformer модели (depth={opt.model_depth})")

        # Создание гибридной модели
        model = resnet.HybridResNetTransformer(
            sample_input_D=opt.input_D,
            sample_input_H=opt.input_H,
            sample_input_W=opt.input_W,
            num_classes=opt.n_seg_classes,
            transformer_d_model=getattr(opt, 'transformer_d_model', 512),
            transformer_nhead=getattr(opt, 'transformer_nhead', 8),
            transformer_dim_feedforward=getattr(opt, 'transformer_dim_feedforward', 1024),
            transformer_num_layers=getattr(opt, 'transformer_num_layers', 2),
            transformer_dropout=getattr(opt, 'transformer_dropout', 0.1),
            use_cbam=getattr(opt, 'use_cbam', False)
        )

        print(f"✅ Гибридная модель создана:")
        print(f"   ResNet backbone: ResNet{opt.model_depth}")
        print(f"   CBAM attention: {getattr(opt, 'use_cbam', False)}")
        print(f"   Transformer layers: {getattr(opt, 'transformer_num_layers', 2)}")
        print(f"   Transformer heads: {getattr(opt, 'transformer_nhead', 8)}")

    # Настройка для GPU/CPU
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

        # Загрузка предобученной модели (только для ResNet, не для гибридной модели)
    if opt.phase != 'test' and opt.pretrain_path and opt.model == 'resnet':
        print('📥 Загрузка предобученной модели {}'.format(opt.pretrain_path))

        # Загрузка с совместимостью CPU/GPU
        if opt.no_cuda or not torch.cuda.is_available():
            pretrain = torch.load(opt.pretrain_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            pretrain = torch.load(opt.pretrain_path, weights_only=True)

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict, strict=False)

        # Проверка необходимости адаптации размеров
        current_input_size = (opt.input_W, opt.input_H, opt.input_D)
        pretrained_input_size = (128, 128, 128)  # размер предобученной модели

        if current_input_size != pretrained_input_size:
            print(f"⚠️ Обнаружено изменение размера входа: {pretrained_input_size} → {current_input_size}")
            print("🔧 Выполняется адаптация модели...")

            # Адаптируем модель
            model, parameters = adapt_model_for_input_size(
                model, current_input_size, opt.model_depth, opt.n_seg_classes
            )

            print("✅ Адаптация модели завершена!")
            return model, parameters

        # Стандартный путь без изменения размеров
        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    # Для гибридной модели возвращаем все параметры для обучения с нуля
    if opt.model == 'hybrid_resnet_transformer':
        return model, model.parameters()

    return model, model.parameters()