import torch
from torch import nn

# Заменяем относительный импорт на абсолютный
try:
    from .models import resnet
except ImportError:
    # Если относительный импорт не работает, используем абсолютный
    from models import resnet

def adapt_pretrained_weights_for_hybrid(pretrain_dict, model_type='hybrid', use_dataparallel=True):
    """
    Адаптирует ключи предобученных весов ResNet для гибридной модели.
    Добавляет префикс 'module.backbone.' к ключам ResNet компонентов.
    """
    if model_type != 'hybrid':
        return pretrain_dict

    # Показываем примеры ключей для отладки
    sample_keys = list(pretrain_dict.keys())[:5]
    print(f"  Примеры ключей в предобученной модели: {sample_keys}")

    adapted_dict = {}
    resnet_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    # Ключи которые НЕ нужно переносить (специфичны для классификации/сегментации)
    skip_layers = ['fc', 'conv_seg']

    converted_count = 0
    for key, value in pretrain_dict.items():
        # Пропускаем ключи, которые не нужны
        if any(key.startswith(skip_key) for skip_key in skip_layers):
            continue

        # Проверяем, относится ли ключ к ResNet компонентам
        # Проверяем как точное совпадение, так и начало строки
        is_resnet_layer = False
        for layer in resnet_layers:
            if key == layer or key.startswith(layer + '.') or key.startswith(layer + '_'):
                is_resnet_layer = True
                break

        if is_resnet_layer:
            # Добавляем префикс module.backbone для DataParallel
            if use_dataparallel:
                new_key = f'module.backbone.{key}'
            else:
                new_key = f'backbone.{key}'
            adapted_dict[new_key] = value
            converted_count += 1
            if converted_count <= 5:  # Показываем только первые 5
                print(f"  {key} → {new_key}")

    if converted_count > 5:
        print(f"  ... и еще {converted_count - 5} слоев")
    print(f"  Всего адаптировано: {converted_count} слоев")

    # Если ничего не адаптировано, показываем все уникальные префиксы
    if converted_count == 0:
        prefixes = set()
        for key in pretrain_dict.keys():
            prefix = key.split('.')[0] if '.' in key else key
            prefixes.add(prefix)
        print(f"  ⚠️ Найденные префиксы в чекпоинте: {sorted(prefixes)}")

    return adapted_dict


def freeze_backbone_blocks(model, blocks_to_freeze=['conv1', 'bn1', 'layer1', 'layer2']):
    """
    Замораживает указанные блоки в backbone гибридной модели.
    По умолчанию замораживает первые 2 блока ResNet (conv1, bn1, layer1, layer2).
    Работает с DataParallel обертками.
    """
    # Проверяем, обернута ли модель в DataParallel
    actual_model = model.module if hasattr(model, 'module') else model

    if not hasattr(actual_model, 'backbone'):
        print("⚠️ Модель не имеет атрибута backbone, заморозка пропущена")
        return

    frozen_params = 0
    for block_name in blocks_to_freeze:
        if hasattr(actual_model.backbone, block_name):
            block = getattr(actual_model.backbone, block_name)
            for param in block.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            print(f"  ❄️ {block_name} заморожен")
        else:
            print(f"  ⚠️ {block_name} не найден в backbone")

    print(f"  ❄️ Всего заморожено параметров: {frozen_params:,}")



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

    # Загрузка предобученной модели
    if opt.phase != 'test' and opt.pretrain_path and opt.model in ['resnet', 'hybrid_resnet_transformer']:
        print('📥 Загрузка предобученной модели {}'.format(opt.pretrain_path))

        # Загрузка с совместимостью CPU/GPU
        if opt.no_cuda or not torch.cuda.is_available():
            pretrain = torch.load(opt.pretrain_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            pretrain = torch.load(opt.pretrain_path, weights_only=True)

        # Извлекаем state_dict из чекпоинта
        if 'state_dict' in pretrain:
            pretrain_state = pretrain['state_dict']
        else:
            pretrain_state = pretrain

        print(f"  Всего ключей в чекпоинте: {len(pretrain_state)}")

        # Удаляем все возможные префиксы (от Lightning и DataParallel)
        pretrain_dict = {}
        for k, v in pretrain_state.items():
            clean_key = k

            # Убираем префиксы в порядке: model.module. -> model. -> module.
            if clean_key.startswith('model.module.'):
                clean_key = clean_key[13:]  # убираем 'model.module.'
            elif clean_key.startswith('model.'):
                clean_key = clean_key[6:]  # убираем 'model.'
            elif clean_key.startswith('module.'):
                clean_key = clean_key[7:]  # убираем 'module.'

            # Пропускаем не-модельные ключи
            if clean_key in ['class_weights', 'loss_fn'] or clean_key.startswith('loss_fn.'):
                continue

            pretrain_dict[clean_key] = v

        print(f"  Ключей после очистки префиксов: {len(pretrain_dict)}")
        # Показываем примеры очищенных ключей
        sample_clean_keys = list(pretrain_dict.keys())[:5]
        print(f"  Примеры очищенных ключей: {sample_clean_keys}")

        # Для гибридной модели адаптируем ключи
        if opt.model == 'hybrid_resnet_transformer':
            print("🔄 Адаптация ключей предобученных весов для гибридной модели:")
            # Определяем, используется ли DataParallel
            use_dp = len(opt.gpu_id) > 1 or (hasattr(model, 'module'))
            pretrain_dict = adapt_pretrained_weights_for_hybrid(
                pretrain_dict, 
                model_type='hybrid',
                use_dataparallel=use_dp
            )

        # Загружаем веса (strict=False чтобы игнорировать несовпадающие слои)
        # Фильтруем только совместимые ключи
        compatible_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict.keys()}

        incompatible_keys = set(pretrain_dict.keys()) - set(compatible_dict.keys())
        if incompatible_keys:
            print(f"⚠️ Несовместимые ключи (будут пропущены): {len(incompatible_keys)}")
            # Показываем несколько примеров для отладки
            examples = list(incompatible_keys)[:3]
            if examples:
                print(f"   Примеры: {examples}")

        print(f"✅ Загружено предобученных весов: {len(compatible_dict)}/{len(net_dict)} слоев")

        if len(compatible_dict) > 0:
            net_dict.update(compatible_dict)
            model.load_state_dict(net_dict, strict=False)
        else:
            print("⚠️ ВНИМАНИЕ: Ни один слой не был загружен из предобученной модели!")

        # Для гибридной модели замораживаем первые 2 блока ResNet
        if opt.model == 'hybrid_resnet_transformer':
            print("❄️ Заморозка первых 2 блоков ResNet в backbone:")
            freeze_backbone_blocks(model, blocks_to_freeze=['conv1', 'bn1', 'layer1', 'layer2'])

        # Проверка необходимости адаптации размеров
        current_input_size = (opt.input_W, opt.input_H, opt.input_D)
        pretrained_input_size = (128, 128, 128)  # размер предобученной модели

        if current_input_size != pretrained_input_size and opt.model == 'resnet':
            print(f"⚠️ Обнаружено изменение размера входа: {pretrained_input_size} → {current_input_size}")
            print("🔧 Выполняется адаптация модели...")

            # Адаптируем модель
            model, parameters = adapt_model_for_input_size(
                model, current_input_size, opt.model_depth, opt.n_seg_classes
            )

            print("✅ Адаптация модели завершена!")
            return model, parameters

        # Формирование обучаемых параметров
        if opt.model == 'hybrid_resnet_transformer':
            # Для гибридной модели обучаем:
            # - layer3, layer4 в backbone
            # - все Transformer слои
            # - классификатор

            trainable_params = []

            # Backbone layer3 и layer4
            if hasattr(model, 'module'):
                # DataParallel случай
                trainable_params.extend(list(model.module.backbone.layer3.parameters()))
                trainable_params.extend(list(model.module.backbone.layer4.parameters()))
                # Transformer layers
                trainable_params.extend(list(model.module.transformer_layers.parameters()))
                # Positional encoding
                trainable_params.append(model.module.pos_encoding)
                # Classifier
                trainable_params.extend(list(model.module.classifier.parameters()))
            else:
                trainable_params.extend(list(model.backbone.layer3.parameters()))
                trainable_params.extend(list(model.backbone.layer4.parameters()))
                trainable_params.extend(list(model.transformer_layers.parameters()))
                trainable_params.append(model.pos_encoding)
                trainable_params.extend(list(model.classifier.parameters()))

            # Убеждаемся что все в trainable_params имеют requires_grad=True
            for p in trainable_params:
                p.requires_grad = True

            print(f"✅ Обучаемых параметров в гибридной модели: {sum(p.numel() for p in trainable_params):,}")

            parameters = {
                'base_parameters': trainable_params,  # используем одинаковый LR для всех
                'new_parameters': []
            }

            return model, parameters
        else:
            # Стандартный путь для обычного ResNet
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

    return model, model.parameters()