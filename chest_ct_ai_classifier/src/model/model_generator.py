import torch
from torch import nn
from models import resnet

def adapt_model_for_larger_input(model, opt):
    """
    Адаптирует модель для работы с входным тензором большего размера.
    Изменяет архитектуру начальных слоев для корректной обработки больших входов.
    """
    print(f"Адаптация модели с входа {opt.pretrain_input_size} на вход {opt.input_W}×{opt.input_H}×{opt.input_D}")
    
    # Получаем доступ к модели (убираем DataParallel если есть)
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    
    # Для ResNet 3D нужно модифицировать начальные слои
    # Добавляем дополнительный пулинг или изменяем страйды
    
    original_conv1 = base_model.conv1
    original_maxpool = base_model.maxpool
    
    # Если размер увеличился в 2 раза (128->256), добавляем дополнительный пулинг
    scale_factor = opt.input_W // 128  # предполагаем, что претрейн был на 128
    
    if scale_factor == 2:
        # Добавляем дополнительный пулинг после первой свертки
        base_model.additional_pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Переопределяем forward (создаем новый forward метод)
        original_forward = base_model.forward
        
        def new_forward(x):
            x = base_model.conv1(x)
            x = base_model.bn1(x)
            x = base_model.relu(x)
            x = base_model.additional_pool(x)  # Дополнительный пулинг
            x = base_model.maxpool(x)
            x = base_model.layer1(x)
            x = base_model.layer2(x)
            x = base_model.layer3(x)
            x = base_model.layer4(x)
            x = base_model.avgpool(x)
            x = torch.flatten(x, 1)
            x = base_model.fc(x)
            return x
        
        base_model.forward = new_forward
        print("Добавлен дополнительный пулинг слой для адаптации к размеру 256×256×256")
    
    elif scale_factor > 2:
        # Для больших масштабов изменяем страйд первой свертки
        new_conv1 = nn.Conv3d(
            1,  # входные каналы
            64, # выходные каналы
            kernel_size=7,
            stride=(2 * scale_factor // 2, 2 * scale_factor // 2, 2 * scale_factor // 2),
            padding=(3, 3, 3),
            bias=False
        )
        
        # Копируем веса из оригинальной свертки
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.clone()
        
        base_model.conv1 = new_conv1
        print(f"Изменен страйд первой свертки для адаптации к большему размеру входа")
    
    return model

def generate_model_with_transfer_learning(opt):
    """
    Создает модель с трансферным обучением для адаптации размера входа.
    """
    print("=== Создание модели с трансферным обучением ===")
    
    # Сначала создаем модель с новым размером входа
    model = generate_base_model(opt)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
    
    # Загружаем предобученные веса
    if opt.phase != 'test' and opt.pretrain_path:
        print(f'Загружаем предобученную модель {opt.pretrain_path}')
        
        # Загружаем checkpoint
        if opt.no_cuda or not torch.cuda.is_available():
            pretrain = torch.load(opt.pretrain_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            pretrain = torch.load(opt.pretrain_path, weights_only=True)
        
        pretrain_dict = pretrain['state_dict']
        
        # Получаем словарь весов текущей модели
        net_dict = model.state_dict()
        
        # Фильтруем только совместимые веса (исключая fc и проблемные слои)
        filtered_dict = {}
        for k, v in pretrain_dict.items():
            if k in net_dict.keys():
                # Проверяем совместимость размеров
                if net_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                    print(f"✓ Загружен слой: {k}")
                else:
                    print(f"✗ Пропущен слой {k}: несоответствие размеров {net_dict[k].shape} != {v.shape}")
            else:
                print(f"✗ Слой {k} не найден в новой модели")
        
        # Обновляем веса модели
        net_dict.update(filtered_dict)
        model.load_state_dict(net_dict)
        
        # Адаптируем модель для нового размера входа, если необходимо
        if hasattr(opt, 'pretrain_input_size') and opt.pretrain_input_size != opt.input_W:
            model = adapt_model_for_larger_input(model, opt)
        
        # --- Заморозка параметров ---
        print("Замораживаем все параметры...")
        for param in model.parameters():
            param.requires_grad = False

        # --- Замена/настройка последнего слоя (fc) ---
        expansion = 1 if opt.model_depth in [10, 18, 34] else 4
        in_features = 512 * expansion
        
        # Заменяем fc слой
        old_fc = model.module.fc if hasattr(model, 'module') else model.fc
        new_num_classes = opt.n_seg_classes
        
        if hasattr(model, 'module'):
            model.module.fc = nn.Linear(in_features, new_num_classes)
            new_fc_params = model.module.fc.parameters()
        else:
            model.fc = nn.Linear(in_features, new_num_classes)
            new_fc_params = model.fc.parameters()
        
        # Размораживаем только новый fc слой
        print(f"Размораживаем fc слой (in_features={in_features}, out_features={new_num_classes})...")
        for param in new_fc_params:
            param.requires_grad = True
        
        # Опционально размораживаем последний блок
        if hasattr(opt, 'unfreeze_last_block') and opt.unfreeze_last_block:
            if hasattr(model, 'module'):
                layer4_params = model.module.layer4.parameters()
            else:
                layer4_params = model.layer4.parameters()
            
            print("Размораживаем layer4...")
            for param in layer4_params:
                param.requires_grad = True
        
        # Возвращаем только обучаемые параметры
        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return model, trainable_parameters
    
    return model, model.parameters()

def generate_base_model(opt):
    """Создает базовую модель ResNet без загрузки весов."""
    assert opt.model in ['resnet']
    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
    
    model_constructors = {
        10: resnet.resnet10,
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        200: resnet.resnet200
    }
    
    constructor = model_constructors[opt.model_depth]
    model = constructor(
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes
    )
    
    return model

def generate_model(opt):
    """
    Главная функция генерации модели.
    Автоматически определяет, нужна ли адаптация размера входа.
    """
    # Если есть претрейн и размеры не совпадают, используем трансферное обучение
    if (hasattr(opt, 'pretrain_path') and opt.pretrain_path and 
        hasattr(opt, 'pretrain_input_size') and 
        opt.pretrain_input_size != opt.input_W):
        return generate_model_with_transfer_learning(opt)
    
    # Иначе используем стандартную генерацию
    return generate_standard_model(opt)

def generate_standard_model(opt):
    """Оригинальная функция генерации модели."""
    assert opt.model in ['resnet']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        model = generate_base_model(opt)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        # Исправленная загрузка для CPU/GPU совместимости
        if opt.no_cuda or not torch.cuda.is_available():
            pretrain = torch.load(opt.pretrain_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            pretrain = torch.load(opt.pretrain_path, weights_only=True)

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

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
