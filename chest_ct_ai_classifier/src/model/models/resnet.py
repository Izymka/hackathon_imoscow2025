import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# Для инициализации позиционных эмбеддингов в Transformer
try:
    from torch.nn.init import trunc_normal_
except ImportError:
    # Для старых версий PyTorch
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        with torch.no_grad():
            tensor.normal_(mean, std).clamp_(a, b)

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200', 'SpatialAttentionEncoder', 'TransformerEncoderLayer3D', 
    'HybridResNetTransformer', 'hybrid_resnet34_transformer', 'CBAM'
]


class CBAM(nn.Module):
    """Convolutional Block Attention Module для 3D CNN"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        # Spatial Attention Module  
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Применяем канальное внимание
        x = self.channel_attention(x) * x

        # Применяем пространственное внимание
        x = self.spatial_attention(x) * x

        return x


class ChannelAttention(nn.Module):
    """Модуль канального внимания для CBAM"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention(nn.Module):
    """Модуль пространственного внимания для CBAM"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(concat))
        return spatial_att


class SpatialAttentionEncoder(nn.Module):
    """Пространственный attention энкодер для 3D CNN"""

    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttentionEncoder, self).__init__()

        # Channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP for channel attention
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        # Spatial attention branch
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_attention

        # Spatial attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))

        # Apply spatial attention
        x_out = x_channel * spatial_attention

        return x_out

class TransformerEncoderLayer3D(nn.Module):
    """Transformer encoder layer адаптированный для 3D фич"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True  # важный параметр!
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.GELU()

    def forward(self, src):
        # Self-attention with residual
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(src2, src2, src2)
        src = src + self.dropout(src2)

        # Feed-forward with residual
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout(src2)

        return src, attn_weights


class HybridResNetTransformer(nn.Module):
    """Гибрид ResNet34 + Transformer для 3D КТ снимков"""

    def __init__(self, 
                 sample_input_D=128,
                 sample_input_H=128, 
                 sample_input_W=128,
                 num_classes=2,
                 transformer_d_model=512,
                 transformer_nhead=8,
                 transformer_dim_feedforward=1024,
                 transformer_num_layers=2,
                 transformer_dropout=0.1,
                 use_cbam=False):
        super().__init__()

        # ResNet34 backbone (без финального FC слоя)
        self.backbone = ResNet(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            sample_input_D=sample_input_D,
            sample_input_H=sample_input_H, 
            sample_input_W=sample_input_W,
            num_seg_classes=num_classes,  # временно
            shortcut_type='B',
            no_cuda=False,
            use_cbam=use_cbam
        )

        # Убираем финальные слои ResNet, оставляем только feature extractor
        self.backbone.fc = nn.Identity()  # убираем классификатор
        if hasattr(self.backbone, 'conv_seg'):
            self.backbone.conv_seg = nn.Identity()

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer3D(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            ) for _ in range(transformer_num_layers)
        ])

        # Positional encoding для spatial dimensions
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, transformer_d_model))

        # Классификатор
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_d_model),
            nn.Dropout(transformer_dropout),
            nn.Linear(transformer_d_model, 256),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(256, num_classes)
        )

        # Adaptive pooling для разного размера фич
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # фиксируем размер

        # Инициализация
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def extract_resnet_features(self, x):
        """Извлекаем фичи из ResNet без финальных слоев"""
        # Forward через ResNet
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x) 
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # [B, 512, D, H, W]

        return x

    def forward(self, x, return_attention=False):
        # 1. Извлекаем фичи с помощью ResNet
        features = self.extract_resnet_features(x)  # [B, 512, D, H, W]

        # 2. Adaptive pooling к фиксированному размеру
        features = self.adaptive_pool(features)  # [B, 512, 4, 4, 4]

        # 3. Подготовка для transformer (flatten spatial dimensions)
        B, C, D, H, W = features.shape
        features_flat = features.view(B, C, -1)  # [B, 512, D*H*W]
        features_flat = features_flat.transpose(1, 2)  # [B, N, 512] где N=64 (4*4*4)

        # 4. Добавляем позиционные эмбеддинги
        features_flat = features_flat + self.pos_encoding[:, :features_flat.size(1), :]

        # 5. Проходим через transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            features_flat, attn_weights = layer(features_flat)
            attention_weights.append(attn_weights.detach())

        # 6. Global average pooling по spatial dimensions
        global_features = features_flat.mean(dim=1)  # [B, 512]

        # 7. Классификация
        output = self.classifier(global_features)

        if return_attention:
            return output, attention_weights
        return output



def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        # CBAM после второго сверточного слоя
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Применяем CBAM после сверточных слоев, но до residual соединения
        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        # CBAM после третьего сверточного слоя
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Применяем CBAM после сверточных слоев, но до residual соединения
        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,  # для совместимости, но используем как num_classes
                 shortcut_type='B',
                 no_cuda=False,
                 use_cbam=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.use_cbam = use_cbam
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, use_cbam=self.use_cbam)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2, use_cbam=self.use_cbam)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2, use_cbam=self.use_cbam)

        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4, use_cbam=self.use_cbam)

        # Заменяем сегментационный выход на классификационный
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_seg_classes)  # num_seg_classes теперь num_classes

        # Убираем старый сегментационный слой
        # self.conv_seg = ...

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Классификационный forward
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def hybrid_resnet34_transformer(**kwargs):
    """Constructs a Hybrid ResNet34-Transformer model.
    """
    model = HybridResNetTransformer(**kwargs)
    return model