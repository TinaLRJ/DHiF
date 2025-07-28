import os
import time
from thop import profile, clever_format
from torch import nn
import torch
import torch.nn.functional as F
from thop import profile, clever_format
from model.basic import SDifferenceConv, DyfConv, RepConv
from model.convs.APConv import PConv
from model.convs.FDConv_initialversion import FDConv
from model.wtconv.wtconv2d import WTConv2d
import numpy as np
import cv2

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 输出1×1的特征图
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_Dyf(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_Dyf, self).__init__()
        self.conv1 = DyfConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_WT(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_WT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = WTConv2d(out_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_SD(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_SD, self).__init__()
        self.conv1 = SDifferenceConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_PC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_PC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = PConv(out_channels, out_channels, 3,1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_FD(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_FD, self).__init__()
        self.conv1 = FDConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


class ResNet_Ref(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_Ref, self).__init__()
        self.conv1 = RepConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,),
            nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        residual1 = self.shortcut1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out += residual1
        out = self.relu(out)
        return out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, -1, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) + self.pe(v.reshape(B, -1, H, W))
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class ConvMlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )
        self.mlp = ConvMlp(in_channels=2 * self.c, hidden_channels=(2 * self.c) * 4, out_channels=2 * self.c, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        concatenated = torch.cat((a, b), 1)
        return self.cv2(self.mlp(concatenated))



class TransposeUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposeUpsample, self).__init__()
        # 步长为2的2x2转置卷积
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        # 批量归一化
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x

class TransposeUpsample4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposeUpsample4, self).__init__()
        # 使用步长为4的4x4转置卷积，一次性放大4倍
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=4, padding=0
        )
        # 批量归一化
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x

class TransposeUpsample8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposeUpsample8, self).__init__()
        # 使用步长为8的8x8转置卷积，一次性放大8倍
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=8, stride=8, padding=0
        )
        # 批量归一化
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x

class SubPixelConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(SubPixelConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class APTNet(nn.Module):
    def __init__(self, input_channels, block=ResNet, mode='test', conv='usual'):
        super().__init__()
        device = torch.device('cuda')
        self.mode = mode
        self.device = device
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [1, 2, 3, 4]
        self.pool = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.pool8 = nn.MaxPool2d(8, 8)
        self.pool16 = nn.MaxPool2d(16, 16)

        # 在decode部分用转置卷积进行上采样
        self.dup1 = TransposeUpsample(240, 272)
        self.dup2 = TransposeUpsample(128, 160)
        self.dup3 = TransposeUpsample(64, 96)
        self.dup4 = TransposeUpsample(32, 64)
        self.up = TransposeUpsample(1, 1)
        self.up_4 = TransposeUpsample4(1, 1)
        self.up_8 = TransposeUpsample8(1, 1)

        # 1×1卷积升维
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        if conv == 'usual':
            self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block, param_blocks[0])  # ResNet×2
            self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[1])  # MresNet×2
            self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[2])  # MresNet×2
            self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[3])  # MresNet×2
        else:
            if conv == 'Dyf':
                block_D = ResNet_Dyf
            elif conv == 'WTC':
                block_D = ResNet_WT
            elif conv == 'SDC':
                block_D = ResNet_SD
            elif conv == 'PC':
                block_D = ResNet_PC
            elif conv == 'FD':
                block_D = ResNet_FD
            elif conv == 'Ref':
                block_D = ResNet_Ref
            self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block_D, param_blocks[0])  # block_D
            self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block_D, param_blocks[1])  # block_D
            self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block_D, param_blocks[2])  # block_D
            self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block_D, param_blocks[3])  # block_D

        self.middle_layer = self._make_layer(240, 240, block, param_blocks[1])

        # PSA modules
        self.psa0 = PSA(240, 240, 0.5)  # 默认参数
        self.psa1 = PSA(240, 240, 0.5)  # 默认参数

        self.decoder_3 = self._make_layer(400, param_channels[3], block, param_blocks[1])  # MresNet×2
        self.decoder_2 = self._make_layer(224, param_channels[2], block, param_blocks[1])  # MresNet×2
        self.decoder_1 = self._make_layer(128, param_channels[1], block, param_blocks[1])  # MresNet×2
        self.decoder_0 = self._make_layer(80, param_channels[0], block, param_blocks[1])  # MresNet×2

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(ResNet(in_channels, out_channels))
        for i in range(block_num - 1):
            if i < block_num - 2:
                layer.append(ResNet(out_channels, out_channels))
            else:
                layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, flag=True):


        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = torch.cat([self.pool16(x_e0), self.pool8(x_e1), self.pool4(x_e2), self.pool(x_e3)], 1)

        x_m_channels = x_m.size(1)
        if x_m_channels < 112:
            psa_param = 0.9
        elif x_m_channels < 240:
            psa_param = 0.7
        else:
            psa_param = 0.5

        self.psa0 = PSA(x_m_channels, x_m_channels, psa_param).cuda()
        self.psa1 = PSA(x_m_channels, x_m_channels, psa_param).cuda()

        x_m1 = self.psa0(x_m)
        x_m2 = self.psa1(x_m1)

        x_d3 = self.decoder_3(torch.cat([x_e3, self.dup1(x_m2)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.dup2(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.dup3(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.dup4(x_d1)], 1))

        if flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.up(self.output_1(x_d1))
            mask2 = self.up_4(self.output_2(x_d2))
            mask3 = self.up_8(self.output_3(x_d3))
            output = self.final(torch.cat([mask0, mask1, mask2, mask3], dim=1))
            if self.mode == 'train':
                return [mask0.sigmoid(), mask1.sigmoid(), mask2.sigmoid(), mask3.sigmoid(), output.sigmoid()]
            else:
                return output.sigmoid()

        else:
            output = self.output_0(x_d0)
            return output.sigmoid()





if __name__ == '__main__':


    model = APTNet(3).to('cuda')
    x = torch.randn(1, 3, 256, 256).to('cuda')
    output = model(x,flag=False)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')






