"""
Encoder architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .blocks import ConvBNAct, ResBasicBlock
from ..utils import apply_weight_init


class TinyFeat(nn.Module):
    """Lightweight feature encoder: [RGB_norm, P/dmax, E_norm, ML] -> 64ch"""
    
    def __init__(self, in_ch=6, ch=64, drop_p: float = 0.10, use_bn: bool = True):
        super().__init__()
        layers = []
        cur_in = in_ch
        for _ in range(3):
            layers.append(nn.Conv2d(cur_in, ch, 3, 1, 1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            if drop_p > 0:
                layers.append(nn.Dropout2d(drop_p))
            cur_in = ch
        self.net = nn.Sequential(*layers)
        apply_weight_init(self)

    def forward(self, x):
        return self.net(x)


def _make_layer(inplanes, planes, blocks, stride, drop=0.0):
    """Create residual layer."""
    down = None
    if stride != 1 or inplanes != planes:
        down = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = [ResBasicBlock(inplanes, planes, stride=stride, downsample=down, drop=drop)]
    for _ in range(1, blocks):
        layers.append(ResBasicBlock(planes, planes, stride=1, downsample=None, drop=drop))
    return nn.Sequential(*layers)


class ResNetLiteEncoder(nn.Module):
    """Lightweight ResNet encoder with stride=4 output + shallow upsampler for full-res 64ch features."""
    
    def __init__(self, in_ch=3, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, 64, k=7, s=2, p=3)  # /2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # /4
        
        # ResNet layers (keep stride=4)
        self.layer1 = _make_layer(64, 64, blocks=2, stride=1, drop=drop)
        self.layer2 = _make_layer(64, 128, blocks=2, stride=1, drop=drop)
        
        # Upsampler: 128ch -> 64ch, /4 -> /1
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        apply_weight_init(self)

    def forward(self, x):
        x = self.conv1(x)      # /2
        x = self.maxpool(x)    # /4
        x = self.layer1(x)     # /4, 64ch
        x = self.layer2(x)     # /4, 128ch
        x = self.up(x)         # /1, 64ch
        return x


class CurvatureGen(nn.Module):
    """
    Curvature & FiLM generator for affinity computation.
    inputs: feat(64) + E_norm(1)
    outputs per-kernel: kappa∈[kmin,kmax], scale∈[0.5,1.5], bias∈[-0.5,0.5]
    """
    
    def __init__(self, in_ch=65, K: Tuple[int, ...] = (3, 5, 7),
                 kappa_min=1e-3, kappa_max=1.0,
                 drop_p: float = 0.10, use_bn: bool = True):
        super().__init__()
        self.K = K
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        out_ch = len(K) * 3

        body = []
        body.append(nn.Conv2d(in_ch, 64, 3, 1, 1, bias=not use_bn))
        if use_bn:
            body.append(nn.BatchNorm2d(64))
        body.append(nn.ReLU(inplace=True))
        if drop_p > 0:
            body.append(nn.Dropout2d(drop_p))

        body.append(nn.Conv2d(64, 64, 3, 1, 1, bias=not use_bn))
        if use_bn:
            body.append(nn.BatchNorm2d(64))
        body.append(nn.ReLU(inplace=True))
        if drop_p > 0:
            body.append(nn.Dropout2d(drop_p))

        body.append(nn.Conv2d(64, out_ch, 1, 1, 0))
        self.body = nn.Sequential(*body)
        apply_weight_init(self)

    def forward(self, feat, E_norm):
        x = torch.cat([feat, E_norm], dim=1)
        y = self.body(x)
        B, C, H, W = y.shape
        m = len(self.K)
        y = y.view(B, m, 3, H, W)  # [B, K, {kappa,scale,bias}, H, W]

        kappa = torch.sigmoid(y[:, :, 0]) * (self.kappa_max - self.kappa_min) + self.kappa_min
        scale = torch.sigmoid(y[:, :, 1]) * 1.0 + 0.5   # [0.5, 1.5]
        bias = torch.tanh(y[:, :, 2]) * 0.5              # [-0.5, 0.5]
        return kappa, scale, bias