# models/backbones.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(inplanes, planes, k=3, s=stride)
        self.conv2 = ConvBNAct(planes, planes, k=3, s=1, act=False)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

def _make_layer(inplanes, planes, blocks, stride, drop=0.0):
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
    """
    라이트 ResNet 인코더 (stride=4 출력) + 얕은 업샘플러로 full-res 64ch 피처 생성.
    arch='resnet18' -> [2,2] 블록, 'resnet34' -> [3,4] 블록 사용 (layer1, layer2만 사용).
    """
    def __init__(self,
                 in_ch: int = 6,
                 arch: Literal["resnet18","resnet34"]="resnet18",
                 out_ch: int = 64,
                 drop2d: float = 0.1,
                 freeze_bn: bool = False):
        super().__init__()
        blocks = (2,2) if arch == "resnet18" else (3,4)
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 64, k=7, s=2, p=3),           # H/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4
        )
        self.layer1 = _make_layer(64, 64, blocks=blocks[0], stride=1, drop=drop2d)  # H/4
        self.layer2 = _make_layer(64, 128, blocks=blocks[1], stride=2, drop=drop2d) # H/8

        # FPN-lite: 1/8 -> 1/4 (업), 1/4 skip(layer1) 결합 → 64ch, full-res 업
        self.lat_1_8 = ConvBNAct(128, 64, k=1, s=1)
        self.lat_1_4 = ConvBNAct(64,  64, k=1, s=1)
        self.smooth_1_4 = ConvBNAct(64, 64, k=3, s=1)
        self.proj_full  = ConvBNAct(64, out_ch, k=3, s=1)
        self.drop = nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity()

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):  # (B,in_ch,H,W)
        H, W = x.shape[-2:]
        x = self.stem(x)           # 1/4
        c1_4 = self.layer1(x)      # 1/4, 64
        c1_8 = self.layer2(c1_4)   # 1/8, 128

        p1_8 = self.lat_1_8(c1_8)                               # 1/8, 64
        p1_4 = self.lat_1_4(c1_4) + F.interpolate(p1_8, size=c1_4.shape[-2:], mode="bilinear", align_corners=False)
        p1_4 = self.smooth_1_4(p1_4)                            # 1/4, 64

        full = F.interpolate(p1_4, size=(H,W), mode="bilinear", align_corners=False)  # 1/1
        out  = self.proj_full(full)                             # 64
        out  = self.drop(out)
        return out  # (B,64,H,W)
