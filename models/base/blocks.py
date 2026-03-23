"""
Basic building blocks for neural networks.
"""
import torch.nn as nn
from ..utils import init_weights


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block with group convolution support."""
    
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)


class ResBasicBlock(nn.Module):
    """Basic residual block."""
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(inplanes, planes, k=3, s=stride)
        self.conv2 = ConvBNAct(planes, planes, k=3, s=1, act=False)
        self.relu = nn.ReLU(inplace=True)
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


class SobelGrad(nn.Module):
    """Compute gradient magnitude using fixed Sobel kernels."""
    
    def __init__(self):
        super().__init__()
        import torch
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        import torch.nn.functional as F
        from ..utils import compute_gradient_magnitude
        return compute_gradient_magnitude(x, self.kx, self.ky)