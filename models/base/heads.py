"""
Network heads for various tasks.
"""
import torch
import torch.nn as nn
from ..utils import apply_weight_init


class ResidualHead(nn.Module):
    """Head for residual depth prediction."""
    
    def __init__(self, in_ch=64, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 3, 1, 1),
        )
        apply_weight_init(self)
        
    def forward(self, feat):
        return self.net(feat)


class KernelGate(nn.Module):
    """Gating mechanism for kernel selection."""
    
    def __init__(self, in_ch=64, K=(3, 5, 7)):
        super().__init__()
        self.K = K
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, len(K), 1), 
            nn.Sigmoid()
        )
        apply_weight_init(self)
        
    def forward(self, feat):
        return self.gate(feat)  # (B, len(K), 1, 1)


class AnchorHead(nn.Module):
    """Head for anchor point prediction with learnable alpha."""
    
    def __init__(self, in_ch=64, mode="map", learnable=False, init_alpha=0.1):
        super().__init__()
        self.mode = mode  # "scalar" or "map"
        self.learnable = learnable
        
        if learnable:
            if mode == "scalar":
                self.alpha_param = nn.Parameter(torch.tensor(init_alpha))
            elif mode == "map":
                self.alpha_head = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
                )
                apply_weight_init(self)
        else:
            self.register_buffer('alpha_fixed', torch.tensor(init_alpha))

    def forward(self, feat=None):
        if not self.learnable:
            return self.alpha_fixed
            
        if self.mode == "scalar":
            return torch.sigmoid(self.alpha_param)
        elif self.mode == "map":
            return self.alpha_head(feat)  # (B, 1, H, W)
        
        
class InputFusionStem(nn.Module):
    """Fuse multi-channel priors (image + depth priors + sparse mask) -> 3ch 'image-like' tensor."""
    
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        from .blocks import ConvBNReLU
        self.fuse = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, k=3, s=1, p=1),
            ConvBNReLU(mid_ch, mid_ch, k=3, s=1, p=1),
            nn.Conv2d(mid_ch, 3, kernel_size=1, bias=False)
        )
        apply_weight_init(self)

    def forward(self, x):
        return self.fuse(x)