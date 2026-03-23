"""
Hyperbolic-inspired affinity computation.
"""
import torch
import torch.nn as nn
from typing import Tuple, List
from ..utils import apply_weight_init


class HCLApproxAffinity(nn.Module):
    """
    (Approx.) Hyperbolic Convolution for affinity maps
      A_k = Conv_k( scale_k * feat + bias_k )  -> (B, k*k, H, W)
    
    Uses FiLM (Feature-wise Linear Modulation) followed by convolution.
    """
    
    def __init__(self, in_ch=64, K: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleDict()
        for k in K:
            kk = k * k
            self.convs[str(k)] = nn.Conv2d(in_ch, kk, kernel_size=3, stride=1, padding=1, bias=True)
        apply_weight_init(self)

    def forward(self, feat, scale, bias) -> List[torch.Tensor]:
        """
        Args:
            feat: Feature tensor (B, C, H, W)
            scale: Scale parameters (B, len(K), H, W)
            bias: Bias parameters (B, len(K), H, W)
            
        Returns:
            List of affinity tensors for each kernel size
        """
        out = []
        for idx, k in enumerate(self.K):
            s = scale[:, idx:idx+1]  # (B, 1, H, W)
            b = bias[:, idx:idx+1]   # (B, 1, H, W)
            Fm = s * feat + b        # FiLM modulation
            Ak = self.convs[str(k)](Fm)  # (B, k*k, H, W)
            out.append(Ak)
        return out