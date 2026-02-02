from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Hyperbolic-inspired (approx.) affinity --------

class HCLApproxAffinity(nn.Module):
    """
    (Approx.) Hyperbolic Convolution for affinity maps
      A_k = Conv_k( scale_k * feat + bias_k )  -> (B, k*k, H, W)
    이후 normalize_affinity_list()로 안정화.
    """
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7)):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleDict()
        for k in K:
            kk = k*k
            self.convs[str(k)] = nn.Conv2d(in_ch, kk, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feat, scale, bias) -> List[torch.Tensor]:
        out = []
        for idx, k in enumerate(self.K):
            s = scale[:, idx:idx+1]  # (B,1,H,W)
            b = bias[:,  idx:idx+1]
            Fm = s * feat + b        # FiLM
            Ak = self.convs[str(k)](Fm)  # (B, k*k, H, W)
            out.append(Ak)
        return out


# -------- Elliptic/Spherical affinity (ours) --------

class EllipticAffinity(nn.Module):
    """
    Local spherical (elliptic) affinity via cosine similarity with temperature from curvature.
      - For each kernel k:
          q_k = Conv1x1(feat), k_k = Conv1x1(feat)  (Ca channels)
          Ak_logits(i, offsets) = cos( q_k(i), k_k(j) ) / tau_k(i)
      - logits → tanh → L1-normalize (signed CSPN style).
    """
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7), c_aff: int = 32,
                 tau_min: float = 0.03, tau_max: float = 0.5):
        super().__init__()
        self.K = K
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.qconvs = nn.ModuleDict()
        self.kconvs = nn.ModuleDict()
        for k in K:
            self.qconvs[str(k)] = nn.Conv2d(in_ch, c_aff, kernel_size=1, bias=False)
            self.kconvs[str(k)] = nn.Conv2d(in_ch, c_aff, kernel_size=1, bias=False)

    def _tau_from_kappa(self, kappa: torch.Tensor) -> torch.Tensor:
        """kappa ∈ [kmin,kmax] -> tau ∈ [tau_min, tau_max]"""
        kappa_norm = (kappa - kappa.min()) / (kappa.max() - kappa.min() + 1e-6)
        tau = self.tau_max - kappa_norm * (self.tau_max - self.tau_min)
        return tau.clamp(self.tau_min, self.tau_max)  # (B,1,H,W)

    def forward(self, feat, kappa) -> List[torch.Tensor]:
        out = []
        for idx, k in enumerate(self.K):
            q = self.qconvs[str(k)](feat)                    # (B,Ca,H,W)
            key = self.kconvs[str(k)](feat)                  # (B,Ca,H,W)
            qn = F.normalize(q, dim=1)
            kn = F.normalize(key, dim=1)

            kk = k*k
            patches = F.unfold(kn, kernel_size=k, padding=k//2)  # (B, Ca*kk, H*W)
            B, Ca, H, W = kn.shape
            patches = patches.view(B, Ca, kk, H, W)              # (B,Ca,kk,H,W)

            sim = (qn.unsqueeze(2) * patches).sum(1)             # (B,kk,H,W), cosine
            tau = self._tau_from_kappa(kappa[:, idx:idx+1])      # (B,1,H,W)
            logits = sim / (tau + 1e-6)                          # (B,kk,H,W)
            out.append(logits)
        return out
