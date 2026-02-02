from typing import Tuple

import torch
import torch.nn.functional as F

# ==================== Sparse helpers ====================
def _sample_sparse(valid_mask: torch.Tensor, N: int) -> torch.Tensor:
    H, W = valid_mask.shape[-2:]
    idx = valid_mask.view(-1).nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return torch.zeros_like(valid_mask)
    N_eff = min(int(N), idx.numel())
    sel = idx[torch.randperm(idx.numel())[:N_eff]]
    ML = torch.zeros((H*W,), dtype=torch.float32, device=valid_mask.device)
    ML[sel] = 1.0
    return ML.view(1, H, W)

def _make_sparse_from_gt(GT_m: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = (GT_m > 0.0)
    ML = _sample_sparse(valid, N)
    DL = GT_m * ML
    return DL, ML


# ==================== Poisson completion ====================
@torch.no_grad()
def poisson_complete(E_m: torch.Tensor, DL: torch.Tensor, ML: torch.Tensor,
                     lam: float = 800.0, iters: int = 300, hard: bool = False,
                     dmax: float = 10.0) -> torch.Tensor:
    """
    Solve: minimize ∫|∇D - ∇E|^2 + λ Σ M(D-DL)^2
    E_m, DL, ML: (1,1,H,W) meters
    """
    def neigh_sum(X):
        up    = F.pad(X, (0,0,1,0), mode='replicate')[:,:,:-1,:]
        down  = F.pad(X, (0,0,0,1), mode='replicate')[:,:,1:,:]
        left  = F.pad(X, (1,0,0,0), mode='replicate')[:,:,:,:-1]
        right = F.pad(X, (0,1,0,0), mode='replicate')[:,:,:,1:]
        return up + down + left + right

    lap_E = neigh_sum(E_m) - 4.0 * E_m
    b = (-lap_E) + lam * ML * DL
    denom = 4.0 + lam * ML

    D = E_m.clone()
    for _ in range(int(iters)):
        sumN = neigh_sum(D)
        D_new = (sumN + b) / denom
        if hard:
            D_new = ML * DL + (1.0 - ML) * D_new
        D = D_new

    return D.clamp_(0.0, float(dmax))