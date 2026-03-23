from typing import Tuple
import torch
import numpy as np

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

def sample_sparse_depth(gt_depth, n=500, max_depth=10.0, seed=None):
    rng = np.random.default_rng(seed)
    valid = (gt_depth > 0) & np.isfinite(gt_depth)
    if max_depth is not None:
        valid &= (gt_depth <= max_depth)
    idx = np.where(valid)
    num_valid = len(idx[0])
    mask = np.zeros_like(gt_depth, dtype=bool)
    if num_valid > 0:
        k = min(n, num_valid)
        sel = rng.choice(num_valid, size=k, replace=False)
        mask[idx[0][sel], idx[1][sel]] = True
    sparse = np.zeros_like(gt_depth, dtype=np.float32)
    sparse[mask] = gt_depth[mask]
    return sparse, mask