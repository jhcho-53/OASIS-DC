"""
Shape manipulation utilities for tensor operations.
"""
import torch
import torch.nn.functional as F


def _strip_extra_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Remove unnecessary batch dimensions like (B,1,C,H,W) -> (B,C,H,W)"""
    while x.dim() > 4 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def _as_chw4(x: torch.Tensor) -> torch.Tensor:
    """
    Force tensor to (B,C,H,W) format:
    - (B,1,C,H,W) -> (B,C,H,W)
    - (B,H,W,C)   -> (B,C,H,W)
    """
    x = _strip_extra_batch_dim(x)
    if x.dim() == 4 and x.size(1) in (1, 3):
        return x
    if x.dim() == 4 and x.size(-1) in (1, 3) and x.size(1) not in (1, 3):
        return x.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Expected 4D (B,C,H,W) or (B,H,W,C), got {tuple(x.shape)}")


def _as_1ch4(x: torch.Tensor) -> torch.Tensor:
    """
    Force depth/mask to (B,1,H,W) format:
    - (B,1,1,H,W) -> (B,1,H,W)
    - (B,H,W)     -> (B,1,H,W)
    - (B,C,H,W) with C>1 -> first channel only
    """
    x = _strip_extra_batch_dim(x)
    if x.dim() == 3:  # (B,H,W)
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    if x.dim() == 4 and x.size(1) > 1:
        return x[:, :1, ...]
    raise RuntimeError(f"Expected 3D (B,H,W) or 4D with C>=1, got {tuple(x.shape)}")


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize x to match ref's (H,W). Use 'nearest' for masks."""
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W):
        return x
    if mode == "bilinear":
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        return F.interpolate(x, size=(H, W), mode="nearest")