import torch
import torch.nn.functional as F

def _as_chw4(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(0) == 1:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (1,H,W) or (B,1,H,W), got {x.shape}")

def _as_1ch4(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (H,W), (1,H,W), or (B,1,H,W), got {x.shape}")

def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape[-2:] == ref.shape[-2:]:
        return src
    H, W = ref.shape[-2:]
    return F.interpolate(src, size=(H, W), mode='bilinear', align_corners=False)