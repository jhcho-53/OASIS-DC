from models.model import MCPropNet, MCPropCfg 
import os
import torch

# -----------------------------
# Utilities
# -----------------------------
def default_iters_for_shots(k):
    # Heuristic from the paper-level guidance
    if k <= 1:   return 600
    if k <= 10:  return 1500
    if k <= 100: return 3000
    return 3000

def save_ckpt(path, net, optimizer, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": net.state_dict(),
        "opt": optimizer.state_dict(),
        "meta": meta,
    }, path)
    print(f"[ckpt] saved: {path}")

def build_cfg_from_args(args):
    # 1) 우선 MCPropCfg가 받는 “원래” 필드만 생성자에 전달
    cfg = MCPropCfg(
        dmax=args.dmax,
        steps=args.steps,
        kernels=args.kernels,
        geometry=args.geometry,
        use_residual=not args.no_residual,
        use_sparse=not args.no_sparse,
        anchor_learnable=not args.anchor_fixed,
        anchor_mode=args.anchor_mode,
        anchor_alpha=args.anchor_alpha,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
    )
    # 2) 그 다음 Poisson 옵션은 “동적 속성”으로 붙이기
    cfg.use_poisson = (not args.no_poisson)
    cfg.poisson_tol = args.poisson_tol
    cfg.poisson_maxiter = args.poisson_maxiter
    cfg.poisson_init = args.poisson_init          # "est" | "zero"
    cfg.poisson_clip_to_max_gt = args.poisson_clip_to_max_gt
    return cfg

# utils.py
import torch, os
from collections import defaultdict
import numpy as np
from PIL import Image

# optional colorizers
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from matplotlib import cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
def rmse(pred, target, mask):
    diff = (pred - target) * mask
    mse = (diff**2).sum(dim=[1,2,3]) / torch.clamp(mask.sum(dim=[1,2,3]), min=1.0)
    return torch.sqrt(mse).mean().item()

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.tot=0.0; self.cnt=0
    def update(self, v, n=1): self.tot+=v*n; self.cnt+=n
    @property
    def avg(self): return self.tot/max(self.cnt,1)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def depth_to_uint16_mm(depth_m: np.ndarray, max_m: float = 10.0) -> np.uint16:
    x = np.clip(depth_m, 0.0, max_m) * 1000.0
    return x.astype(np.uint16)

def save_depth_png16(path_png: str, depth_m: np.ndarray, max_m: float = 10.0):
    arr16 = depth_to_uint16_mm(depth_m, max_m)
    Image.fromarray(arr16).save(path_png)

def save_depth_color(path_png: str, depth_m: np.ndarray, max_m: float = 10.0):
    x = np.clip(depth_m / max_m, 0.0, 1.0)
    x8 = (x * 255.0).astype(np.uint8)
    if _HAS_CV2:
        color = cv2.applyColorMap(x8, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        Image.fromarray(color).save(path_png)
    elif _HAS_MPL:
        color = (cm.jet(x)[:,:,:3] * 255.0).astype(np.uint8)
        Image.fromarray(color).save(path_png)
    else:
        # fallback: gray
        Image.fromarray(x8).save(path_png)

def save_depth_bundle(prefix: str, depth_m: np.ndarray, max_m: float = 10.0):
    """
    저장: {prefix}.png (16-bit mm), {prefix}_jet.png (color)
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    save_depth_png16(prefix + ".png", depth_m, max_m=max_m)
    save_depth_color(prefix + "_jet.png", depth_m, max_m=max_m)