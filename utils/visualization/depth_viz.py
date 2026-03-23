import torch
import numpy as np
from PIL import Image
import os

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from matplotlib import cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

def _jet_lut_build(n: int = 256) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut = np.zeros((n, 3), dtype=np.float32)
    r = np.clip(1.5 - np.abs(4*xs - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4*xs - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4*xs - 1), 0.0, 1.0)
    lut[:, 0] = r
    lut[:, 1] = g
    lut[:, 2] = b
    return (lut * 255.0 + 0.5).astype(np.uint8)

_JET_LUT = _jet_lut_build()

def _depth_to_jet_rgb(
    depth_m: torch.Tensor,
    valid_mask: torch.Tensor = None,
    *,
    vmin: float = None,
    vmax: float = None,
    dynamic: bool = False,
    percentiles=(1.0, 99.0),
    dmax_fallback: float = 10.0
) -> np.ndarray:
    x = depth_m
    if x.ndim == 3: 
        x = x.squeeze(0)
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 3: 
            vm = vm.squeeze(0)
        vm = vm > 0
    else:
        vm = torch.ones_like(x, dtype=torch.bool)

    vals = x[vm]
    if vals.numel() == 0:
        lo, hi = 0.0, dmax_fallback
    else:
        if (vmin is not None) and (vmax is not None) and (vmax > vmin):
            lo, hi = float(vmin), float(vmax)
        elif dynamic:
            arr = vals.detach().cpu().numpy().astype(np.float32)
            p0, p1 = float(percentiles[0]), float(percentiles[1])
            lo = float(np.percentile(arr, p0))
            hi = float(np.percentile(arr, p1))
            if hi - lo < 1e-6:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, dmax_fallback

    y = (x - lo) / (hi - lo + 1e-6)
    y = y.clamp(0, 1).detach().cpu().numpy()

    if _HAS_MPL:
        rgb = cm.get_cmap("jet")(y)[..., :3]
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
    else:
        idx = np.clip((y * 255.0).astype(np.int32), 0, 255)
        rgb = _JET_LUT[idx]
    return rgb

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
        Image.fromarray(x8).save(path_png)

def save_depth_bundle(prefix: str, depth_m: np.ndarray, max_m: float = 10.0):
    from ..core.io import save_depth_png16
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    save_depth_png16(prefix + ".png", depth_m, max_m=max_m)
    save_depth_color(prefix + "_jet.png", depth_m, max_m=max_m)