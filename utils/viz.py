import torch
import numpy as np
try:
    import matplotlib.cm as _cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def _jet_lut_build(n: int = 256) -> np.ndarray:
    """
    Build a simple JET-like LUT (fallback when matplotlib is unavailable).
    Returns uint8 array of shape (n,3) in [0,255].
    """
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut = np.zeros((n, 3), dtype=np.float32)
    # piecewise triangular waves to mimic 'jet'
    r = np.clip(1.5 - np.abs(4*xs - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4*xs - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4*xs - 1), 0.0, 1.0)
    lut[:, 0] = r; lut[:, 1] = g; lut[:, 2] = b
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
    """
    Convert depth (meters) to JET-colored RGB (uint8 HxWx3).
    - depth_m: (H,W) or (1,H,W) torch float tensor (meters)
    - valid_mask: (H,W) or (1,H,W) torch bool/float tensor
    - If dynamic: compute [vmin,vmax] from percentiles on valid region.
    - Else: use provided vmin/vmax; if None, use [0, dmax_fallback].
    """
    x = depth_m
    if x.ndim == 3: x = x.squeeze(0)
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 3: vm = vm.squeeze(0)
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
        rgb = _cm.get_cmap("jet")(y)[..., :3]  # float in [0,1]
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
    else:
        idx = np.clip((y * 255.0).astype(np.int32), 0, 255)
        rgb = _JET_LUT[idx]  # (H,W,3) uint8
    return rgb