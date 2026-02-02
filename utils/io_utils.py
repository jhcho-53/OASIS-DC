import cv2
import torch
import numpy as np
import torch.nn.functional as F

# -------------------- I/O utils --------------------

def load_rgb(path: str) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(im).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)

def load_depth16_mm_as_m(path: str) -> torch.Tensor:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: raise FileNotFoundError(path)
    if d.ndim == 3: d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    d = d.astype(np.float32) / 256.0
    return torch.from_numpy(d).unsqueeze(0).unsqueeze(0)     # (1,1,H,W)

def load_pseudo_auto(path: str, zmin=0.5, zmax=80.0) -> torch.Tensor:
    """pseudo는 16-bit(m*256) 또는 8-bit일 수 있음 → m 단위로 반환"""
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None: raise FileNotFoundError(path)
    if x.ndim == 3: x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if x.dtype == np.uint16 and x.max() > 255:
        z = x.astype(np.float32) / 256.0
    else:
        z = (x.astype(np.float32)/255.0)*(zmax-zmin)+zmin
    return torch.from_numpy(z).unsqueeze(0).unsqueeze(0)

def load_estimation_8bit_norm(path: str) -> torch.Tensor:
    """Estimation 0-255 → [0,1]"""
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None: raise FileNotFoundError(path)
    if x.ndim == 3: x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.astype(np.float32)/255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

def save_jet(path: str, z_m: torch.Tensor, dmin=0.5, dmax=80.0):
    z = z_m.detach().cpu().clamp(dmin, dmax)
    z = ((z - dmin) / (dmax - dmin + 1e-6)).squeeze().numpy()
    z8 = np.uint8(np.clip(z*255.0, 0, 255))
    jet = cv2.applyColorMap(z8, cv2.COLORMAP_JET)
    cv2.imwrite(path, jet)

def save_sparse_jet(path: str, DL: torch.Tensor, ML: torch.Tensor, dmin=0.5, dmax=80.0):
    """유효 LiDAR만 컬러, 나머지는 검정."""
    z = DL.detach().cpu().clamp(dmin, dmax).squeeze().numpy()
    m = (ML.detach().cpu().squeeze().numpy() > 0).astype(np.uint8)
    z_norm = np.uint8(np.clip((z - dmin) / (dmax - dmin + 1e-6) * 255.0, 0, 255))
    jet = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
    jet[m == 0] = (0, 0, 0)  # invalid은 검정
    cv2.imwrite(path, jet)

def unfold_neighbors(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k//2
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (B, k*k, H*W) for 1ch
    return patches.view(x.size(0), k*k, x.size(2), x.size(3))  # (B, kk, H, W)
