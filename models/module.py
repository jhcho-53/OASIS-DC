import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
import os, re, cv2, json, time, argparse, numpy as np
from glob import glob

# -------------------- Model blocks --------------------

class TinyFeat(nn.Module):
    """[RGB_norm, P/dmax, E_norm, ML] -> 64ch (BN+Dropout 추가)"""
    def __init__(self, in_ch=3+1+1+1, ch=64, drop_p: float = 0.10, use_bn: bool = True):
        super().__init__()
        layers = []
        cur_in = in_ch
        for _ in range(3):
            layers.append(nn.Conv2d(cur_in, ch, 3, 1, 1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            if drop_p > 0:
                layers.append(nn.Dropout2d(drop_p))
            cur_in = ch
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class ResidualHead(nn.Module):
    def __init__(self, in_ch=64, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(hid,   hid, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(hid,     1, 3,1,1),
        )
    def forward(self, feat): return self.net(feat)

class CurvatureGen(nn.Module):
    """
    Curvature & FiLM generator (shared)
      inputs: feat(64) + E_norm(1)
      outputs per-kernel: kappa∈[kmin,kmax], scale∈[0.5,1.5], bias∈[-0.5,0.5]
    """
    def __init__(self, in_ch=64+1, K: Tuple[int,...]=(3,5,7),
                 kappa_min=1e-3, kappa_max=1.0,
                 drop_p: float = 0.10, use_bn: bool = True):
        super().__init__()
        self.K = K
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        out_ch = len(K) * 3

        body = []
        body.append(nn.Conv2d(in_ch, 64, 3, 1, 1, bias=not use_bn))
        if use_bn: body.append(nn.BatchNorm2d(64))
        body.append(nn.ReLU(inplace=True))
        if drop_p > 0: body.append(nn.Dropout2d(drop_p))

        body.append(nn.Conv2d(64, 64, 3, 1, 1, bias=not use_bn))
        if use_bn: body.append(nn.BatchNorm2d(64))
        body.append(nn.ReLU(inplace=True))
        if drop_p > 0: body.append(nn.Dropout2d(drop_p))

        body.append(nn.Conv2d(64, out_ch, 1, 1, 0))
        self.body = nn.Sequential(*body)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, feat, E_norm):
        x = torch.cat([feat, E_norm], dim=1)
        y = self.body(x)
        B, C, H, W = y.shape
        m = len(self.K)
        y = y.view(B, m, 3, H, W)  # [B, K, {kappa,scale,bias}, H, W]

        kappa = torch.sigmoid(y[:, :, 0]) * (self.kappa_max - self.kappa_min) + self.kappa_min
        scale = torch.sigmoid(y[:, :, 1]) * 1.0 + 0.5   # [0.5, 1.5]
        bias  = torch.tanh(  y[:, :, 2]) * 0.5          # [-0.5, 0.5]
        return kappa, scale, bias

# -------- shared helpers --------

def normalize_affinity_list(A_list: List[torch.Tensor],
                            rho: float = 0.9, eps: float = 1e-6) -> List[torch.Tensor]:
    """
    각 커널의 affinity A(B, kk, H, W)에 대해:
      A' = tanh(A)                  # 값 제한(포화)으로 이상치 완화
      A_norm = rho * A' / (sum(|A'|, dim=kk) + eps)
    ⇒ ∑_j |w_ij| = rho < 1  보장 → 반복 전파 수치 안정성 ↑, 과적합/발산 억제
    """
    out = []
    for Ak in A_list:
        Ak = torch.tanh(Ak)
        Z = Ak.abs().sum(dim=1, keepdim=True) + eps
        out.append(rho * (Ak / Z))
    return out

def contract_normalize_affinity_list(A_list_raw: List[torch.Tensor],
                                     rho: float = 0.9,
                                     eps: float = 1e-6) -> List[torch.Tensor]:
    """
    각 커널의 affinity A (B, kk, H, W)를 signed-L1 수축 정규화:
        A_norm = rho * A / (sum(|A|, dim=kk) + eps)
    => ∑_j |w_ij| = rho < 1 보장 → 반복 전파 수치 안정성 확보.
    """
    A_list = []
    for A in A_list_raw:
        Z = A.abs().sum(dim=1, keepdim=True) + eps
        A_norm = rho * (A / Z)
        A_list.append(A_norm)
    return A_list

class KernelGate(nn.Module):
    """σ_k(x): 커널 혼합 게이트 (softmax)"""
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7)):
        super().__init__()
        self.K = K
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(32, len(K), 1,1,0)
        )
    def forward(self, feat):
        g = self.head(feat)                  # (B,K,H,W)
        return torch.softmax(g, dim=1)       # σ over K
    
# learnable anchor (α map) head
class AnchorHead(nn.Module):
    """feat -> α(x) in [0,1]"""
    def __init__(self, in_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,  1,  1,1,0)
        )
    def forward(self, feat):
        return torch.sigmoid(self.net(feat))
    
def check_device(dev_str: str) -> torch.device:
    dev = torch.device(dev_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check driver/CUDA or use --gpu_device cpu.")
    return dev

@torch.no_grad()
def poisson_gpu(
    sparse_m: np.ndarray, est_m: np.ndarray,
    tol: float = 1e-5, maxiter: int = 1000,
    device: str = "cuda:0", init: str = "est",
    clip_to_max_gt: bool = False
):
    """
    A: 5-점 라플라시안(conv2d), Dirichlet 제약: known = (sparse>0) ∪ border
    RHS: b = A@est - A@v_known (unknown 영역만)
    CG: Jacobi(대각 4) 전처리
    """
    dev = check_device(device)
    H, W = est_m.shape

    est    = torch.from_numpy(est_m).to(dev, dtype=torch.float32)[None, None]   # (1,1,H,W)
    sparse = torch.from_numpy(sparse_m).to(dev, dtype=torch.float32)[None, None]

    mask_sparse = (sparse > 0)
    mask_border = torch.zeros_like(mask_sparse, dtype=torch.bool)
    mask_border[:, :, 0, :]  = True
    mask_border[:, :, -1, :] = True
    mask_border[:, :, :, 0]  = True
    mask_border[:, :, :, -1] = True

    known_mask   = mask_sparse | mask_border
    unknown_mask = ~known_mask

    v_known = torch.where(mask_sparse, sparse, est) * known_mask.float()

    K = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], device=dev, dtype=torch.float32).view(1,1,3,3)
    def lap(x): return F.conv2d(x, K, padding=1)

    b_full = lap(est)
    b_u = (b_full - lap(v_known)) * unknown_mask.float()

    if init == "est":
        x = (est * unknown_mask.float()).clone()
    else:
        x = torch.zeros_like(est)

    def A_apply(xfull):
        xfull = xfull * unknown_mask.float()
        y = lap(xfull)
        return y * unknown_mask.float()

    Minv = 0.25 * unknown_mask.float()  # Jacobi

    use_events = (dev.type == "cuda")
    if use_events:
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True); t0.record()
    else:
        t_start = time.perf_counter()

    r = b_u - A_apply(x)
    z = Minv * r
    p = z.clone()
    rz_old = torch.sum(r*z)
    rhs_norm = torch.sqrt(torch.sum(b_u*b_u)) + 1e-12

    it = 0
    for it in range(1, maxiter+1):
        Ap = A_apply(p)
        denom = torch.sum(p*Ap) + 1e-12
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rel = torch.sqrt(torch.sum(r*r)) / rhs_norm
        if rel.item() < tol:
            break

        z = Minv * r
        rz_new = torch.sum(r*z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new

    if use_events:
        t1.record(); torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0
    else:
        elapsed = time.perf_counter() - t_start

    x_full = x + v_known
    out = x_full.squeeze().detach().cpu().numpy().astype(np.float32)

    if clip_to_max_gt:
        max_gt = float(sparse_m.max())
        if max_gt > 0:
            np.clip(out, 0.0, max_gt, out=out)

    stats = {"solver": "cg(torch)", "time_sec": elapsed, "cg_iters": it, "device": str(dev)}
    return out, stats