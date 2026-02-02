from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- config / dataset / utils ----
from config.schema import MCPropCfg           # dataclass for model cfg (anchor options 포함)
from utils.io_utils import save_jet, save_sparse_jet, unfold_neighbors
# ---- model parts (동일) ----
from models.module import TinyFeat, ResidualHead, CurvatureGen, KernelGate, normalize_affinity_list, AnchorHead
from models.affinity import EllipticAffinity, HCLApproxAffinity
from models.module import poisson_gpu
# -------------------- shape helpers --------------------
def _strip_extra_batch_dim(x: torch.Tensor) -> torch.Tensor:
    # (B,1,C,H,W) 같이 배치 뒤에 불필요한 1차원이 붙은 경우 제거
    while x.dim() > 4 and x.size(1) == 1:
        x = x.squeeze(1)
    return x

def _as_chw4(x: torch.Tensor) -> torch.Tensor:
    """
    x -> (B,C,H,W) 강제
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
    depth/mask -> (B,1,H,W)
    - (B,1,1,H,W) -> (B,1,H,W)
    - (B,H,W)     -> (B,1,H,W)
    - (B,C,H,W) with C>1 -> 첫 채널
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
    """x를 ref의 (H,W)에 맞추어 리사이즈. mask류는 nearest."""
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W):
        return x
    if mode == "bilinear":
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        return F.interpolate(x, size=(H, W), mode="nearest")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TinyFeat, ResidualHead, CurvatureGen, KernelGate, AnchorHead,
# HCLApproxAffinity, EllipticAffinity, normalize_affinity_list,
# unfold_neighbors, poisson_gpu 는 기존 모듈을 그대로 사용합니다.

class MCPropNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # -------- Global / Poisson toggles --------
        self.poisson_only = bool(getattr(cfg, "poisson_only", False))
        self.use_poisson = bool(getattr(cfg, "use_poisson", True))
        self.poisson_tol = float(getattr(cfg, "poisson_tol", 1e-5))
        self.poisson_maxiter = int(getattr(cfg, "poisson_maxiter", 1000))
        self.poisson_init = str(getattr(cfg, "poisson_init", "est"))
        self.poisson_clip_to_max_gt = bool(getattr(cfg, "poisson_clip_to_max_gt", False))
        # advanced poisson
        self.poisson_auto_flip   = bool(getattr(cfg, "poisson_auto_flip", True))
        self.poisson_est_affine  = bool(getattr(cfg, "poisson_est_affine", True))
        self.poisson_smooth_est  = bool(getattr(cfg, "poisson_smooth_est", True))

        # -------- P-only affinity switches --------
        self.use_p_affinity = bool(getattr(cfg, "use_p_affinity", True))
        self.p_only_gate    = bool(getattr(cfg, "p_only_gate", False))

        # -------- Encoders / Heads --------
        in_ch = 3 + 1 + 1 + 1  # I(3), P/dmax(1), E(1), ML(1)
        self.enc = TinyFeat(in_ch, 64)

        # P-only encoder: [Pn, edge(P), ML]
        self.p_in_ch = int(getattr(cfg, "p_in_ch", 3))
        if self.p_in_ch != 3:
            self.p_in_ch = 3
        self.penc = TinyFeat(self.p_in_ch, 64) if self.use_p_affinity or self.p_only_gate else None

        self.res = ResidualHead(64, 64) if getattr(cfg, "use_residual", False) else None
        self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            # ✅ “hyperbolic network” 경로: HCLApproxAffinity 사용
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # Learnable anchor
        self.anchor_learnable = bool(cfg.anchor_learnable)
        self.anchor_mode = cfg.anchor_mode.lower()
        self.anchor_init = float(cfg.anchor_alpha)
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(64)
            else:
                raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
        else:
            self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
            self.anchor_param = None
            self.anchor_head = None

        # (옵션) 안정 하한
        self.min_gate  = float(getattr(cfg, "min_gate", 0.0))
        self.min_alpha = float(getattr(cfg, "min_alpha", 0.0))

    # ---------- helpers ----------
    def _alpha_map(self, feat, ML):
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1,1,1,1).expand_as(ML)
            else:
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    @staticmethod
    def _edge_mag_01(X01: torch.Tensor) -> torch.Tensor:
        gx = F.pad(X01[..., :, 1:] - X01[..., :, :-1], (1,0,0,0))
        gy = F.pad(X01[..., 1:, :] - X01[..., :-1, :], (0,0,1,0))
        g = (gx.abs() + gy.abs()) * 0.5
        return g.clamp(0,1)

    @staticmethod
    def _smooth3_reflect_np(e: np.ndarray) -> np.ndarray:
        pad = 1
        ep = np.pad(e, ((pad,pad),(pad,pad)), mode="reflect")
        out = (
            ep[0:-2,0:-2] + ep[0:-2,1:-1] + ep[0:-2,2:] +
            ep[1:-1,0:-2] + 2*ep[1:-1,1:-1] + ep[1:-1,2:] +
            ep[2:  ,0:-2] + ep[2:  ,1:-1] + ep[2:  ,2:]
        ) / 10.0
        return out.astype(np.float32)

    @torch.no_grad()
    def _poisson_batch(self, DL, ML, E01):
        B = DL.shape[0]; dmax = float(self.cfg.dmax); dev = DL.device
        dev_str = str(dev)
        P_list, stats_list = [], []

        for b in range(B):
            e  = E01[b,0].detach().cpu().numpy().astype(np.float32)         # [0,1]
            dl = (DL[b,0] / dmax).detach().cpu().numpy().astype(np.float32) # ~[0,1]
            m  = (ML[b,0].detach().cpu().numpy().astype(np.float32) > 0)

            if self.poisson_auto_flip and m.sum() >= 10:
                em = e[m].reshape(-1); dm = dl[m].reshape(-1)
                if em.size > 1 and dm.size > 1:
                    corr = np.corrcoef(em, dm)[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    if corr < 0.0: e = 1.0 - e

            if self.poisson_est_affine and m.sum() >= 10:
                x = e[m].reshape(-1,1); y = dl[m].reshape(-1,1)
                A = np.concatenate([x, np.ones_like(x)], axis=1)
                w = np.ones((A.shape[0],1), dtype=np.float32)
                for _ in range(3):
                    Aw = A * w; yw = y * w
                    theta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
                    r = (A @ theta - y)
                    c = 1.345 * np.median(np.abs(r)) + 1e-6
                    w = (1.0 / np.maximum(1.0, np.abs(r)/c)).astype(np.float32)
                a, b0 = float(theta[0,0]), float(theta[1,0])
                e = np.clip(a*e + b0, 0.0, 1.0)

            if self.poisson_smooth_est:
                e = self._smooth3_reflect_np(e)

            est_m    = (e * dmax).astype(np.float32)
            sparse_m = (DL[b,0].detach().cpu().numpy().astype(np.float32) * m.astype(np.float32))

            P_np, st = poisson_gpu(
                sparse_m=sparse_m, est_m=est_m,
                tol=self.poisson_tol, maxiter=self.poisson_maxiter,
                device=dev_str, init=self.poisson_init,
                clip_to_max_gt=self.poisson_clip_to_max_gt
            )
            P_list.append(torch.from_numpy(P_np)[None, None])
            stats_list.append(st)

        P = torch.cat(P_list, dim=0).to(device=dev, dtype=torch.float32)
        return P, stats_list

    def forward(self, I, DL, ML, E_norm):
        # shape/dtype normalize
        if I.ndim == 3:  I = I.unsqueeze(0)
        if DL.ndim == 3: DL = DL.unsqueeze(0)
        if ML.ndim == 3: ML = ML.unsqueeze(0)
        if E_norm.ndim == 3: E_norm = E_norm.unsqueeze(0)

        I  = I.float()
        DL = DL.float()
        ML = (ML > 0).float()
        E  = E_norm.float()

        H, W = I.shape[-2:]
        assert DL.shape[-2:] == (H,W) and ML.shape[-2:] == (H,W) and E.shape[-2:] == (H,W), \
            "[MCPropNet] All inputs must share (H,W)"
        if torch.any((E < -1e-6) | (E > 1.0 + 1e-6)):
            raise ValueError("[MCPropNet] E_norm must be in [0,1]")

        dmax = float(self.cfg.dmax)

        # ---- 0) Poisson P (metric) ----
        if self.use_poisson:
            P, poisson_stats = self._poisson_batch(DL, ML, E)
        else:
            P, poisson_stats = (E * dmax).clamp(0, dmax), None

        # (원 요청) 항상 D_pred를 만들기 위해 Poisson-only 조기 리턴 비활성
        # if self.poisson_only: ...
        #   return P, aux

        # ---- 1) main feat (I,P/dmax,E,ML) ----
        x_in = torch.cat([I, P / dmax, E, ML], dim=1)
        feat = self.enc(x_in)

        # ---- 2) P-only feat (Pn, edge(P), ML) ----
        if self.use_p_affinity or self.p_only_gate:
            Pn   = (P / dmax).clamp(0,1)
            edgeP= self._edge_mag_01(Pn)
            p_in = torch.cat([Pn, edgeP, ML], dim=1)   # (B,3,H,W)
            p_feat = self.penc(p_in)                   # (B,64,H,W)
        else:
            p_feat = None

        # ---- 3) 초기값 D0 ----
        res_out = None
        D0 = P
        if self.res is not None:
            res_out = self.res(feat)
            D0 = (P + res_out).clamp(0, dmax)

        # ---- 4) curvature & affinity (HCLApprox / hyperbolic 경로) ----
        if self.use_p_affinity and (p_feat is not None):
            kappa, scale, bias = self.curv(p_feat, (P / dmax).clamp(0,1))
            if self.cfg.geometry.lower().startswith("ellip"):
                A_list_raw = self.aff_head(p_feat, kappa)
            else:
                A_list_raw = self.aff_head(p_feat, scale, bias)  # HCLApprox (hyperbolic)
        else:
            kappa, scale, bias = self.curv(feat, E)
            if self.cfg.geometry.lower().startswith("ellip"):
                A_list_raw = self.aff_head(feat, kappa)
            else:
                A_list_raw = self.aff_head(feat, scale, bias)    # HCLApprox (hyperbolic)

        A_list = normalize_affinity_list(A_list_raw)

        # ---- 5) gate / alpha ----
        if self.p_only_gate and (p_feat is not None):
            sigma = self.gate(p_feat)
            alpha = self._alpha_map(p_feat, ML)
        else:
            sigma = self.gate(feat)
            alpha = self._alpha_map(feat, ML)

        if self.min_gate > 0:
            sigma = sigma.clamp_min(self.min_gate)
        if self.min_alpha > 0:
            alpha = alpha.clamp_min(self.min_alpha)

        # ---- 6) Original propagation loop (baseline) ----
        Dt = D0.clone()
        for _ in range(self.cfg.steps):
            mix = torch.zeros_like(Dt)
            for idx, ksz in enumerate(self.cfg.kernels):
                Ak = A_list[idx]
                kk = ksz * ksz
                patches = unfold_neighbors(Dt, ksz)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0  # 중심은 항상 D0 참조
                Dk = (Ak * patches_center).sum(1, keepdim=True)
                mix = mix + sigma[:, idx:idx+1] * Dk

            if self.cfg.use_sparse:
                Dt = (1.0 - alpha * ML) * mix + (alpha * ML) * DL
            else:
                Dt = mix
            Dt = Dt.clamp(0, dmax)

        aux = {
            "D_pred": Dt,    # 항상 D_pred를 명시
            "D0": D0, "P": P, "residual": res_out,
            "sigma": sigma, "alpha": alpha,
            "A_list": A_list,
            "poisson_stats": poisson_stats,
            "p_affinity_used": bool(self.use_p_affinity)
        }
        return Dt, aux

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small encoders ----------
class TinyFeat(nn.Module):
    def __init__(self, in_ch, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3,1,1), nn.ReLU(True),
            nn.Conv2d(ch,   ch, 3,1,1), nn.ReLU(True),
            nn.Conv2d(ch,   ch, 3,1,1), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class AnisoHead(nn.Module):
    """Predict axis-aligned anisotropy (a_x, a_y), screening λ, sparse anchor μ."""
    def __init__(self, in_ch=64, floor_ax=1e-4, floor_ay=1e-4, floor_lam=1e-3, floor_mu=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3,1,1), nn.ReLU(True),
            nn.Conv2d(32,   4,  1,1,0)
        )
        self.floor_ax  = floor_ax
        self.floor_ay  = floor_ay
        self.floor_lam = floor_lam
        self.floor_mu  = floor_mu

    def forward(self, feat):
        raw = self.conv(feat)   # (B,4,H,W)
        ax  = F.softplus(raw[:,0:1]) + self.floor_ax   # a_x >= floor
        ay  = F.softplus(raw[:,1:2]) + self.floor_ay   # a_y >= floor
        lam = F.softplus(raw[:,2:3]) + self.floor_lam  # λ  >= floor
        mu  = F.softplus(raw[:,3:4]) + self.floor_mu   # μ  >= 0
        return ax, ay, lam, mu

# ---------- anisotropic operator & PCG ----------
def _avg_east(x):  # face weight at i+1/2
    xe = torch.roll(x, shifts=-1, dims=3)
    w  = 0.5*(x + xe)
    w[:,:,:, -1] = 0.0
    return w
def _avg_west(x):
    xw = torch.roll(x, shifts=+1, dims=3)
    w  = 0.5*(x + xw)
    w[:,:,:, 0] = 0.0
    return w
def _avg_south(y):
    ys = torch.roll(y, shifts=-1, dims=2)
    w  = 0.5*(y + ys)
    w[:, :, -1, :] = 0.0
    return w
def _avg_north(y):
    yn = torch.roll(y, shifts=+1, dims=2)
    w  = 0.5*(y + yn)
    w[:, :,  0, :] = 0.0
    return w

def build_axis_aligned_weights(ax, ay):
    # diffusion weights on faces
    wE = _avg_east(ax)   # (B,1,H,W)
    wW = _avg_west(ax)
    wS = _avg_south(ay)
    wN = _avg_north(ay)
    return wE, wW, wS, wN

def apply_operator(u, wE,wW,wS,wN, lam, muML):
    # -div(A∇u) + (lam + muML)*u
    uE = torch.roll(u, -1, dims=3); uW = torch.roll(u, +1, dims=3)
    uS = torch.roll(u, -1, dims=2); uN = torch.roll(u, +1, dims=2)

    L = wE*(uE - u) + wW*(uW - u) + wS*(uS - u) + wN*(uN - u)  # -div term
    Auu = L + (lam + muML) * u
    return Auu

@torch.no_grad()
def pcg_solve(b, wE,wW,wS,wN, lam, muML, tol=1e-5, iters=200, x0=None):
    # A(u) = (-div A∇)u + (lam+muML)u, rhs = b
    B,_,H,W = b.shape
    device  = b.device
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    diag = (wE + wW + wS + wN) + (lam + muML) + 1e-6
    Minv = 1.0 / diag

    r = b - apply_operator(x, wE,wW,wS,wN, lam, muML)
    z = Minv * r
    p = z.clone()
    rz_old = torch.sum(r*z)

    for _ in range(iters):
        Ap = apply_operator(p, wE,wW,wS,wN, lam, muML)
        alpha = rz_old / (torch.sum(p*Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        if (torch.norm(r) / (torch.norm(b)+1e-12)).item() < tol:
            break

        z = Minv * r
        rz_new = torch.sum(r*z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new

    return x

# ---------- LaSPNet ----------
class LaSPNet(nn.Module):
    """
    Learnable anisotropic Screened-Poisson Network.
    - Always returns D_pred (Poisson 보정 결과).
    - 네트워크는 PDE 계수만 예측.
    """
    def __init__(self, dmax=10.0, ax_floor=1e-4, ay_floor=1e-4, lam_floor=1e-3, mu_floor=0.0,
                 max_global_delta=0.20, # trust region around P (m)
                 pcg_tol=1e-5, pcg_iters=200):
        super().__init__()
        self.dmax = float(dmax)
        self.enc  = TinyFeat(in_ch=3+1+1+1+1, ch=64)  # [RGB, Pn, E_norm, ML, edge(Pn)]
        self.head = AnisoHead(64, ax_floor, ay_floor, lam_floor, mu_floor)
        self.max_global_delta = float(max_global_delta)
        self.pcg_tol   = float(pcg_tol)
        self.pcg_iters = int(pcg_iters)

    @staticmethod
    def _edge01(x01: torch.Tensor) -> torch.Tensor:
        gx = F.pad(x01[..., :, 1:] - x01[..., :, :-1], (1,0,0,0))
        gy = F.pad(x01[..., 1:, :] - x01[..., :-1, :], (0,0,1,0))
        return (gx.abs() + gy.abs())*0.5

    @torch.no_grad()
    def _poisson_P(self, DL, ML, E01, poisson_fn):
        # 외부의 poisson_gpu(sparse_m, est_m, ...)를 그대로 사용
        B,_,H,W = DL.shape
        dev = DL.device
        out = []
        for b in range(B):
            est_m = (E01[b,0].clamp(0,1) * self.dmax).float().cpu().numpy()
            sparse = (DL[b,0] * (ML[b,0]>0).float()).float().cpu().numpy()
            P_np, _ = poisson_fn(sparse_m=sparse, est_m=est_m, tol=1e-6, maxiter=3000,
                                 device=str(dev), init="est", clip_to_max_gt=False)
            out.append(torch.from_numpy(P_np)[None,None].to(device=dev, dtype=torch.float32))
        return torch.cat(out, dim=0)

    def forward(self, I, DL, ML, E_norm, poisson_fn):
        """
        I: (B,3,H,W), DL/ML/E_norm: (B,1,H,W), poisson_fn: callable(sparse_m, est_m, ...)
        """
        I, DL, ML, E = I.float(), DL.float(), (ML>0).float(), E_norm.float()
        dmax = self.dmax

        # 1) Poisson 기반 초기치 P (metric)
        P = self._poisson_P(DL, ML, E, poisson_fn)  # (B,1,H,W)

        # 2) 특징 인코딩 (edge는 Pn에서)
        Pn = (P/dmax).clamp(0,1)
        edgeP = self._edge01(Pn)
        x = torch.cat([I, Pn, E, ML, edgeP], dim=1)
        feat = self.enc(x)

        # 3) PDE 계수 예측
        ax, ay, lam, mu = self.head(feat)   # (B,1,H,W) 각각

        # 4) 선형계 A u = b 구성 & PCG
        wE,wW,wS,wN = build_axis_aligned_weights(ax, ay)
        rhs = lam*P + (mu*ML)*DL
        u0  = P.clone()  # warm-start at Poisson
        D = pcg_solve(rhs, wE,wW,wS,wN, lam, muML=(mu*ML),
                      tol=self.pcg_tol, iters=self.pcg_iters, x0=u0)

        # 5) trust region: P 주변 소보정 강제 + 범위 클램프
        D = P + (D - P).clamp(-self.max_global_delta, self.max_global_delta)
        D = D.clamp(0, dmax)

        aux = {"P": P, "ax": ax, "ay": ay, "lam": lam, "mu": mu}
        return D, aux

        
# # -------------------- Model --------------------
# class MCPropNet(nn.Module):
#     def __init__(self, cfg: MCPropCfg):
#         super().__init__()
#         self.cfg = cfg

#         # -------- Poisson 옵션 (모델 내부 사용) --------
#         self.use_poisson = bool(getattr(cfg, "use_poisson", True))
#         self.poisson_tol = float(getattr(cfg, "poisson_tol", 1e-5))
#         self.poisson_maxiter = int(getattr(cfg, "poisson_maxiter", 1000))
#         self.poisson_init = str(getattr(cfg, "poisson_init", "est"))  # "est" | "zero"
#         self.poisson_clip_to_max_gt = bool(getattr(cfg, "poisson_clip_to_max_gt", False))

#         # -------- 네트워크 구성 --------
#         in_ch = 3 + 1 + 1 + 1  # RGB, P/dmax, E_norm, ML
#         self.enc = TinyFeat(in_ch, 64)
#         self.res = ResidualHead(64, 64) if cfg.use_residual else None
#         self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
#         self.gate = KernelGate(64, cfg.kernels)

#         if cfg.geometry.lower().startswith("ellip"):
#             self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
#         else:
#             self.aff_head = HCLApproxAffinity(64, cfg.kernels)

#         # Learnable anchor
#         self.anchor_learnable = bool(cfg.anchor_learnable)
#         self.anchor_mode = cfg.anchor_mode.lower()
#         self.anchor_init = float(cfg.anchor_alpha)
#         if self.anchor_learnable:
#             if self.anchor_mode == "scalar":
#                 self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
#                 self.anchor_head = None
#             elif self.anchor_mode == "map":
#                 self.anchor_param = None
#                 self.anchor_head = AnchorHead(64)
#             else:
#                 raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
#         else:
#             self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
#             self.anchor_param = None
#             self.anchor_head = None

#     def _alpha_map(self, feat, ML):
#         if self.anchor_learnable:
#             if self.anchor_mode == "scalar":
#                 a = torch.sigmoid(self.anchor_param)
#                 return a.view(1,1,1,1).expand_as(ML)
#             else:
#                 return self.anchor_head(feat)
#         else:
#             return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

#     @torch.no_grad()
#     def _poisson_batch(self, DL, ML, E01):
#         B = DL.shape[0]; dmax = float(self.cfg.dmax); dev = DL.device
#         dev_str = str(dev)
#         use_affine = bool(getattr(self.cfg, "poisson_est_affine", True))
#         use_auto_flip = bool(getattr(self.cfg, "poisson_auto_flip", True))

#         P_list, stats_list = [], []
#         for b in range(B):
#             e  = E01[b,0].detach().cpu().numpy().astype(np.float32)  # [0,1]
#             dl = (DL[b,0] / dmax).detach().cpu().numpy().astype(np.float32)  # metric→[0,1] 근사
#             m  = (ML[b,0].detach().cpu().numpy().astype(np.float32) > 0)

#             # --- 1) 방향 자동 판정 & 반전 ---
#             if use_auto_flip and m.sum() >= 10:
#                 corr = np.corrcoef(e[m].reshape(-1), dl[m].reshape(-1))[0,1]
#                 if np.isnan(corr) or np.isinf(corr): corr = 0.0
#                 if corr < 0:   # inverse-depth 계열 보정
#                     e = 1.0 - e

#             # --- 2) 강건 1D affine 보정 ---
#             if use_affine and m.sum() >= 10:
#                 x = e[m].reshape(-1,1); y = dl[m].reshape(-1,1)
#                 A = np.concatenate([x, np.ones_like(x)], axis=1)
#                 w = np.ones((A.shape[0],1), dtype=np.float32)
#                 for _ in range(3):
#                     Aw = A * w; yw = y * w
#                     theta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
#                     r = (A @ theta - y)
#                     c = 1.345 * np.median(np.abs(r)) + 1e-6
#                     w = (1.0 / np.maximum(1.0, np.abs(r)/c)).astype(np.float32)
#                 a, b0 = float(theta[0,0]), float(theta[1,0])
#                 e = np.clip(a*e + b0, 0.0, 1.0)

#             # --- 3) 경계 완화(Neumann 유사) & 미세 스무딩(양자화 완화) ---
#             # 간단한 3x3 box filter와 가장자리 reflect
#             if getattr(self.cfg, "poisson_smooth_est", True):
#                 pad = 1
#                 ep = np.pad(e, ((pad,pad),(pad,pad)), mode="reflect")
#                 ker = np.array([[1,1,1],[1,2,1],[1,1,1]], np.float32); ker /= ker.sum()
#                 ep = cv2.filter2D(ep, -1, ker) if 'cv2' in globals() else \
#                     np.convolve(ep.reshape(-1), np.ones(9)/9, 'same').reshape(ep.shape)
#                 e = ep[pad:-pad, pad:-pad]

#             est_m    = (e * dmax).astype(np.float32)
#             sparse_m = (DL[b,0].detach().cpu().numpy().astype(np.float32) * m.astype(np.float32))

#             P_np, st = poisson_gpu(
#                 sparse_m=sparse_m, est_m=est_m,
#                 tol=self.poisson_tol, maxiter=self.poisson_maxiter,
#                 device=dev_str, init=self.poisson_init,
#                 clip_to_max_gt=self.poisson_clip_to_max_gt
#             )
#             P_list.append(torch.from_numpy(P_np)[None, None])
#             stats_list.append(st)

#         P = torch.cat(P_list, dim=0).to(device=dev, dtype=torch.float32)
#         return P, stats_list

#     def forward(self, I, DL, ML, E_norm):
#         """
#         전처리 가정(데이터로더에서 수행됨):
#         - I:      (B,3,H,W) float, [0,1] 또는 (선택) mean/std 정규화된 텐서
#         - DL:     (B,1,H,W) float, metric depth
#         - ML:     (B,1,H,W) float, {0,1}
#         - E_norm: (B,1,H,W) float, [0,1] (mono_scale=255 로 스케일링되었다고 가정)
#         - 위 모든 텐서는 동일한 (H,W)
#         """
#         # 타입/차원만 보정 (스케일/리사이즈 없음)
#         if I.ndim == 3:  I = I.unsqueeze(0)
#         if DL.ndim == 3: DL = DL.unsqueeze(0)
#         if ML.ndim == 3: ML = ML.unsqueeze(0)
#         if E_norm.ndim == 3: E_norm = E_norm.unsqueeze(0)

#         I  = I.float()
#         DL = DL.float()
#         ML = (ML > 0).float()
#         E  = E_norm.float()

#         # 모든 모달리티가 같은 H,W 인지 확인(전처리는 데이터로더 책임)
#         H, W = I.shape[-2:]
#         assert DL.shape[-2:] == (H, W) and ML.shape[-2:] == (H, W) and E.shape[-2:] == (H, W), \
#             f"[MCPropNet] All inputs must share the same (H,W). Got I={I.shape[-2:]}, DL={DL.shape[-2:]}, ML={ML.shape[-2:]}, E={E.shape[-2:]}"

#         # mono는 [0,1] 가정
#         if torch.any((E < -1e-6) | (E > 1.0 + 1e-6)):
#             raise ValueError("[MCPropNet] E_norm must be in [0,1]. Please scale mono in DataLoader (e.g., mono_scale=255).")

#         # ---- Poisson으로 P 생성 (또는 비활성 시 단순 스케일) ----
#         if getattr(self, "use_poisson", True):
#             P, poisson_stats = self._poisson_batch(DL, ML, E)   # metric depth
#         else:
#             P, poisson_stats = (E * self.cfg.dmax).clamp(0, self.cfg.dmax), None

#         # 인코더 입력 (P는 metric → /dmax로 정규화)
#         x_in = torch.cat([I, P / self.cfg.dmax, E, ML], dim=1)
#         feat = self.enc(x_in)

#         # 초기값 D0 = P (+ residual 보정)
#         res_out = None
#         if self.res is not None:
#             res_out = self.res(feat)                 # unclamped residual (B,1,H,W)
#             D0 = (P + res_out).clamp(0, self.cfg.dmax)
#         else:
#             D0 = P

#         # curvature & affinity
#         kappa, scale, bias = self.curv(feat, E)
#         if self.cfg.geometry.lower().startswith("ellip"):
#             A_list_raw = self.aff_head(feat, kappa)
#         else:
#             A_list_raw = self.aff_head(feat, scale, bias)
#         A_list = normalize_affinity_list(A_list_raw)  # per‑kernel normalized
#         sigma  = self.gate(feat)                      # (B,K,H,W)
#         alpha  = self._alpha_map(feat, ML)            # (B,1,H,W)

#         # -------- multi‑kernel propagation --------
#         Dt = D0.clone()
#         for _ in range(self.cfg.steps):
#             mix = torch.zeros_like(Dt)
#             for idx, k in enumerate(self.cfg.kernels):
#                 Ak = A_list[idx]                    # (B, k*k, H, W)
#                 kk = k * k
#                 patches = unfold_neighbors(Dt, k)   # (B, kk, H, W)
#                 center = kk // 2
#                 patches_center = patches.clone()
#                 patches_center[:, center:center+1, :, :] = D0  # center ← D0
#                 Dk = (Ak * patches_center).sum(1, keepdim=True)
#                 mix = mix + sigma[:, idx:idx+1] * Dk

#             if self.cfg.use_sparse:
#                 Dt = (1.0 - alpha * ML) * mix + (alpha * ML) * DL
#             else:
#                 Dt = mix
#             Dt = Dt.clamp(0, self.cfg.dmax)

#         # ---- 반환: 기존 키 + P와 residual을 추가 ----
#         aux = {
#             "D0": D0,                      # 초기 보정 깊이 (P+residual, clamp 후)
#             "P": P,                        # Poisson/anchor 깊이 (metric)
#             "residual": res_out,           # ResidualHead 출력 (clamp 전), 없으면 None
#             "sigma": sigma,                # 커널 게이트
#             "alpha": alpha,                # 앵커/스파스 혼합 계수
#             "poisson_stats": poisson_stats # (선택) 시간/반복 등 통계
#         }
#         return Dt, aux


class MCPropNetMSHR(nn.Module):
    """
    Multi-Scale Hierarchical Refinement version of MCPropNet.
    - 스케일 피라미드: 기본 (4x down) → (2x down) → (1x, full)
    - 각 스케일에서 Poisson으로 P_s 생성 후, propagation으로 정련
    - 다음(더 고해상도) 스케일의 초기값은 upsample된 이전 예측과 P_{s+1}를 edge-aware blend로 구성
    - 최종 출력은 full-resolution Dt
    - ResidualHead는 보유하되, 기본 cfg.use_residual=False로 비활성 권장 (NYUv2)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reg_sparse_drop_p     = float(getattr(cfg, "reg_sparse_drop_p", 0.10))  # 0~1, 스파스 점 일부 드롭
        self.reg_sparse_noise_std  = float(getattr(cfg, "reg_sparse_noise_std", 0.01)) # [m], DL에 미세 노이즈
        self.reg_gate_drop_p       = float(getattr(cfg, "reg_gate_drop_p", 0.10))  # 0~1, 게이트 드롭커넥트
        # -------- Poisson 옵션 --------
        self.use_poisson = bool(getattr(cfg, "use_poisson", True))
        self.poisson_tol = float(getattr(cfg, "poisson_tol", 1e-5))
        self.poisson_maxiter = int(getattr(cfg, "poisson_maxiter", 1000))
        self.poisson_init = str(getattr(cfg, "poisson_init", "est"))  # "est" | "zero"
        self.poisson_clip_to_max_gt = bool(getattr(cfg, "poisson_clip_to_max_gt", False))

        # -------- 다중 스케일 옵션 --------
        # 배율 목록: 큰 수가 coarse (예: [4,2,1])
        self.mshr_scales = tuple(getattr(cfg, "mshr_scales", (4, 2, 1)))
        assert len(self.mshr_scales) >= 1 and all(isinstance(s, int) and s >= 1 for s in self.mshr_scales)
        # 스케일별 propagation 스텝: None이면 cfg.steps 동일 적용
        self.mshr_steps = getattr(cfg, "mshr_steps", None)  # None | int | list/tuple[int,...]
        # coarse 예측과 현 스케일 Poisson을 섞는 블렌드 계수(상수)
        self.mshr_blend = float(getattr(cfg, "mshr_blend", 0.7))  # upsample(prev) 비중
        # 에지 기반 가변 블렌드 사용 여부 및 민감도
        self.mshr_edge_blend = bool(getattr(cfg, "mshr_edge_blend", True))
        self.mshr_edge_tau = float(getattr(cfg, "mshr_edge_tau", 0.10))  # E_norm 그래디언트 임계 (0~1)
        self.mshr_edge_gamma = float(getattr(cfg, "mshr_edge_gamma", 1.0))  # 가중 강조

        # -------- 네트워크 구성 --------
        in_ch = 3 + 1 + 1 + 1  # RGB, P/dmax, E_norm, ML
        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if getattr(cfg, "use_residual", False) else None
        self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # Learnable anchor
        self.anchor_learnable = bool(cfg.anchor_learnable)
        self.anchor_mode = cfg.anchor_mode.lower()
        self.anchor_init = float(cfg.anchor_alpha)
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(64)
            else:
                raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
        else:
            self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
            self.anchor_param = None
            self.anchor_head = None
    @staticmethod
    def _sparse_dropout_noise(DL, ML, p_drop: float, noise_std: float):
        """
        훈련 시 스파스 포인트 일부 드롭 + DL에 미세 노이즈 추가.
        DL, ML: (B,1,H,W)
        """
        if p_drop > 0.0:
            drop = (torch.rand_like(ML) < p_drop) & (ML > 0)
            if drop.any():
                DL = DL.clone(); ML = ML.clone()
                DL[drop] = 0.0
                ML[drop] = 0.0
        if noise_std > 0.0:
            noise = torch.randn_like(DL) * noise_std
            DL = DL + noise * (ML > 0).float()
        return DL, ML

    def _gate_dropconnect(self, sigma):
        """
        게이트 드롭커넥트(채널별). 학습시에만 mask 적용, 기대값 유지되도록 /keep 보정.
        sigma: (B,K,H,W)
        """
        p = float(self.reg_gate_drop_p)
        if not self.training or p <= 0.0:
            return sigma
        B, K, Hs, Ws = sigma.shape
        keep = 1.0 - p
        mask = torch.bernoulli(torch.full((B, K, 1, 1), keep,
                        device=sigma.device, dtype=sigma.dtype))
        return sigma * mask / keep
    # ---------- helpers ----------
    def _alpha_map(self, feat, ML):
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1,1,1,1).expand_as(ML)
            else:
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    @torch.no_grad()
    def _poisson_batch(self, DL, ML, E01):
        """
        DL: (B,1,H,W) metric sparse depth
        ML: (B,1,H,W) mask (0/1)
        E01: (B,1,H,W) monocular normalized [0,1]
        return: P (B,1,H,W), stats_list
        """
        B = DL.shape[0]
        dmax = float(self.cfg.dmax)
        dev = DL.device
        dev_str = str(dev)

        P_list, stats_list = [], []
        for b in range(B):
            sparse_m = (DL[b,0] * (ML[b,0] > 0).float()).detach().cpu().numpy().astype(np.float32)
            est_m    = (E01[b,0] * dmax).detach().cpu().numpy().astype(np.float32)

            P_np, st = poisson_gpu(
                sparse_m=sparse_m, est_m=est_m,
                tol=self.poisson_tol, maxiter=self.poisson_maxiter,
                device=dev_str, init=self.poisson_init,
                clip_to_max_gt=self.poisson_clip_to_max_gt
            )
            P_list.append(torch.from_numpy(P_np)[None, None])
            stats_list.append(st)

        P = torch.cat(P_list, dim=0).to(device=dev, dtype=torch.float32)
        return P, stats_list

    @staticmethod
    def _edge_mag_01(E01):
        """ 간단한 1차 차분 기반 그래디언트 크기 (0~1 근사 정규화). """
        # E01: (B,1,H,W) in [0,1]
        gx = F.pad(E01[..., :, 1:] - E01[..., :, :-1], (1,0,0,0))
        gy = F.pad(E01[..., 1:, :] - E01[..., :-1, :], (0,0,1,0))
        g = (gx.abs() + gy.abs()) * 0.5
        # 약한 정규화: 상한 클리핑
        return g.clamp(0, 1)

    @staticmethod
    def _downsample_rgb(I, H, W):
        return F.interpolate(I, size=(H, W), mode="bilinear", align_corners=False)

    @staticmethod
    def _downsample_mono(E, H, W):
        E_s = F.interpolate(E, size=(H, W), mode="bilinear", align_corners=False)
        return E_s.clamp(0.0, 1.0)

    @staticmethod
    def _downsample_sparse(DL, ML, factor):
        """
        DL, ML: (B,1,H,W)
        factor: int
        Returns DL_s, ML_s pooled to (H//factor, W//factor).
        DL_s는 윈도 내 유효 포인트 평균(없으면 0), ML_s는 MaxPool로 집계.
        """
        if factor == 1:
            return DL, ML
        k = factor
        # count = sum(ML), sum_DL = sum(DL*ML)
        sum_DL = F.avg_pool2d(DL * ML, kernel_size=k, stride=k) * (k * k)
        count  = F.avg_pool2d(ML,     kernel_size=k, stride=k) * (k * k)
        ML_s = (count > 0).float()
        DL_s = torch.where(count > 0, sum_DL / (count + 1e-6), torch.zeros_like(sum_DL))
        return DL_s, ML_s

    @staticmethod
    def _resize_like(x, H, W, mode="bilinear", is_mask=False):
        if is_mask:
            # mask는 보간 대신 최근접
            return F.interpolate(x, size=(H, W), mode="nearest")
        if mode == "bilinear":
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        elif mode == "nearest":
            return F.interpolate(x, size=(H, W), mode="nearest")
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _steps_for_scale(self, idx):
        """ 스케일 인덱스별 propagation 스텝 수를 반환. """
        if self.mshr_steps is None:
            return int(self.cfg.steps)
        if isinstance(self.mshr_steps, int):
            return int(self.mshr_steps)
        if isinstance(self.mshr_steps, (list, tuple)):
            if idx < len(self.mshr_steps):
                return int(self.mshr_steps[idx])
            return int(self.mshr_steps[-1])
        return int(self.cfg.steps)

    # ---------- forward ----------
    def forward(self, I, DL, ML, E_norm):
        """
        입력:
          - I:      (B,3,H,W) float, [0,1] 또는 mean/std 정규화
          - DL:     (B,1,H,W) float, metric sparse depth
          - ML:     (B,1,H,W) float, {0,1}
          - E_norm: (B,1,H,W) float, [0,1]
        출력:
          - Dt:  (B,1,H,W) 최종 깊이 (metric)
          - aux: 부가 정보(dict). multi-scale 통계 포함
        """
        # 차원 정리
        if I.ndim == 3:  I = I.unsqueeze(0)
        if DL.ndim == 3: DL = DL.unsqueeze(0)
        if ML.ndim == 3: ML = ML.unsqueeze(0)
        if E_norm.ndim == 3: E_norm = E_norm.unsqueeze(0)

        I  = I.float()
        DL = DL.float()
        ML = (ML > 0).float()
        E  = E_norm.float()

        # 크기 확인
        H, W = I.shape[-2:]
        assert DL.shape[-2:] == (H, W) and ML.shape[-2:] == (H, W) and E.shape[-2:] == (H, W), \
            f"[MCPropNetMSHR] All inputs must share (H,W). I={I.shape[-2:]}, DL={DL.shape[-2:]}, ML={ML.shape[-2:]}, E={E.shape[-2:]}"

        if torch.any((E < -1e-6) | (E > 1.0 + 1e-6)):
            raise ValueError("[MCPropNetMSHR] E_norm must be in [0,1]. Please scale mono in DataLoader.")

        dmax = float(self.cfg.dmax)
        scales = list(self.mshr_scales)  # e.g., [4,2,1]
        # coarse -> fine 순회
        scales = sorted(scales, reverse=True)

        Dt_prev = None
        per_scale = []     # 각 스케일 결과/통계 저장
        poisson_all = []   # Poisson 통계

        for si, s in enumerate(scales):
            Hs, Ws = (H // s), (W // s)
            # 1) 다운샘플
            I_s  = self._downsample_rgb(I,  Hs, Ws)
            E_s  = self._downsample_mono(E, Hs, Ws)
            DL_s, ML_s = self._downsample_sparse(DL, ML, s)
            if self.training and (self.reg_sparse_drop_p > 0.0 or self.reg_sparse_noise_std > 0.0):
                DL_s, ML_s = self._sparse_dropout_noise(
                    DL_s, ML_s,
                    p_drop=self.reg_sparse_drop_p,
                    noise_std=self.reg_sparse_noise_std
                )
            # 2) Poisson으로 P_s
            if self.use_poisson:
                P_s, st = self._poisson_batch(DL_s, ML_s, E_s)   # metric
            else:
                P_s, st = (E_s * dmax).clamp(0, dmax), None
            poisson_all.append(st)

            # 3) 초기값 D0_s: (coarse는 P_s, 그 외는 upsample(prev)와 blend)
            if Dt_prev is None:
                D0_s = P_s
            else:
                D_up = self._resize_like(Dt_prev, Hs, Ws, mode="bilinear")
                if self.mshr_edge_blend:
                    # edge-aware weight w: 에지에서 P_s 비중↑ (D_up 비중↓)
                    gE = self._edge_mag_01(E_s)           # (B,1,Hs,Ws) in [0,1]
                    # w_up = mshr_blend * (1 - norm_grad), 지수 감쇠 옵션
                    w_up = self.mshr_blend * torch.pow((1.0 - (gE / max(self.mshr_edge_tau, 1e-6)).clamp(0,1)), self.mshr_edge_gamma)
                    D0_s = w_up * D_up + (1.0 - w_up) * P_s
                else:
                    D0_s = self.mshr_blend * D_up + (1.0 - self.mshr_blend) * P_s

            # 4) 인코더 & 파라메터 맵 산출 (P_s는 /dmax로 정규화하여 입력)
            x_in = torch.cat([I_s, P_s / dmax, E_s, ML_s], dim=1)
            feat = self.enc(x_in)

            # curvature & affinity
            kappa, scale, bias = self.curv(feat, E_s)
            if self.cfg.geometry.lower().startswith("ellip"):
                A_list_raw = self.aff_head(feat, kappa)
            else:
                A_list_raw = self.aff_head(feat, scale, bias)
            A_list = normalize_affinity_list(A_list_raw)  # per-kernel normalized
            sigma  = self.gate(feat)                      # (B,K,Hs,Ws)
            if self.training and self.reg_gate_drop_p > 0.0:
                sigma = self._gate_dropconnect(sigma)
            alpha  = self._alpha_map(feat, ML_s)          # (B,1,Hs,Ws)

            # 5) propagation (Dt 초기값은 D0_s)
            Dt_s = D0_s.clone()
            steps = self._steps_for_scale(si)
            for _ in range(steps):
                mix = torch.zeros_like(Dt_s)
                for idx, k in enumerate(self.cfg.kernels):
                    Ak = A_list[idx]                      # (B, k*k, Hs, Ws)
                    kk = k * k
                    patches = unfold_neighbors(Dt_s, k)   # (B, kk, Hs, Ws)
                    center = kk // 2
                    patches_center = patches.clone()
                    patches_center[:, center:center+1, :, :] = D0_s  # center ← D0_s (고정 중심)
                    Dk = (Ak * patches_center).sum(1, keepdim=True)
                    mix = mix + sigma[:, idx:idx+1] * Dk

                if self.cfg.use_sparse:
                    Dt_s = (1.0 - alpha * ML_s) * mix + (alpha * ML_s) * DL_s
                else:
                    Dt_s = mix
                Dt_s = Dt_s.clamp(0, dmax)

            # 6) 다음 스케일 준비
            Dt_prev = Dt_s
            per_scale.append({
                "scale": s,
                "H": Hs, "W": Ws,
                "P": P_s, "D0": D0_s, "Dt": Dt_s,
                "sigma": sigma, "alpha": alpha
            })

        # 최종 출력: full scale (s==1) Dt
        Dt = Dt_prev
        # 상위 해상도로 맞춰져 있을 것이지만, 혹시 마지막 스케일이 1이 아니었다면 보간
        if Dt.shape[-2:] != (H, W):
            Dt = self._resize_like(Dt, H, W, mode="bilinear")

        # ---- 반환 ----
        aux = {
            "hier": per_scale,             # 각 스케일의 P, D0, Dt 등
            "poisson_stats": poisson_all,  # 스케일별 Poisson 통계
        }
        return Dt, aux
    

def _align_hw_like(x: torch.Tensor, ref: torch.Tensor, mode="bilinear"):
    """x를 ref의 (H,W)에 맞춤. 이미 같으면 그대로 반환."""
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == "bilinear":
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode)

class MCPropNetv2(nn.Module):
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3 + 1 + 1 + 1  # RGB, P/dmax, E_norm, ML

        self.enc = TinyFeat(in_ch, 64)  # (ResFp로 교체 시에도 하위 가드가 보호)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None
        self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # (권장) 홀수 커널 보장
        assert all(k % 2 == 1 for k in cfg.kernels), "All kernels must be odd (3,5,7,...) for center replacement to be valid."

        # Learnable anchor
        self.anchor_learnable = bool(cfg.anchor_learnable)
        self.anchor_mode = cfg.anchor_mode.lower()
        self.anchor_init = float(cfg.anchor_alpha)
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(64)
            else:
                raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
        else:
            self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
            self.anchor_param = None
            self.anchor_head = None

    def _alpha_map(self, feat, ML):
        """α map in [0,1], (B,1,H,W)"""
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1,1,1,1).expand_as(ML)
            else:
                return self.anchor_head(feat)  # AnchorHead 안에서 sigmoid가 적용되는지 확인 권장
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    def forward(self, I, DL, ML, P, E_norm):
        cfg = self.cfg

        # --- Robust input normalization to (B,C,H,W) ---
        I  = _as_chw4(I.float())
        DL = _as_1ch4(DL.float())
        ML = _as_1ch4(ML.float())
        P  = _as_1ch4(P.float())
        E  = _as_1ch4(E_norm.float())

        # 조건부 스케일링 (이미 [0,1]이면 보존)
        I01 = I / 255.0 if I.max() > 1.5 else I
        if E.max() > 1.5:
            E = E / 255.0

        # 해상도 정렬: RGB 기준
        DL  = _resize_like(DL, I, mode="bilinear")
        ML  = _resize_like((ML > 0).float(), I, mode="nearest")
        P   = _resize_like(P,  I, mode="bilinear")
        E   = _resize_like(E,  I, mode="bilinear")

        x_in = torch.cat([I01, P/cfg.dmax, E, ML], dim=1)
        feat = self.enc(x_in)

        # (중요) 인코더가 stride/downsample을 쓴 경우를 대비해 정렬
        feat = _align_hw_like(feat, I, mode="bilinear")

        # Residual prior
        if self.res is not None:
            res = self.res(feat)             # 구현에 따라 1ch 보장 필요
            res = _align_hw_like(res, P)     # 혹시 모를 크기 불일치 방지
            D0  = (P + res).clamp(0, cfg.dmax)
        else:
            D0  = P

        # curvature & affinity
        kappa, scale, bias = self.curv(feat, E)
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)
        else:
            A_list_raw = self.aff_head(feat, scale, bias)
        A_list = normalize_affinity_list(A_list_raw)  # per-kernel normalized
        sigma  = self.gate(feat)                      # (B,K,H,W)

        alpha = self._alpha_map(feat, ML)             # (B,1,H,W)

        # multi‑kernel propagation
        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix = torch.zeros_like(Dt)
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]                     # (B, k*k, H, W)
                kk = k * k
                patches = unfold_neighbors(Dt, k)    # (B, kk, H, W)
                center = kk // 2

                # 메모리 복제 없이 "center ← D0" 효과 주기
                # Dk = sum(A * patches) + w_center * (D0 - patches_center)
                w_c = Ak[:, center:center+1, :, :]             # (B,1,H,W)
                p_c = patches[:, center:center+1, :, :]        # (B,1,H,W)
                Dk  = (Ak * patches).sum(1, keepdim=True) + w_c * (D0 - p_c)

                mix = mix + sigma[:, idx:idx+1, :, :] * Dk     # (B,1,H,W)

            if cfg.use_sparse:
                Dt = (1.0 - alpha * ML) * mix + (alpha * ML) * DL
            else:
                Dt = mix

            Dt = Dt.clamp(0, cfg.dmax)

        aux = {"D0": D0, "sigma": sigma, "alpha": alpha}
        return Dt, aux
# class MCPropNetv2(nn.Module):
#     """
#     Multi‑kernel Curvature‑guided Propagation Network with ResFp encoder.

#     입력 채널: [RGB, P/dmax, E_norm, ML] = 6ch
#     인코더: ResFp (ResNet-FPN 기반)  → enc_ch (= cfg.enc_out_ch or 128)
#     헤드:
#       - ResidualHead(enc_ch, 64)  -> D0 = P + residual
#       - CurvatureGen(enc_ch + 1, ...) -> (kappa, scale, bias)
#       - KernelGate(enc_ch, K) -> sigma (B,K,H,W)
#       - EllipticAffinity(enc_ch, K, ...) or HCLApproxAffinity(enc_ch, K, ...)

#     forward:
#       1) 입력 정규화/리사이즈
#       2) feat = enc([I/255, P/dmax, E, ML])
#       3) D0 = P (+ residual)
#       4) A_list, sigma, alpha 계산
#       5) cfg.steps 만큼 multi-kernel propagation
#       6) 최종 Dt 반환 (+ aux 로깅)
#     """

#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg

#         # ----- Encoder -----
#         in_ch = 3 + 1 + 1 + 1  # [RGB, P/dmax, E_norm, ML]
#         enc_out_ch = getattr(cfg, "enc_out_ch", 128)  # 기본 128
#         # ResFp는 사용자가 이미 정의한 인코더여야 합니다.
#         # 시그니처 예: ResFp(in_ch=6, out_ch=128, ...)
#         self.enc = ResFPNFeat(in_ch=in_ch, out_ch=enc_out_ch)
#         # 일부 구현에서는 self.enc.out_ch 속성을 제공합니다.
#         self.enc_ch = getattr(self.enc, "out_ch", enc_out_ch)

#         # ----- Residual head (선택) -----
#         self.res = ResidualHead(self.enc_ch, 64) if getattr(cfg, "use_residual", False) else None

#         # ----- Curvature / Gate / Affinity heads -----
#         self.curv = CurvatureGen(self.enc_ch + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
#         self.gate = KernelGate(self.enc_ch, cfg.kernels)

#         # Geometry-specific affinity
#         c_aff = getattr(cfg, "c_aff", 32)
#         tau_min = getattr(cfg, "tau_min", 0.03)
#         tau_max = getattr(cfg, "tau_max", 0.5)
#         if cfg.geometry.lower().startswith("ellip"):
#             self.aff_head = EllipticAffinity(self.enc_ch, cfg.kernels, c_aff=c_aff, tau_min=tau_min, tau_max=tau_max)
#         else:
#             self.aff_head = HCLApproxAffinity(self.enc_ch, cfg.kernels)

#         # ----- Learnable anchor -----
#         self.anchor_learnable = bool(cfg.anchor_learnable)
#         self.anchor_mode = cfg.anchor_mode.lower()
#         self.anchor_init = float(cfg.anchor_alpha)
#         if self.anchor_learnable:
#             if self.anchor_mode == "scalar":
#                 self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
#                 self.anchor_head = None
#             elif self.anchor_mode == "map":
#                 self.anchor_param = None
#                 self.anchor_head = AnchorHead(self.enc_ch)
#             else:
#                 raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
#         else:
#             self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
#             self.anchor_param = None
#             self.anchor_head = None

#     # --- internal: alpha map ---
#     def _alpha_map(self, feat, ML):
#         """α map in [0,1], (B,1,H,W)"""
#         if self.anchor_learnable:
#             if self.anchor_mode == "scalar":
#                 a = torch.sigmoid(self.anchor_param)
#                 return a.view(1,1,1,1).expand_as(ML)
#             else:  # "map"
#                 return self.anchor_head(feat)
#         else:
#             return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

#     def forward(self, I, DL, ML, P, E_norm):
#         cfg = self.cfg

#         # --- Robust input normalization to (B,C,H,W) ---
#         I  = _as_chw4(I.float())
#         DL = _as_1ch4(DL.float())
#         ML = _as_1ch4(ML.float())
#         P  = _as_1ch4(P.float())
#         E  = _as_1ch4(E_norm.float())

#         # E가 0~255 범위일 가능성 보정
#         if E.max() > 1.5:
#             E = E / 255.0

#         # 해상도 정렬: RGB 기준
#         I01 = I / 255.0
#         DL  = _resize_like(DL, I, mode="bilinear")
#         ML  = _resize_like((ML > 0).float(), I, mode="nearest")
#         P   = _resize_like(P,  I, mode="bilinear")
#         E   = _resize_like(E,  I, mode="bilinear")

#         # ----- Encoder -----
#         x_in = torch.cat([I01, P/cfg.dmax, E, ML], dim=1)  # [B,6,H,W]
#         feat = self.enc(x_in)                              # [B,enc_ch,H,W]

#         # ----- Residual depth prior -----
#         if self.res is not None:
#             D0 = (P + self.res(feat)).clamp(0, cfg.dmax)
#         else:
#             D0 = P

#         # ----- Curvature & Affinity & Gate -----
#         kappa, scale, bias = self.curv(feat, E)  # shape 규약은 기존 구현 따름
#         if cfg.geometry.lower().startswith("ellip"):
#             A_list_raw = self.aff_head(feat, kappa)
#         else:
#             A_list_raw = self.aff_head(feat, scale, bias)

#         # per-kernel 정규화된 affinity
#         A_list = normalize_affinity_list(A_list_raw)  # List[ (B, k*k, H, W) ], len=K
#         sigma  = self.gate(feat)                      # (B, K, H, W), 커널 게이트

#         # ----- Anchor (alpha) -----
#         alpha = self._alpha_map(feat, ML)             # (B,1,H,W)

#         # ----- Multi‑kernel propagation -----
#         Dt = D0.clone()
#         for _ in range(cfg.steps):
#             mix = torch.zeros_like(Dt)
#             for idx, k in enumerate(cfg.kernels):
#                 Ak = A_list[idx]                      # (B, k*k, H, W)
#                 kk = k * k
#                 patches = unfold_neighbors(Dt, k)     # (B, kk, H, W)

#                 # center 픽셀을 D0로 고정 (prior 보존)
#                 center = kk // 2
#                 patches_center = patches.clone()
#                 patches_center[:, center:center+1, :, :] = D0

#                 # 가중 합
#                 Dk = (Ak * patches_center).sum(1, keepdim=True)  # (B,1,H,W)
#                 mix = mix + sigma[:, idx:idx+1] * Dk             # 게이트 혼합

#             if getattr(cfg, "use_sparse", False):
#                 # sparse 관측이 있을 때(ML>0) DL을 anchor로 사용
#                 Dt = (1.0 - alpha * ML) * mix + (alpha * ML) * DL
#             else:
#                 Dt = mix

#             Dt = Dt.clamp(0, cfg.dmax)

#         aux = {
#             "D0": D0,           # 초기 prior
#             "sigma": sigma,     # 커널 게이트
#             "alpha": alpha,     # 앵커 맵
#             # 필요시 디버깅용 정보 추가:
#             # "kappa": kappa, "scale": scale, "bias": bias, "A_list": A_list
#         }
#         return Dt, aux