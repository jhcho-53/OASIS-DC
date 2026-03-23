"""
OASIS-DC architecture for depth completion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

from ..base import TinyFeat, ResidualHead, CurvatureGen, KernelGate, AnchorHead
from ..affinity import HCLApproxAffinity, EllipticAffinity, normalize_affinity_list
from ..poisson import poisson_gpu
from ..utils import _as_chw4, _as_1ch4, _resize_like


class OASIS_DC(nn.Module):
    """
    OASIS-DC: Metric Completion via Propagation Network
    
    Combines monocular foundation priors with sparse anchors using
    affinity propagation and Poisson equation solving.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Configuration parsing
        self._parse_config(cfg)
        
        # Build network components
        self._build_encoders()
        self._build_affinity_modules()
        self._build_heads()
        self._build_anchor_system()

    def _parse_config(self, cfg):
        """Parse configuration parameters."""
        # Poisson solver settings
        self.poisson_only = bool(getattr(cfg, "poisson_only", False))
        self.use_poisson = bool(getattr(cfg, "use_poisson", True))
        self.poisson_tol = float(getattr(cfg, "poisson_tol", 1e-5))
        self.poisson_maxiter = int(getattr(cfg, "poisson_maxiter", 1000))
        self.poisson_init = str(getattr(cfg, "poisson_init", "est"))
        self.poisson_clip_to_max_gt = bool(getattr(cfg, "poisson_clip_to_max_gt", False))
        self.poisson_auto_flip = bool(getattr(cfg, "poisson_auto_flip", True))
        self.poisson_est_affine = bool(getattr(cfg, "poisson_est_affine", True))
        self.poisson_smooth_est = bool(getattr(cfg, "poisson_smooth_est", True))
        
        # Affinity settings
        self.use_p_affinity = bool(getattr(cfg, "use_p_affinity", True))
        self.p_only_gate = bool(getattr(cfg, "p_only_gate", False))
        self.geometry = str(getattr(cfg, "geometry", "hyper")).lower()
        
        # Architecture settings
        self.use_residual = bool(getattr(cfg, "use_residual", True))
        self.kernels = getattr(cfg, "kernels", (3, 5))
        self.steps = int(getattr(cfg, "steps", 8))
        
        # Anchor settings
        self.anchor_learnable = bool(getattr(cfg, "anchor_learnable", False))
        self.anchor_mode = str(getattr(cfg, "anchor_mode", "map")).lower()
        self.anchor_init = float(getattr(cfg, "anchor_alpha", 0.1))
        
        # Stability settings
        self.min_gate = float(getattr(cfg, "min_gate", 0.0))
        self.min_alpha = float(getattr(cfg, "min_alpha", 0.0))

    def _build_encoders(self):
        """Build feature encoders."""
        # Main encoder: [RGB(3), P/dmax(1), E(1), ML(1)]
        in_ch = 3 + 1 + 1 + 1  
        self.enc = TinyFeat(in_ch, 64)
        
        # P-only encoder for affinity: [Pn, edge(P), ML]
        self.p_in_ch = 3
        if self.use_p_affinity or self.p_only_gate:
            self.penc = TinyFeat(self.p_in_ch, 64)
        else:
            self.penc = None

    def _build_affinity_modules(self):
        """Build affinity computation modules."""
        self.curv = CurvatureGen(
            in_ch=65,  # feat(64) + E_norm(1)
            K=self.kernels,
            kappa_min=getattr(self.cfg, "kappa_min", 0.03),
            kappa_max=getattr(self.cfg, "kappa_max", 0.5)
        )
        
        self.gate = KernelGate(64, self.kernels)
        
        # Choose affinity computation method
        if self.geometry.startswith("ellip"):
            self.aff_head = EllipticAffinity(
                in_ch=64, 
                K=self.kernels, 
                c_aff=32, 
                tau_min=0.03, 
                tau_max=0.5
            )
        else:
            self.aff_head = HCLApproxAffinity(64, self.kernels)

    def _build_heads(self):
        """Build prediction heads."""
        if self.use_residual:
            self.res = ResidualHead(64, 64)
        else:
            self.res = None

    def _build_anchor_system(self):
        """Build anchor/Dirichlet boundary system."""
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                self.anchor_param = nn.Parameter(
                    torch.tensor(self.anchor_init, dtype=torch.float32)
                )
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(
                    in_ch=64, 
                    mode="map", 
                    learnable=True, 
                    init_alpha=self.anchor_init
                )
            else:
                raise ValueError(f"Unknown anchor_mode: {self.anchor_mode}")
        else:
            self.register_buffer(
                "anchor_fixed", 
                torch.tensor(self.anchor_init, dtype=torch.float32)
            )
            self.anchor_param = None
            self.anchor_head = None

    def _alpha_map(self, feat, ML):
        """Get alpha (Dirichlet strength) map."""
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1, 1, 1, 1).expand_as(ML)
            else:
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1, 1, 1, 1).expand_as(ML)

    @staticmethod
    def _edge_mag_01(X01: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude in [0,1] range."""
        gx = F.pad(X01[..., :, 1:] - X01[..., :, :-1], (1, 0, 0, 0))
        gy = F.pad(X01[..., 1:, :] - X01[..., :-1, :], (0, 0, 1, 0))
        g = (gx.abs() + gy.abs()) * 0.5
        return g.clamp(0, 1)

    @staticmethod 
    def _smooth3_reflect_np(e: np.ndarray) -> np.ndarray:
        """Apply 3x3 smoothing with reflection padding."""
        pad = 1
        ep = np.pad(e, ((pad, pad), (pad, pad)), mode="reflect")
        out = (
            ep[0:-2, 0:-2] + ep[0:-2, 1:-1] + ep[0:-2, 2:] +
            ep[1:-1, 0:-2] + 2*ep[1:-1, 1:-1] + ep[1:-1, 2:] +
            ep[2:, 0:-2] + ep[2:, 1:-1] + ep[2:, 2:]
        ) / 10.0
        return out.astype(np.float32)

    @torch.no_grad()
    def _poisson_batch(self, DL, ML, E01):
        """Apply Poisson solving to batch."""
        B = DL.shape[0]
        dmax = float(self.cfg.dmax)
        dev = DL.device
        dev_str = str(dev)
        P_list, stats_list = [], []

        for b in range(B):
            e = E01[b, 0].detach().cpu().numpy().astype(np.float32)  # [0,1]
            dl = DL[b, 0].detach().cpu().numpy().astype(np.float32)
            ml = ML[b, 0].detach().cpu().numpy().astype(np.float32)

            # Convert to metric scale
            e_metric = e * dmax
            dl_metric = dl * dmax
            sparse_metric = dl_metric * (ml > 0.5)

            # Optional smoothing and affine adjustment
            if self.poisson_smooth_est:
                e_metric = self._smooth3_reflect_np(e_metric)
                
            if self.poisson_est_affine:
                # Simple affine correction using sparse points
                valid_mask = ml > 0.5
                if np.sum(valid_mask) > 10:  # Enough points for fitting
                    valid_sparse = sparse_metric[valid_mask]
                    valid_est = e_metric[valid_mask]
                    
                    # Least squares: est * scale + bias = sparse
                    A = np.column_stack([valid_est, np.ones(len(valid_est))])
                    try:
                        params = np.linalg.lstsq(A, valid_sparse, rcond=None)[0]
                        scale, bias = params[0], params[1]
                        # Clamp scale to reasonable range
                        scale = np.clip(scale, 0.1, 10.0)
                        e_metric = e_metric * scale + bias
                    except np.linalg.LinAlgError:
                        pass  # Keep original if fitting fails

            # Solve Poisson equation
            result, stats = poisson_gpu(
                sparse_metric, e_metric,
                tol=self.poisson_tol,
                maxiter=self.poisson_maxiter,
                device=dev_str,
                init=self.poisson_init,
                clip_to_max_gt=self.poisson_clip_to_max_gt
            )

            # Convert back to [0,1] scale and add to batch
            P_01 = torch.from_numpy(result / dmax).to(dev, dtype=torch.float32)
            P_list.append(P_01[None, None])  # Add batch and channel dims
            stats_list.append(stats)

        return torch.cat(P_list, dim=0), stats_list

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
            "[OASIS-DC] All inputs must share (H,W)"
        if torch.any((E < -1e-6) | (E > 1.0 + 1e-6)):
            raise ValueError("[OASIS-DC] E_norm must be in [0,1]")

        dmax = float(self.cfg.dmax)

        # Poisson P (metric)
        if self.use_poisson:
            P, poisson_stats = self._poisson_batch(DL, ML, E)
        else:
            P, poisson_stats = (E * dmax).clamp(0, dmax), None

        # main feat (I,P/dmax,E,ML)
        x_in = torch.cat([I, P / dmax, E, ML], dim=1)
        feat = self.enc(x_in)

        # P-only feat (Pn, edge(P), ML)
        if self.use_p_affinity or self.p_only_gate:
            Pn   = (P / dmax).clamp(0,1)
            edgeP= self._edge_mag_01(Pn)
            p_in = torch.cat([Pn, edgeP, ML], dim=1)   # (B,3,H,W)
            p_feat = self.penc(p_in)                   # (B,64,H,W)
        else:
            p_feat = None

        # Initial value D0
        res_out = None
        D0 = P
        if self.res is not None:
            res_out = self.res(feat)
            D0 = (P + res_out).clamp(0, dmax)

        # curvature & affinity
        if self.use_p_affinity and (p_feat is not None):
            kappa, scale, bias = self.curv(p_feat, (P / dmax).clamp(0,1))
            if self.cfg.geometry.lower().startswith("ellip"):
                A_list_raw = self.aff_head(p_feat, kappa)
            else:
                A_list_raw = self.aff_head(p_feat, scale, bias)
        else:
            kappa, scale, bias = self.curv(feat, E)
            if self.cfg.geometry.lower().startswith("ellip"):
                A_list_raw = self.aff_head(feat, kappa)
            else:
                A_list_raw = self.aff_head(feat, scale, bias)

        A_list = normalize_affinity_list(A_list_raw)

        # gate / alpha
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

        # Original propagation loop
        from ..utils import unfold_neighbors
        Dt = D0.clone()
        for _ in range(self.cfg.steps):
            mix = torch.zeros_like(Dt)
            for idx, ksz in enumerate(self.cfg.kernels):
                Ak = A_list[idx]
                kk = ksz * ksz
                patches = unfold_neighbors(Dt, ksz)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0  # Center always references D0
                Dk = (Ak * patches_center).sum(1, keepdim=True)
                mix = mix + sigma[:, idx:idx+1] * Dk

            if self.cfg.use_sparse:
                Dt = (1.0 - alpha * ML) * mix + (alpha * ML) * DL
            else:
                Dt = mix
            Dt = Dt.clamp(0, dmax)

        aux = {
            "D_pred": Dt,
            "D0": D0, "P": P, "residual": res_out,
            "sigma": sigma, "alpha": alpha,
            "A_list": A_list,
            "poisson_stats": poisson_stats,
            "p_affinity_used": bool(self.use_p_affinity)
        }
        return Dt, aux