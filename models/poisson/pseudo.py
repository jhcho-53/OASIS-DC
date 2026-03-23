"""
Pseudo depth generation using Poisson equations with differentiable solvers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def grad_xy(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute finite difference gradients."""
    gx = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1))
    return gx, gy


def div_xy(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Compute finite difference divergence."""
    px_l = F.pad(px[:, :, :, :-1], (1, 0, 0, 0))
    py_u = F.pad(py[:, :, :-1, :], (0, 0, 1, 0))
    return (px - px_l) + (py - py_u)


def laplacian_weighted(u: torch.Tensor, wg: torch.Tensor) -> torch.Tensor:
    """Compute weighted Laplacian: -div(wg * grad(u))"""
    gx, gy = grad_xy(u)
    return -div_xy(wg * gx, wg * gy)


def jacobi_diag(wg: torch.Tensor, lam: torch.Tensor, mu_eff: torch.Tensor) -> torch.Tensor:
    """Compute diagonal for Jacobi preconditioning."""
    B, _, H, W = wg.shape
    mask_r = torch.ones_like(wg)
    mask_r[:, :, :, -1] = 0.0
    mask_d = torch.ones_like(wg) 
    mask_d[:, :, -1, :] = 0.0
    
    diag_x = wg * mask_r + F.pad(wg * mask_r, (1, 0, 0, 0))[:, :, :, :W]
    diag_y = wg * mask_d + F.pad(wg * mask_d, (0, 0, 1, 0))[:, :, :H, :]
    return lam + mu_eff + diag_x + diag_y


@torch.no_grad()
def cg_solve(Aop, b, x0=None, Mdiag=None, max_iter=150, tol=1e-6):
    """Batched Conjugate Gradient solver."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Aop(x)
    z = r if Mdiag is None else r / (Mdiag + 1e-8)
    p = z.clone()
    
    def bdot(a, c): 
        return (a * c).flatten(1).sum(dim=1, keepdim=True)
    
    rz_old = bdot(r, z)
    
    for _ in range(max_iter):
        Ap = Aop(p)
        denom = bdot(p, Ap) + 1e-12
        alpha = rz_old / denom
        x = x + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1).unsqueeze(-1) * Ap
        
        # Check convergence
        if (r.flatten(1).norm(dim=1) <= tol * b.flatten(1).norm(dim=1).clamp_min(1e-12)).all():
            break
            
        z = r if Mdiag is None else r / (Mdiag + 1e-8)
        rz_new = bdot(r, z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta.unsqueeze(-1).unsqueeze(-1) * p
        rz_old = rz_new
        
    return x


class ScreenedPoissonFn(torch.autograd.Function):
    """Differentiable screened Poisson solver with implicit backward pass."""
    
    @staticmethod
    def forward(ctx, Ehat, DL, M, lam, mu, wg, max_iter, tol):
        ctx.save_for_backward(Ehat, DL, M, lam, mu, wg)
        ctx.max_iter = int(max_iter)
        ctx.tol = float(tol)
        
        mu_eff = mu * M
        
        def Aop(x): 
            return laplacian_weighted(x, wg) + lam * x + mu_eff * x
        
        b = div_xy(*(wg * t for t in grad_xy(Ehat))) + lam * Ehat + mu_eff * DL
        Mdiag = jacobi_diag(wg, lam, mu_eff)
        P = cg_solve(Aop, b, x0=None, Mdiag=Mdiag, max_iter=ctx.max_iter, tol=ctx.tol)
        ctx.P = P.detach()
        return P

    @staticmethod
    def backward(ctx, dP):
        Ehat, DL, M, lam, mu, wg = ctx.saved_tensors
        mu_eff = mu * M
        
        def Aop(x): 
            return laplacian_weighted(x, wg) + lam * x + mu_eff * x
        
        Mdiag = jacobi_diag(wg, lam, mu_eff)
        v = cg_solve(Aop, dP, x0=None, Mdiag=Mdiag, max_iter=ctx.max_iter, tol=ctx.tol)
        
        gx_v, gy_v = grad_xy(v)
        gx_E, gy_E = grad_xy(Ehat)
        gx_P, gy_P = grad_xy(ctx.P)
        
        dEhat = -div_xy(wg * gx_v, wg * gy_v) + lam * v
        dDL = mu_eff * v
        dlam = v * (Ehat - ctx.P)
        dmu = v * M * (DL - ctx.P)
        dwg = -(gx_v * (gx_E + gx_P) + gy_v * (gy_E + gy_P))
        
        return dEhat, dDL, None, dlam, dmu, dwg, None, None


class ScreenedPoissonLayer(nn.Module):
    """Neural network layer for screened Poisson equation solving."""
    
    def __init__(self, max_iter=150, tol=1e-6):
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        
    def forward(self, Ehat, DL, M, lam, mu, wg):
        """
        Args:
            Ehat: Depth estimates
            DL: Sparse depth measurements  
            M: Measurement mask
            lam: Data term weight
            mu: Measurement weight
            wg: Edge weights
        """
        return ScreenedPoissonFn.apply(Ehat, DL, M, lam, mu, wg, self.max_iter, self.tol)