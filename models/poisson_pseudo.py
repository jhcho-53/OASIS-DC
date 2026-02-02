# train_mcpropnet.py
import os, math, argparse, time
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # <<< NEW

# --------------------- import your dataloader (same folder)
from dataloader.dataloader_nyu import build_mcprop_dataloaders, seed_all

# ===================== finite-difference ops =====================
def grad_xy(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gx = F.pad(u[:,:,:,1:] - u[:,:,:,:-1], (0,1,0,0))
    gy = F.pad(u[:,:,1:,:] - u[:,:,:-1,:], (0,0,0,1))
    return gx, gy

def div_xy(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    px_l = F.pad(px[:,:,:,:-1], (1,0,0,0))
    py_u = F.pad(py[:,:,:-1,:], (0,0,1,0))
    return (px - px_l) + (py - py_u)

def laplacian_weighted(u: torch.Tensor, wg: torch.Tensor) -> torch.Tensor:
    gx, gy = grad_xy(u)
    return -div_xy(wg * gx, wg * gy)

def jacobi_diag(wg: torch.Tensor, lam: torch.Tensor, mu_eff: torch.Tensor) -> torch.Tensor:
    B, _, H, W = wg.shape
    mask_r = torch.ones_like(wg); mask_r[:,:,:, -1] = 0.0
    mask_d = torch.ones_like(wg); mask_d[:,:, -1,:] = 0.0
    diag_x = wg * mask_r + F.pad(wg * mask_r, (1,0,0,0))[:,:,:,:W]
    diag_y = wg * mask_d + F.pad(wg * mask_d, (0,0,1,0))[:,:,:H,:]
    return lam + mu_eff + diag_x + diag_y

# ===================== CG solver (batched) =====================
@torch.no_grad()
def cg_solve(Aop, b, x0=None, Mdiag=None, max_iter=150, tol=1e-6):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Aop(x)
    z = r if Mdiag is None else r / (Mdiag + 1e-8)
    p = z.clone()
    def bdot(a, c): return (a*c).flatten(1).sum(dim=1, keepdim=True)
    rz_old = bdot(r, z)
    for _ in range(max_iter):
        Ap = Aop(p)
        denom = bdot(p, Ap) + 1e-12
        alpha = rz_old / denom
        x = x + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1).unsqueeze(-1) * Ap
        if (r.flatten(1).norm(dim=1) <= tol * b.flatten(1).norm(dim=1).clamp_min(1e-12)).all():
            break
        z = r if Mdiag is None else r / (Mdiag + 1e-8)
        rz_new = bdot(r, z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta.unsqueeze(-1).unsqueeze(-1) * p
        rz_old = rz_new
    return x

# ===================== Autograd Poisson (implicit backward) =====================
class ScreenedPoissonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ehat, DL, M, lam, mu, wg, max_iter, tol):
        ctx.save_for_backward(Ehat, DL, M, lam, mu, wg)
        ctx.max_iter = int(max_iter); ctx.tol = float(tol)
        mu_eff = mu * M
        def Aop(x): return laplacian_weighted(x, wg) + lam * x + mu_eff * x
        b = div_xy(*(wg * t for t in grad_xy(Ehat))) + lam * Ehat + mu_eff * DL
        Mdiag = jacobi_diag(wg, lam, mu_eff)
        P = cg_solve(Aop, b, x0=None, Mdiag=Mdiag, max_iter=ctx.max_iter, tol=ctx.tol)
        ctx.P = P.detach()
        return P

    @staticmethod
    def backward(ctx, dP):
        Ehat, DL, M, lam, mu, wg = ctx.saved_tensors
        mu_eff = mu * M
        def Aop(x): return laplacian_weighted(x, wg) + lam * x + mu_eff * x
        Mdiag = jacobi_diag(wg, lam, mu_eff)
        v = cg_solve(Aop, dP, x0=None, Mdiag=Mdiag, max_iter=ctx.max_iter, tol=ctx.tol)
        gx_v, gy_v = grad_xy(v)
        gx_E, gy_E = grad_xy(Ehat)
        gx_P, gy_P = grad_xy(ctx.P)
        dEhat = -div_xy(wg * gx_v, wg * gy_v) + lam * v
        dDL   = mu_eff * v
        dlam  = v * (Ehat - ctx.P)
        dmu   = v * M * (DL - ctx.P)
        dwg   = -(gx_v * (gx_E + gx_P) + gy_v * (gy_E + gy_P))
        return dEhat, dDL, None, dlam, dmu, dwg, None, None

class ScreenedPoissonLayer(nn.Module):
    def __init__(self, max_iter=150, tol=1e-6):
        super().__init__()
        self.max_iter = int(max_iter); self.tol = float(tol)
    def forward(self, Ehat, DL, M, lam, mu, wg):
        return ScreenedPoissonFn.apply(Ehat, DL, M, lam, mu, wg, self.max_iter, self.tol)

# ===================== Light MCPropNet (learn-light heads) =====================
class LightMCPropNet(nn.Module):
    def __init__(self, depth_max=10.0, base_ch=32, lambda_max=5.0, mu_max=200.0, wg_max=1.0):
        super().__init__()
        self.depth_max, self.lambda_max, self.mu_max, self.wg_max = float(depth_max), float(lambda_max), float(mu_max), float(wg_max)
        in_ch, ch = 4, base_ch
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_flip = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.SiLU(), nn.Conv2d(ch, 1, 1))
        self.head_ab   = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.SiLU(), nn.Conv2d(ch, 2, 1))
        self.head_lambda = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, 1, 1))
        self.head_mu     = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, 1, 1))
        self.head_wg     = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, 1, 1))
        self.poisson = ScreenedPoissonLayer(max_iter=150, tol=1e-6)

    def forward(self, batch):
        E  = batch["E_norm"].to(dtype=torch.float32)
        DL = batch["DL"].to(dtype=torch.float32)
        ML = batch["ML"].to(dtype=torch.float32)
        DLn = (DL / self.depth_max).clamp(0, 1)
        gx, gy = grad_xy(E); gmag = torch.sqrt(gx*gx + gy*gy + 1e-6)
        x = torch.cat([E, DLn, ML, gmag], dim=1)
        feat = self.trunk(x)
        flip_logit = self.head_flip(self.gap(feat)); alpha = torch.sigmoid(flip_logit)
        ab = self.head_ab(self.gap(feat)); a = F.softplus(ab[:,0:1]) + 1e-3; b = ab[:,1:2].tanh() * 0.5
        E_oriented = E + alpha * (1.0 - 2.0*E)
        E_affine = (a * E_oriented + b).clamp(0.0, 1.0)
        lam = torch.sigmoid(self.head_lambda(feat)) * self.lambda_max
        mu  = torch.sigmoid(self.head_mu(feat))     * self.mu_max
        wg  = torch.sigmoid(self.head_wg(feat))     * self.wg_max + 1e-6
        Ehat_m = E_affine * self.depth_max
        P = self.poisson(Ehat_m, DL, ML, lam, mu, wg)
        return {"pred_depth": P, "E_affine": E_affine,
                "alpha": alpha.squeeze(-1).squeeze(-1),
                "a": a.squeeze(-1).squeeze(-1), "b": b.squeeze(-1).squeeze(-1),
                "lam": lam, "mu": mu, "wg": wg}

# ===================== Losses & Metrics =====================
def gradient_loss(pred, gt, mask):
    gx_p, gy_p = grad_xy(pred); gx_g, gy_g = grad_xy(gt)
    return ((gx_p-gx_g).abs() + (gy_p-gy_g).abs()) * mask

def si_log_loss(pred, gt, mask):
    eps = 1e-6
    lp = torch.log(pred.clamp_min(eps)); lg = torch.log(gt.clamp_min(eps))
    d = (lp - lg) * mask; n = mask.sum().clamp_min(1.0)
    mean_d = d.sum()/n
    return ((d - mean_d)**2).sum()

def laplacian_consistency(pred, Ehat_m, mask):
    def lap(u): return laplacian_weighted(u, torch.ones_like(u))
    return (lap(pred) - lap(Ehat_m)).abs() * mask

def compute_metrics(pred, gt, mask):
    n = mask.sum().clamp_min(1.0)
    mae = ((pred-gt).abs() * mask).sum()/n
    rmse = torch.sqrt(((pred-gt)**2 * mask).sum()/n)
    return mae.item(), rmse.item()

# ===================== Train/Eval loops (with tqdm) =====================
def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    tot_loss, t0 = 0.0, time.time()
    pbar = tqdm(loader, total=len(loader), desc="Train", ncols=120, disable=cfg.no_tqdm)
    for it, batch in enumerate(pbar):
        batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        valid = batch["meta"]["valid_mask"].to(device).unsqueeze(1).float()
        D_gt  = batch["D_gt"].to(device).float().clamp(0, cfg.depth_max)
        out = model(batch); P = out["pred_depth"].clamp(0, cfg.depth_max)
        l1 = (P - D_gt).abs() * valid
        gl = gradient_loss(P, D_gt, valid)
        si = si_log_loss(P + 1e-3, D_gt + 1e-3, valid)
        Ehat_m = out["E_affine"] * cfg.depth_max
        pc = laplacian_consistency(P, Ehat_m, valid)
        loss = (cfg.w_l1 * l1.mean()
               +cfg.w_grad * gl.mean()
               +cfg.w_si * si / valid.sum().clamp_min(1.0)
               +cfg.w_pc * pc.mean())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tot_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{tot_loss/(it+1):.4f}")
    return tot_loss / max(1,len(loader)), time.time()-t0

@torch.no_grad()
def eval_one_epoch(model, loader, device, depth_max, no_tqdm=False):
    model.eval()
    mae_sum=0.0; rmse_sum=0.0; nimg=0
    pbar = tqdm(loader, total=len(loader), desc="Eval ", ncols=120, disable=no_tqdm)
    for batch in pbar:
        batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        valid = batch["meta"]["valid_mask"].to(device).unsqueeze(1).float()
        D_gt  = batch["D_gt"].to(device).float().clamp(0, depth_max)
        out = model(batch); P = out["pred_depth"].clamp(0, depth_max)
        mae, rmse = compute_metrics(P, D_gt, valid)
        mae_sum += mae; rmse_sum += rmse; nimg += 1
        pbar.set_postfix(MAE=f"{mae_sum/nimg:.4f}", RMSE=f"{rmse_sum/nimg:.4f}")
    return mae_sum/max(1,nimg), rmse_sum/max(1,nimg)

# ===================== Main =====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--train-list", type=str, required=True)
    p.add_argument("--test-list", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-sparse", type=int, default=500)
    p.add_argument("--depth-max", type=float, default=10.0)
    p.add_argument("--emode", type=str, default="foundation", choices=["precomputed","foundation","zeros"])
    p.add_argument("--pre-mono-dir", type=str, default="mono_rel")
    p.add_argument("--strict-precomputed", action="store_true", default=True)
    # loss weights
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-grad", type=float, default=0.5)
    p.add_argument("--w-si", type=float, default=0.2)
    p.add_argument("--w-pc", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--save", type=str, default="ckpts")
    p.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bars")
    p.add_argument("--val-every", type=int, default=30, help="run validation every N epochs")  # <<< NEW
    args = p.parse_args()

    os.makedirs(args.save, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_mcprop_dataloaders(
        root=args.root, train_list=args.train_list, test_list=args.test_list,
        seed=args.seed, shots=None, n_sparse=args.n_sparse, emode=args.emode,
        foundation_infer=None, batch_size=args.batch_size, num_workers=args.num_workers,
        pre_mono_dir=args.pre_mono_dir, strict_precomputed=args.strict_precomputed,
        normalize_on_valid_gt=True
    )

    model = LightMCPropNet(depth_max=args.depth_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)

    best_rmse = 1e9
    for ep in range(1, args.epochs+1):
        tr_loss, tr_time = train_one_epoch(model, train_loader, optimizer, device, args)

        # ---- validate only every N epochs, and always on the last epoch
        do_val = (ep % args.val_every == 0) or (ep == args.epochs)  # <<< NEW
        if do_val:
            mae, rmse = eval_one_epoch(model, test_loader, device, args.depth_max, no_tqdm=args.no_tqdm)
            print(f"[Ep {ep:02d}] train_loss={tr_loss:.4f} ({tr_time:.1f}s)  |  Val MAE={mae:.4f}, RMSE={rmse:.4f}")
            if rmse < best_rmse:
                best_rmse = rmse
                ckpt = {"ep": ep, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "args": vars(args)}
                torch.save(ckpt, os.path.join(args.save, f"mcpropnet_best.pth"))
        else:
            print(f"[Ep {ep:02d}] train_loss={tr_loss:.4f} ({tr_time:.1f}s)  |  Val: (skipped)")

        # periodic snapshot
        if ep % 5 == 0:
            torch.save({"ep":ep, "model":model.state_dict()},
                       os.path.join(args.save, f"mcpropnet_ep{ep:03d}.pth"))

if __name__ == "__main__":
    main()
