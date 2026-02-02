#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Plain Poisson on KITTI by scanning all pairs under <root>/* .

Folder layout (required):
  <root>/
    image/               # RGB (png/jpg/...)
    velodyne_raw/        # sparse depth (16-bit png in mm or npy/npz)
    groundtruth_depth/   # GT depth (16-bit png in mm or npy/npz)
    [est/]               # (optional) precomputed relative depth (png/npy/npz/tif...)

It supports both:
- hierarchical paths (same stem across folders), and
- 'val_selection_cropped' flat naming where the middle token differs:
    *_image_*.png  <->  *_groundtruth_depth_*.png  <->  *_velodyne_raw_*.png

If est is missing while emode=precomputed, we fallback to E_norm=zeros (unless --require-est).

Edge-based outlier pruning (optional):
  --edge-prune: remove sparse points lying on strong edges of E_norm (quantile --edge-q, dilated by --edge-dilate).
  Optionally only remove those with |DL - E_norm*dmax| > --edge-resid-m (meters).

Usage:
python eval_kitti_root_poisson_only.py \
  --root /data/val_selection_cropped \
  --dmax 80.0 \
  --emode precomputed \
  --edge-prune --edge-q 0.9 --edge-dilate 1 --edge-resid-m 1.0 \
  --outdir runs_kitti_poisson \
  --save-pred --save-k 16
"""
import os, sys, time, json, csv, argparse
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------------------------------------------
# try external GPU Poisson (optional)
try:
    # signature: poisson_gpu(sparse_m, est_m, tol, maxiter, device, init, clip_to_max_gt)
    from models.module import poisson_gpu
    _HAS_POISSON_GPU = True
except Exception:
    poisson_gpu = None
    _HAS_POISSON_GPU = False

# ============================ I/O helpers ============================
IMG_EXTS  = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
DEPTH_EXTS= (".png",".npy",".npz",".tif",".tiff")
EST_EXTS  = (".png",".npy",".npz",".tif",".tiff")

def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)

def _read_depth(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    if ext == ".npz":
        z = np.load(path)
        for k in ["arr_0","depth","D","d"]:
            if k in z: return z[k].astype(np.float32)
        return list(z.values())[0].astype(np.float32)
    with Image.open(path) as im:
        if im.mode in ["I;16","I;16B","I"]:
            # KITTI: mm in 16-bit -> meters
            return (np.array(im, dtype=np.uint16).astype(np.float32) / 1000.0)
        return np.array(im, dtype=np.uint8).astype(np.float32)

def _resize_like(x: np.ndarray, target_hw: Tuple[int,int], is_mask: bool=False) -> np.ndarray:
    Ht, Wt = target_hw
    if x.shape == (Ht, Wt): return x
    pil = Image.fromarray(x.astype(np.float32))
    if is_mask:
        return np.array(pil.resize((Wt,Ht), Image.NEAREST)).astype(np.float32)
    return np.array(pil.resize((Wt,Ht), Image.BILINEAR)).astype(np.float32)

def _list_rel_images(img_root: str) -> List[str]:
    rels = []
    for r, _, files in os.walk(img_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                abspath = os.path.join(r, f)
                rel = os.path.relpath(abspath, img_root)
                rels.append(rel)
    rels.sort()
    return rels

def _find_with_exts(base_dir: str, rel_stem: str, exts: Tuple[str,...]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(base_dir, rel_stem + ext)
        if os.path.exists(p): return p
    return None

def _find_any_with_alternative_stems(base_dir: str, stems: List[str], exts: Tuple[str,...]) -> Optional[str]:
    for s in stems:
        p = _find_with_exts(base_dir, s, exts)
        if p is not None: return p
    return None

# ============================ E_norm utils ============================
def _minmax01(arr: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    if mask is not None and mask.any():
        v = arr[mask>0]
        amin, amax = float(np.min(v)), float(np.max(v))
    else:
        amin, amax = float(np.min(arr)), float(np.max(arr))
    if amax - amin < eps: return np.zeros_like(arr, np.float32)
    out = (arr - amin) / (amax - amin + eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _orient_by_sparse(E01: np.ndarray, DL: np.ndarray, ML: np.ndarray) -> np.ndarray:
    yy, xx = np.where(ML > 0)
    if len(yy) < 5: return E01
    e = E01[yy, xx].astype(np.float64); d = DL[yy, xx].astype(np.float64)
    e = (e - e.mean()) / (e.std() + 1e-9); d = (d - d.mean()) / (d.std() + 1e-9)
    corr = float(np.mean(e * d))
    return (1.0 - E01) if corr < -0.1 else E01

def _best_gray_from_rgb(rgb: np.ndarray, DL: np.ndarray, ML: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32) / 255.0
    R, G, B = rgbf[...,0], rgbf[...,1], rgbf[...,2]
    luma = 0.299*R + 0.587*G + 0.114*B
    vmax = np.max(rgbf, axis=2)
    cand = [("luma", luma), ("v", vmax), ("r", R), ("g", G), ("b", B)]
    yy, xx = np.where(ML > 0)
    if len(yy) < 5: return luma.astype(np.float32)
    best, best_val = luma, -1.0
    d = DL[yy, xx].astype(np.float64); d = (d - d.mean()) / (d.std() + 1e-9)
    for _, a in cand:
        e = a[yy, xx].astype(np.float64)
        e = (e - e.mean()) / (e.std() + 1e-9)
        val = abs(float(np.mean(e * d)))
        if val > best_val:
            best_val, best = val, a
    return best.astype(np.float32)

def _smooth3_reflect_np(x: np.ndarray) -> np.ndarray:
    pad=((1,1),(1,1)); xpad=np.pad(x, pad, mode='reflect').astype(np.float32)
    X=torch.from_numpy(xpad).view(1,1,*xpad.shape)
    k=torch.ones((1,1,3,3), dtype=torch.float32)/9.0
    y=F.conv2d(X, k).squeeze().numpy().astype(np.float32)
    return y

# ---------- Sobel & dilation for edge-based pruning ----------
def sobel_mag_np(x: np.ndarray) -> np.ndarray:
    X = torch.from_numpy(x.astype(np.float32)).view(1,1,*x.shape)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
    gx = F.conv2d(X, kx, padding=1); gy = F.conv2d(X, ky, padding=1)
    G  = torch.sqrt(gx*gx + gy*gy + 1e-12).squeeze().numpy().astype(np.float32)
    return G

def dilate_mask_np(mask01: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return (mask01 > 0).astype(np.uint8)
    t = torch.from_numpy((mask01 > 0).astype(np.float32)).view(1,1,*mask01.shape)
    k = torch.ones((1,1,3,3), dtype=torch.float32)
    for _ in range(iters):
        t = (F.conv2d(t, k, padding=1) > 0).float()
    return t.squeeze().numpy().astype(np.uint8)

# ============================ sparse sampling ============================
def subsample_sparse(depth_sparse_m: np.ndarray, n: int, seed: int, tag: str):
    H, W = depth_sparse_m.shape
    ys, xs = np.where(depth_sparse_m > 0)
    sp = np.zeros((H,W), np.float32)
    mk = np.zeros((H,W), np.uint8)
    if len(ys) == 0:
        return sp, mk
    if n <= 0 or n >= len(ys):
        sp[ys, xs] = depth_sparse_m[ys, xs]; mk[ys, xs] = 1
        return sp, mk
    rng = np.random.default_rng(seed ^ (abs(hash(tag)) & 0xffffffff))
    sel = rng.choice(len(ys), n, replace=False)
    y, x = ys[sel], xs[sel]
    sp[y, x] = depth_sparse_m[y, x]; mk[y, x] = 1
    return sp, mk

# ============================ Poisson solvers ============================
def poisson_fallback_np(sparse_m: np.ndarray, est_m: np.ndarray, m_bool: np.ndarray,
                        iters: int = 400, tol: float = 1e-5):
    v = est_m.copy()
    known = m_bool.copy()
    known[0,:] = known[-1,:] = True; known[:,0] = known[:,-1] = True
    v_known = est_m.copy()
    ks = m_bool & (sparse_m > 0)
    v_known[ks] = sparse_m[ks]
    for _ in range(iters):
        v_old = v
        v = 0.25*(np.roll(v,1,0)+np.roll(v,-1,0)+np.roll(v,1,1)+np.roll(v,-1,1))
        v[known] = v_known[known]
        if np.mean(np.abs(v - v_old)) < tol:
            break
    return v.astype(np.float32)

def build_poisson_from_enorm_with_stats(
    DL: torch.Tensor, ML: torch.Tensor, E01: torch.Tensor,
    dmax: float, tol: float, maxiter: int,
    auto_flip: bool, est_affine: bool, smooth_est: bool,
    device_str: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    assert DL.shape[0] == 1, "This helper assumes B=1."
    e  = E01[0,0].detach().cpu().numpy().astype(np.float32)
    dl = (DL [0,0] / dmax).detach().cpu().numpy().astype(np.float32)
    m  = (ML [0,0].detach().cpu().numpy().astype(np.float32) > 0)

    if auto_flip and m.sum() >= 10:
        em=e[m].reshape(-1); dm=dl[m].reshape(-1)
        if em.size>1 and dm.size>1:
            corr = float(np.corrcoef(em, dm)[0,1]); corr = 0.0 if not np.isfinite(corr) else corr
            if corr < 0.0: e = 1.0 - e

    if est_affine and m.sum() >= 10:
        x=e[m].reshape(-1,1); y=dl[m].reshape(-1,1)
        A=np.concatenate([x, np.ones_like(x)], axis=1).astype(np.float32)
        w=np.ones((A.shape[0],1), np.float32)
        for _ in range(3):
            Aw=A*w; yw=y*w
            theta,*_=np.linalg.lstsq(Aw, yw, rcond=None)
            r=(A@theta - y); c=1.345*np.median(np.abs(r))+1e-6
            w = (1.0/np.maximum(1.0, np.abs(r)/c)).astype(np.float32)
        a,b0=float(theta[0,0]), float(theta[1,0]); e = np.clip(a*e + b0, 0.0, 1.0)

    if smooth_est:
        e = _smooth3_reflect_np(e)

    est_m    = (e * dmax).astype(np.float32)
    sparse_m = (DL[0,0].detach().cpu().numpy().astype(np.float32) * m.astype(np.float32))

    t0 = time.perf_counter()
    if _HAS_POISSON_GPU and callable(poisson_gpu):
        P_np, st = poisson_gpu(
            sparse_m=sparse_m, est_m=est_m,
            tol=tol, maxiter=maxiter, device=device_str, init="est",
            clip_to_max_gt=False
        )
        t1 = time.perf_counter()
        stats = {"solver":"cg(torch)", "time_sec": float(st.get("time_sec", t1-t0)),
                 "cg_iters": int(st.get("cg_iters", -1))}
    else:
        P_np = poisson_fallback_np(sparse_m, est_m, m, iters=maxiter, tol=tol)
        t1 = time.perf_counter()
        stats = {"solver":"jacobi(np)", "time_sec": t1-t0, "cg_iters": -1}

    P = torch.from_numpy(P_np)[None,None].to(DL.device, dtype=torch.float32)
    return P, stats

# ============================ Visualization ============================
def save_jet_relative(path: str, depth_m: np.ndarray, valid_mask: Optional[np.ndarray] = None):
    x = depth_m.astype(np.float32)
    if valid_mask is not None and valid_mask.any():
        vmin, vmax = float(np.min(x[valid_mask>0])), float(np.max(x[valid_mask>0]))
    else:
        vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax - vmin < 1e-6:
        n = np.zeros_like(x, np.float32)
    else:
        n = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0*(n-0.75)), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0*(n-0.50)), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0*(n-0.25)), 0.0, 1.0)
    rgb = (np.stack([r,g,b], -1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(path)

def save_png16(path: str, depth_m: np.ndarray, scale_max_m: float):
    arr = np.clip(depth_m, 0.0, scale_max_m) * 1000.0
    Image.fromarray(arr.astype(np.uint16)).save(path)

# ============================ Metrics ============================
@torch.no_grad()
def rmse_t(pred, gt, mask):
    diff = (pred - gt) * mask
    mse  = (diff ** 2).sum() / mask.sum().clamp_min(1.0)
    return float(torch.sqrt(mse).item())

@torch.no_grad()
def mae_t(pred, gt, mask):
    diff = (pred - gt).abs() * mask
    mae  = diff.sum() / mask.sum().clamp_min(1.0)
    return float(mae.item())

@torch.no_grad()
def delta1_t(pred, gt, mask):
    eps = 1e-6
    p = torch.clamp(pred, min=eps)
    g = torch.clamp(gt,   min=eps)
    r = torch.maximum(p/g, g/p) * mask
    r = r[mask > 0]
    if r.numel() == 0:
        return 0.0
    return float((r < 1.25).float().mean().item())

# ============================ Scanner (val_selection compatible) ============================
def _alt_stems_for_rel(rel_stem: str) -> Dict[str, List[str]]:
    stems = {"gt":[rel_stem], "sp":[rel_stem], "est":[rel_stem]}
    if "_image_" in rel_stem:
        stems["gt"].append(rel_stem.replace("_image_", "_groundtruth_depth_", 1))
        stems["sp"].append(rel_stem.replace("_image_", "_velodyne_raw_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_est_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_groundtruth_depth_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_velodyne_raw_", 1))
    return stems

def scan_kitti_pairs(root: str, require_est: bool) -> Tuple[List[Tuple[str,str,str,str,Optional[str]]], Dict[str,int]]:
    img_root = os.path.join(root, "image")
    gt_root  = os.path.join(root, "groundtruth_depth")
    sp_root  = os.path.join(root, "velodyne_raw")
    est_root = os.path.join(root, "est")

    if not os.path.isdir(img_root) or not os.path.isdir(gt_root) or not os.path.isdir(sp_root):
        raise FileNotFoundError(f"Expected subfolders 'image', 'groundtruth_depth', 'velodyne_raw' under: {root}")

    img_rels = _list_rel_images(img_root)
    pairs = []
    missing = {"gt":0, "sp":0, "est":0}

    for rel in img_rels:
        stem = os.path.splitext(rel)[0]
        img_path = os.path.join(img_root, rel)

        alts = _alt_stems_for_rel(stem)
        gt_path = _find_any_with_alternative_stems(gt_root, alts["gt"], DEPTH_EXTS)
        sp_path = _find_any_with_alternative_stems(sp_root, alts["sp"], DEPTH_EXTS)
        if gt_path is None:
            missing["gt"] += 1; continue
        if sp_path is None:
            missing["sp"] += 1; continue

        est_path = None
        if os.path.isdir(est_root):
            est_path = _find_any_with_alternative_stems(est_root, alts["est"], EST_EXTS)
        if require_est and est_path is None:
            missing["est"] += 1; continue

        pairs.append((stem, img_path, gt_path, sp_path, est_path))

    return pairs, missing

# ============================ Main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--dmax", type=float, default=80.0)
    ap.add_argument("--emode", type=str, default="precomputed", choices=["precomputed","zeros"],
                    help="How to get E_norm: precomputed from est/, or zeros (ablation). "
                         "If precomputed but est missing, we fallback to zeros unless --require-est.")
    ap.add_argument("--require-est", action="store_true",
                    help="If set with emode=precomputed, samples without est will be skipped instead of zeros fallback.")
    ap.add_argument("--limit-sparse", type=int, default=-1, help="<=0: use all lidar points; >0: subsample to N")
    # edge-based pruning
    ap.add_argument("--edge-prune", action="store_true", help="Remove sparse points on strong E_norm edges")
    ap.add_argument("--edge-q", type=float, default=0.90, help="Quantile for edge magnitude on E_norm")
    ap.add_argument("--edge-dilate", type=int, default=1, help="Edge mask dilation iterations")
    ap.add_argument("--edge-resid-m", type=float, default=-1.0,
                    help="If >0, only prune points on edges whose |DL - E_norm*dmax| exceeds this (meters)")
    # poisson
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--no-auto-flip", action="store_true")
    ap.add_argument("--no-est-affine", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    # output
    ap.add_argument("--outdir", type=str, default="runs_kitti_poisson")
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--save-k", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.save_pred:
        os.makedirs(os.path.join(args.outdir, "pred_png16"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "pred_jet"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "pred_npy"), exist_ok=True)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[note] CUDA not available; switching device='cpu'")
        args.device = "cpu"

    pairs, missing = scan_kitti_pairs(args.root, require_est=(args.emode=="precomputed" and args.require_est))
    if len(pairs) == 0:
        print("[warn] No valid pairs found.")
        print("  Missing counts:", missing)
        print("  HINT: If you don't have 'est/', use --emode zeros (or drop --require-est).")
        return

    if args.emode=="precomputed" and not args.require_est and missing["est"]>0:
        print(f"[info] est missing for {missing['est']} images -> fallback to E_norm=zeros for those samples.")

    mae_sum = 0.0; rmse_sum = 0.0; d1_sum = 0.0; n = 0
    per_image = []
    solver_times: List[float] = []; solver_iters: List[int] = []
    est_used_cnt = 0; est_fallback_cnt = 0
    pruned_pts_total = 0; pts_before_total = 0

    t0_all = time.perf_counter()
    pbar = tqdm(pairs, ncols=120, desc="[KITTI][Plain Poisson]")
    for stem, img_p, gt_p, sp_p, est_p in pbar:
        rgb = _read_rgb(img_p)
        Dgt = _read_depth(gt_p)
        DL_full = _read_depth(sp_p)

        H, W = rgb.shape[:2]
        if Dgt.shape != (H,W): Dgt = _resize_like(Dgt, (H,W))
        if DL_full.shape != (H,W): DL_full = _resize_like(DL_full, (H,W))

        valid = (Dgt > 0).astype(np.uint8)

        # sparse (optional subsample)
        if args.limit_sparse > 0:
            DL, ML = subsample_sparse(DL_full, args.limit_sparse, seed=0, tag=stem)
        else:
            DL = DL_full.astype(np.float32)
            ML = (DL_full > 0).astype(np.uint8)

        # E_norm
        use_zeros = False
        if args.emode == "precomputed" and est_p is not None:
            raw = _read_depth(est_p)
            if raw.ndim == 2:
                if raw.shape != (H,W):
                    raw = _resize_like(raw, (H,W))
                Er = raw.astype(np.float32)
            else:
                if raw.shape[:2] != (H,W):
                    raw = np.array(Image.fromarray(raw.astype(np.uint8)).resize((W,H), Image.BILINEAR))
                Er = _best_gray_from_rgb(raw.astype(np.uint8), DL, ML) * 255.0
            E01 = _minmax01(Er, mask=valid); E01 = _orient_by_sparse(E01, DL, ML)
            est_used_cnt += 1
        elif args.emode == "precomputed" and est_p is None:
            if args.require_est:
                continue
            E01 = np.zeros((H,W), np.float32); use_zeros = True; est_fallback_cnt += 1
        else:
            E01 = np.zeros((H,W), np.float32); use_zeros = True

        # ---------- Edge-based outlier pruning on sparse points ----------
        n_before = int(ML.sum())
        if args.edge_prune and (E01.max() - E01.min()) > 1e-8 and n_before > 0:
            G = sobel_mag_np(E01.astype(np.float32))
            thr = float(np.quantile(G, args.edge_q))
            edge_mask = (G >= thr).astype(np.uint8)
            if args.edge_dilate > 0:
                edge_mask = dilate_mask_np(edge_mask, iters=args.edge_dilate)

            if args.edge_resid_m is not None and args.edge_resid_m > 0:
                resid = np.abs(DL - (E01 * args.dmax))
                bad = (edge_mask > 0) & (resid > float(args.edge_resid_m))
            else:
                bad = (edge_mask > 0)

            # prune points on edges (and optionally large residuals)
            pruned_mask = (ML > 0) & bad
            pruned_count = int(pruned_mask.sum())
            ML[pruned_mask] = 0
        else:
            pruned_count = 0

        n_after = int(ML.sum())
        pts_before_total += n_before
        pruned_pts_total += pruned_count

        # to tensors (B=1)
        tDL  = torch.from_numpy(DL)[None,None]
        tML  = torch.from_numpy(ML).bool().float()[None,None]
        tE01 = torch.from_numpy(E01)[None,None]
        tGT  = torch.from_numpy(Dgt)[None,None]
        tVM  = torch.from_numpy(valid).bool().float()[None,None]

        # solve plain Poisson
        P, st = build_poisson_from_enorm_with_stats(
            DL=tDL, ML=tML, E01=tE01,
            dmax=args.dmax, tol=args.tol, maxiter=args.maxiter,
            auto_flip=(not args.no_auto_flip),
            est_affine=(not args.no_est_affine),
            smooth_est=(not args.no_smooth),
            device_str=args.device
        )

        # metrics
        mae = mae_t(P, tGT, tVM); rmse = rmse_t(P, tGT, tVM); d1 = delta1_t(P, tGT, tVM)
        mae_sum += mae; rmse_sum += rmse; d1_sum += d1; n += 1

        solver_times.append(float(st.get("time_sec", 0.0))); solver_iters.append(int(st.get("cg_iters", -1)))
        pbar.set_postfix(MAE=f"{mae_sum/n:.4f}", RMSE=f"{rmse_sum/n:.4f}", d1=f"{d1_sum/n:.4f}",
                         t=f"{solver_times[-1]:.3f}s", it=solver_iters[-1],
                         E=("zeros" if use_zeros else "est"),
                         pruned=f"{pruned_count}/{n_before}")

        # save predictions (optional)
        if args.save_pred and (len(per_image) < args.save_k):
            pred = P[0,0].detach().cpu().numpy()
            base = os.path.basename(stem)
            save_png16(os.path.join(args.outdir, "pred_png16", f"{base}.png"), pred, args.dmax)
            save_jet_relative(os.path.join(args.outdir, "pred_jet",  f"{base}_jet.png"), pred, valid_mask=valid)
            np.save(os.path.join(args.outdir, "pred_npy", f"{base}.npy"), pred.astype(np.float32))

        per_image.append({
            "id": stem,
            "mae": mae, "rmse": rmse, "delta1": d1,
            "solver": st.get("solver",""), "time_sec": st.get("time_sec", None),
            "cg_iters": st.get("cg_iters", None),
            "n_sparse_before": n_before,
            "n_sparse_after": n_after,
            "n_pruned_edge": pruned_count,
            "E_norm_source": ("zeros" if use_zeros else "precomputed")
        })

    total = time.perf_counter() - t0_all
    avg_mae  = mae_sum / max(n,1)
    avg_rmse = rmse_sum / max(n,1)
    avg_d1   = d1_sum / max(n,1)
    thr = n/total if total>0 else None
    avg_st = float(np.mean(solver_times)) if solver_times else None
    med_st = float(np.median(solver_times)) if solver_times else None
    avg_it = float(np.mean([x for x in solver_iters if x>=0])) if any(x>=0 for x in solver_iters) else None

    summary = {
        "count": n,
        "dmax": args.dmax,
        "emode": args.emode,
        "require_est": bool(args.require_est),
        "limit_sparse": args.limit_sparse,
        "tol": args.tol, "maxiter": args.maxiter, "device": args.device,
        "MAE": avg_mae, "RMSE": avg_rmse, "delta1": avg_d1,
        "total_eval_time_sec": total,
        "throughput_img_per_sec": thr,
        "avg_solver_time_sec": avg_st,
        "median_solver_time_sec": med_st,
        "avg_cg_iters": avg_it,
        "poisson_auto_flip": not args.no_auto_flip,
        "poisson_est_affine": not args.no_est_affine,
        "poisson_smooth_est": not args.no_smooth,
        "pairs_missing_counts": missing,
        "est_used": est_used_cnt,
        "est_fallback_zeros": est_fallback_cnt,
        "edge_prune": bool(args.edge_prune),
        "edge_q": args.edge_q,
        "edge_dilate": args.edge_dilate,
        "edge_resid_m": args.edge_resid_m,
        "total_pruned_edge": pruned_pts_total,
        "total_sparse_before": pts_before_total
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "metrics.jsonl"), "w") as f:
        for row in per_image:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()) if per_image else
                           ["id","mae","rmse","delta1","solver","time_sec","cg_iters",
                            "n_sparse_before","n_sparse_after","n_pruned_edge","E_norm_source"])
        w.writeheader()
        for r in per_image: w.writerow(r)

    print("\n=== KITTI Plain Poisson (root-scan) ===")
    print(f"Count {n} | MAE {avg_mae:.4f} | RMSE {avg_rmse:.4f} | δ1 {avg_d1:.4f}")
    if thr is not None:
        print(f"Speed: {thr:.3f} img/s (total {total:.2f}s)")
    if avg_st is not None:
        print(f"Solver avg/med: {avg_st:.4f}s / {med_st:.4f}s | iters(avg): {avg_it if avg_it is not None else 'n/a'}")
    print(f"Edge prune: total removed {pruned_pts_total} / {pts_before_total} sparse points")
    print("Missing counts:", summary["pairs_missing_counts"])
    if args.emode=="precomputed":
        print(f"est used: {summary['est_used']} | est→zeros fallback: {summary['est_fallback_zeros']}")

if __name__ == "__main__":
    main()
