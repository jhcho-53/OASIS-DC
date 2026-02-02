#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize MCPropNet outputs on KITTI-like folder root:

Required:
  <root>/
    image/
    velodyne_raw/
    groundtruth_depth/   (optional but recommended for GT/error viz)
    est/                 (optional; relative depth prior)

It supports:
- hierarchical same-stem matching, and
- val_selection_cropped style:
    *_image_* <-> *_velodyne_raw_* <-> *_groundtruth_depth_*

Outputs:
  outdir/
    depth_png16_mm/   (Dt, P, D0, GT) in 16-bit PNG (mm)
    depth_jet/        (Dt, P, D0, GT) in JET colormap
    maps/             (alpha, sigma*, residual, ML, E01, err maps)

Example:
python visualize_mcprop_kitti.py \
  --root /data/val_selection_cropped \
  --ckpt /path/to/ckpt.pth \
  --outdir vis_kitti_mcprop \
  --dmax 80.0 \
  --num 50 \
  --device cuda:0 \
  --steps 1 \
  --use-residual --use-p-affinity
"""

import os
import time
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict

# ====== import your MCPropNet
from models.model import MCPropNet  # <-- 환경에 맞게 수정

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
            # KITTI depth png: usually mm in 16-bit
            return (np.array(im, dtype=np.uint16).astype(np.float32) / 1000.0)  # -> meters
        return np.array(im, dtype=np.uint8).astype(np.float32)

def _resize_like(x: np.ndarray, target_hw: Tuple[int,int], is_mask: bool=False) -> np.ndarray:
    Ht, Wt = target_hw
    if x.shape[:2] == (Ht, Wt): return x
    pil = Image.fromarray(x.astype(np.float32) if x.ndim==2 else x.astype(np.uint8))
    if is_mask:
        out = pil.resize((Wt,Ht), Image.NEAREST)
    else:
        out = pil.resize((Wt,Ht), Image.BILINEAR)
    return np.array(out).astype(np.float32)

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

def _alt_stems_for_rel(rel_stem: str) -> Dict[str, List[str]]:
    stems = {"gt":[rel_stem], "sp":[rel_stem], "est":[rel_stem]}
    if "_image_" in rel_stem:
        stems["gt"].append(rel_stem.replace("_image_", "_groundtruth_depth_", 1))
        stems["sp"].append(rel_stem.replace("_image_", "_velodyne_raw_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_est_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_groundtruth_depth_", 1))
        stems["est"].append(rel_stem.replace("_image_", "_velodyne_raw_", 1))
    return stems

def scan_kitti_pairs(root: str, require_est: bool) -> List[Tuple[str,str,Optional[str],str,Optional[str]]]:
    """
    Returns list of (stem, img_path, gt_path_or_None, sp_path, est_path_or_None)
    GT is optional (for test sets you might not have it).
    """
    img_root = os.path.join(root, "image")
    gt_root  = os.path.join(root, "groundtruth_depth")
    sp_root  = os.path.join(root, "velodyne_raw")
    est_root = os.path.join(root, "est")

    if not os.path.isdir(img_root) or not os.path.isdir(sp_root):
        raise FileNotFoundError(f"Need 'image' and 'velodyne_raw' under: {root}")

    has_gt = os.path.isdir(gt_root)
    has_est= os.path.isdir(est_root)

    img_rels = _list_rel_images(img_root)
    pairs = []
    for rel in img_rels:
        stem = os.path.splitext(rel)[0]
        img_path = os.path.join(img_root, rel)

        alts = _alt_stems_for_rel(stem)
        sp_path = _find_any_with_alternative_stems(sp_root, alts["sp"], DEPTH_EXTS)
        if sp_path is None:
            continue

        gt_path = None
        if has_gt:
            gt_path = _find_any_with_alternative_stems(gt_root, alts["gt"], DEPTH_EXTS)

        est_path = None
        if has_est:
            est_path = _find_any_with_alternative_stems(est_root, alts["est"], EST_EXTS)

        if require_est and est_path is None:
            continue

        pairs.append((stem, img_path, gt_path, sp_path, est_path))
    return pairs

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
    cand = [luma, vmax, R, G, B]
    yy, xx = np.where(ML > 0)
    if len(yy) < 5: return luma.astype(np.float32)
    best, best_val = luma, -1.0
    d = DL[yy, xx].astype(np.float64); d = (d - d.mean()) / (d.std() + 1e-9)
    for a in cand:
        e = a[yy, xx].astype(np.float64)
        e = (e - e.mean()) / (e.std() + 1e-9)
        val = abs(float(np.mean(e * d)))
        if val > best_val:
            best_val, best = val, a
    return best.astype(np.float32)

# ============================ checkpoint load ============================
def strip_module_prefix(sd):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

def load_checkpoint(model, ckpt_path, device="cpu", strict=False):
    ckpt = torch.load(ckpt_path, map_location=device)

    # ✅ 네 ckpt 구조: {'ep','model','optimizer','args'}
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]          # ✅ 여기 추가
        else:
            # dict 자체가 state_dict 형태인지 확인
            if all(hasattr(v, "shape") for v in ckpt.values()):
                sd = ckpt
            else:
                raise KeyError(f"Checkpoint keys={list(ckpt.keys())} 에서 state_dict를 찾지 못함.")

    else:
        sd = ckpt

    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected


# ============================ save utils ============================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def to_uint16_mm(depth_m, dmax_m):
    d = np.nan_to_num(depth_m, nan=0.0, posinf=dmax_m, neginf=0.0)
    d = np.clip(d, 0.0, dmax_m)
    mm = (d * 1000.0 + 0.5).astype(np.uint16)
    return mm

def save_png16_mm(path, depth_m, dmax_m):
    cv2.imwrite(path, to_uint16_mm(depth_m, dmax_m))

def save_colormap(path, x, vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_JET):
    a = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    a = np.clip((a - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cmap)
    cv2.imwrite(path, colored)

def save_gray_u8(path, x, vmin=0.0, vmax=1.0):
    a = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    a = np.clip((a - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, u8)

def chw_to_hwc01(img_chw):
    x = img_chw.transpose(1,2,0)
    x = np.clip(x, 0.0, 1.0)
    return x

# ============================ cfg builder ============================
def make_cfg(
    dmax=80.0,
    steps=1,
    geometry="hyper",
    use_sparse=True,
    use_residual=True,
    use_poisson=True,
    poisson_tol=1e-5,
    poisson_maxiter=1000,
    poisson_init="est",
    poisson_clip_to_max_gt=False,
    poisson_auto_flip=True,
    poisson_est_affine=True,
    poisson_smooth_est=True,
    use_p_affinity=True,
    p_only_gate=False,
    kernels=(3,5),
    kappa_min=0.03,
    kappa_max=0.5,
    anchor_learnable=False,
    anchor_mode="map",
    anchor_alpha=0.1,
    min_gate=0.0,
    min_alpha=0.0,
):
    return SimpleNamespace(
        dmax=float(dmax),
        steps=int(steps),
        geometry=str(geometry),
        use_sparse=bool(use_sparse),
        use_residual=bool(use_residual),

        poisson_only=False,
        use_poisson=bool(use_poisson),
        poisson_tol=float(poisson_tol),
        poisson_maxiter=int(poisson_maxiter),
        poisson_init=str(poisson_init),
        poisson_clip_to_max_gt=bool(poisson_clip_to_max_gt),
        poisson_auto_flip=bool(poisson_auto_flip),
        poisson_est_affine=bool(poisson_est_affine),
        poisson_smooth_est=bool(poisson_smooth_est),

        use_p_affinity=bool(use_p_affinity),
        p_only_gate=bool(p_only_gate),

        kernels=tuple(kernels),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),

        anchor_learnable=bool(anchor_learnable),
        anchor_mode=str(anchor_mode),
        anchor_alpha=float(anchor_alpha),

        min_gate=float(min_gate),
        min_alpha=float(min_alpha),
    )

# ============================ main visualize ============================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="vis_kitti_mcprop")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dmax", type=float, default=80.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--geometry", type=str, default="hyper", choices=["hyper","ellip"])
    ap.add_argument("--use-residual", action="store_true")
    ap.add_argument("--use-p-affinity", action="store_true")
    ap.add_argument("--p-only-gate", action="store_true")
    ap.add_argument("--require-est", action="store_true", help="est 없으면 스킵")
    ap.add_argument("--emode", type=str, default="precomputed", choices=["precomputed","zeros"],
                    help="E_norm: est/ 사용(precomputed) or zeros")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dmax = float(args.dmax)

    # scan pairs
    pairs = scan_kitti_pairs(args.root, require_est=(args.emode=="precomputed" and args.require_est))
    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check folder layout under --root.")

    # model (cfg must match training flags)
    cfg = make_cfg(
        dmax=dmax,
        steps=args.steps,
        geometry=("ellip" if args.geometry.startswith("ellip") else "hyper"),
        use_sparse=True,
        use_residual=bool(args.use_residual),
        use_p_affinity=bool(args.use_p_affinity),
        p_only_gate=bool(args.p_only_gate),
    )
    model = MCPropNet(cfg).to(device)
    missing, unexpected = load_checkpoint(model, args.ckpt, device=device, strict=args.strict)
    model.eval()

    print("[ckpt] loaded:", args.ckpt)
    print("[ckpt] missing keys   :", len(missing))
    print("[ckpt] unexpected keys:", len(unexpected))
    if len(missing) > 0: print("  e.g.", missing[:10])
    if len(unexpected) > 0: print("  e.g.", unexpected[:10])

    # outputs
    out_png16 = ensure_dir(os.path.join(args.outdir, "depth_png16_mm"))
    out_jet   = ensure_dir(os.path.join(args.outdir, "depth_jet"))
    out_maps  = ensure_dir(os.path.join(args.outdir, "maps"))
    out_rgb   = ensure_dir(os.path.join(args.outdir, "rgb"))

    saved = 0
    pbar = tqdm(pairs, ncols=120, desc="[VIS][KITTI]")
    for stem, img_p, gt_p, sp_p, est_p in pbar:
        if saved >= args.num:
            break

        rgb = _read_rgb(img_p)
        DL_full = _read_depth(sp_p)  # meters (if 16bit mm png -> converted)
        H, W = rgb.shape[:2]
        if DL_full.shape != (H,W): DL_full = _resize_like(DL_full, (H,W))

        ML = (DL_full > 0).astype(np.float32)

        GT = None
        VM = None
        if gt_p is not None:
            GT = _read_depth(gt_p)
            if GT.shape != (H,W): GT = _resize_like(GT, (H,W))
            VM = (GT > 0).astype(np.float32)

        # E_norm
        if args.emode == "zeros":
            E01 = np.zeros((H,W), np.float32)
        else:
            if est_p is None:
                if args.require_est:
                    continue
                E01 = np.zeros((H,W), np.float32)
            else:
                raw = _read_depth(est_p)
                if raw.ndim == 2:
                    if raw.shape != (H,W): raw = _resize_like(raw, (H,W))
                    Er = raw.astype(np.float32)
                else:
                    if raw.shape[:2] != (H,W):
                        raw = np.array(Image.fromarray(raw.astype(np.uint8)).resize((W,H), Image.BILINEAR))
                    Er = _best_gray_from_rgb(raw.astype(np.uint8), DL_full, ML) * 255.0

                # mask for minmax: GT valid if available else use ML anchors
                mask = VM if VM is not None else (ML > 0).astype(np.float32)
                E01 = _minmax01(Er, mask=mask)
                E01 = _orient_by_sparse(E01, DL_full, ML)

        # to torch (I in [0,1])
        I = (rgb.astype(np.float32) / 255.0).transpose(2,0,1)[None, ...]
        tI  = torch.from_numpy(I).to(device)
        tDL = torch.from_numpy(DL_full[None,None]).to(device)
        tML = torch.from_numpy(ML[None,None]).to(device)
        tE  = torch.from_numpy(E01[None,None]).to(device)

        Dt, aux = model(tI, tDL, tML, tE)  # (1,1,H,W), aux dict

        tag = f"{saved:05d}"
        base = os.path.basename(stem)

        # save rgb
        cv2.imwrite(os.path.join(out_rgb, f"{tag}_{base}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        dt = Dt[0,0].detach().cpu().numpy()
        p  = aux["P"][0,0].detach().cpu().numpy()
        d0 = aux["D0"][0,0].detach().cpu().numpy()

        # depth (16-bit mm)
        save_png16_mm(os.path.join(out_png16, f"{tag}_{base}_Dt.png"), dt, dmax)
        save_png16_mm(os.path.join(out_png16, f"{tag}_{base}_P.png"),  p,  dmax)
        save_png16_mm(os.path.join(out_png16, f"{tag}_{base}_D0.png"), d0, dmax)

        # colored depth (0~dmax)
        save_colormap(os.path.join(out_jet, f"{tag}_{base}_Dt_jet.png"), dt, vmin=0.0, vmax=dmax)
        save_colormap(os.path.join(out_jet, f"{tag}_{base}_P_jet.png"),  p,  vmin=0.0, vmax=dmax)
        save_colormap(os.path.join(out_jet, f"{tag}_{base}_D0_jet.png"), d0, vmin=0.0, vmax=dmax)

        # alpha / sigma / residual / masks
        alpha = aux["alpha"][0,0].detach().cpu().numpy()
        save_colormap(os.path.join(out_maps, f"{tag}_{base}_alpha_jet.png"), alpha, vmin=0.0, vmax=1.0)
        save_gray_u8(os.path.join(out_maps, f"{tag}_{base}_alpha_gray.png"), alpha, vmin=0.0, vmax=1.0)

        sigma = aux["sigma"][0].detach().cpu().numpy()  # (K,H,W)
        for k in range(sigma.shape[0]):
            save_colormap(os.path.join(out_maps, f"{tag}_{base}_sigma{k}_jet.png"), sigma[k], vmin=0.0, vmax=1.0)

        save_gray_u8(os.path.join(out_maps, f"{tag}_{base}_ML.png"), ML, vmin=0.0, vmax=1.0)
        save_colormap(os.path.join(out_maps, f"{tag}_{base}_E01_jet.png"), E01, vmin=0.0, vmax=1.0)

        if aux.get("residual", None) is not None:
            r = aux["residual"][0,0].detach().cpu().numpy()
            vmax = float(np.percentile(np.abs(r), 99)) + 1e-6
            save_colormap(os.path.join(out_maps, f"{tag}_{base}_residual_jet.png"), r, vmin=-vmax, vmax=vmax)

        # GT / error maps if available
        if GT is not None and VM is not None:
            save_png16_mm(os.path.join(out_png16, f"{tag}_{base}_GT.png"), GT, dmax)
            save_colormap(os.path.join(out_jet,   f"{tag}_{base}_GT_jet.png"), GT, vmin=0.0, vmax=dmax)

            err = (dt - GT) * VM
            vmax = float(np.percentile(np.abs(err[VM>0]), 99)) + 1e-6 if (VM>0).any() else 1.0
            save_colormap(os.path.join(out_maps, f"{tag}_{base}_errDt-GT_jet.png"), err, vmin=-vmax, vmax=vmax)

        saved += 1
        pbar.set_postfix(saved=saved)

    print(f"\nSaved {saved} samples to: {args.outdir}")

if __name__ == "__main__":
    main()
