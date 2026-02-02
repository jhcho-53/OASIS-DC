#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat KITTI K-shot(train) 폴더에서 sparse + estimation(0..255)으로
8/32-lines sparseN 및 pseudoN(= Poisson 보간 결과) 생성 + 시각화 저장.

입력:  ds_root/{sparse, estim}  (+ calib by date)
출력:  ds_root/{sparse8, pseudo8, sparse32, pseudo32}/<basename>.png
시각화: viz_root/{sparse8, pseudo8, sparse32, pseudo32}/<basename>.png  (옵션 --save-viz)

시각화는 'reversed JET'의 상단(near 구간)을 보라색으로 치환해
가까움=보라 → 파랑(청록/녹색 경유) → 빨강=멀리 가 되도록 합니다.
"""

import os, re, argparse, glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import cv2
from tqdm import tqdm

from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

# -------------------- I/O helpers --------------------

def load_depth_m_u16(path: str, scale: float) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im.astype(np.float32) / float(scale))

def load_estim8_gray(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)  # 0..255

def save_depth_u16(path: str, depth_m: np.ndarray, scale: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, np.clip(depth_m * float(scale), 0, 65535).astype(np.uint16))

def read_hw(path: str) -> Optional[Tuple[int,int]]:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None: return None
    return im.shape[:2]

# ---------- VIZ: Reversed JET + 상단(near) 보라 치환 ----------

def build_reversed_jet_with_purple_near(
    purple_bgr=(128, 0, 128), top_bins: int = 32, gamma: float = 2.0
) -> np.ndarray:
    """
    1) 기본 JET LUT(0=파랑, 255=빨강)를 만든 뒤
    2) LUT를 역순으로 뒤집어(0=빨강, 255=파랑),
    3) '상단(top, near)' 구간(기본 32 step)을 보라색으로 부드럽게 치환(가중 γ)합니다.

    결과: idx가 클수록(near) 보라에 가깝고, 점차 파랑→청록→녹색→노랑→빨강(멀리).
    """
    base = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)   # (256,1,3) BGR
    lut  = base[::-1, :, :].copy()  # reversed JET: 0=red, 255=blue

    tb = int(max(1, min(255, top_bins)))
    start = 256 - tb
    purple = np.array(list(purple_bgr), dtype=np.float32)
    denom = max(1, 255 - start)

    # start..255: t=0..1, t^gamma로 보라 영역을 near에 더 집중
    for i in range(start, 256):
        t = (i - start) / float(denom)        # 0..1
        a = t ** float(gamma)                  # 감마 보정(기본 2.0)
        base_col = lut[i, 0, :].astype(np.float32)
        # base(파랑 쪽) ↔ purple 블렌딩: near(255)로 갈수록 purple 쪽으로 붙임
        lut[i, 0, :] = np.clip((1.0 - a) * base_col + a * purple, 0, 255).astype(np.uint8)
    return lut

def depth_to_viz_bgr(depth_m: np.ndarray, max_m: float, lut: np.ndarray) -> np.ndarray:
    """
    depth(m) → idx = (1 - d/max)*255 (near=큰 idx) → LUT 적용.
    미측정(0)은 검정 처리.
    """
    d = depth_m.copy()
    mask = d > 0
    d = np.clip(d, 0.0, float(max_m))
    idx = (1.0 - (d / (float(max_m) + 1e-6))) * 255.0
    idx_u8 = np.clip(idx, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(idx_u8, lut)    # 커스텀 256x1x3 BGR LUT
    color[~mask] = 0
    return color

def save_viz_png(path: str, depth_m: np.ndarray, viz_max_m: float, lut: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    col = depth_to_viz_bgr(depth_m, viz_max_m, lut)
    cv2.imwrite(path, col)

# -------------------- Calib by DATE --------------------

def read_kv_txt(path: str) -> Dict[str, np.ndarray]:
    d = {}
    with open(path, 'r') as f:
        for ln in f:
            if ':' not in ln: continue
            k, vals = ln.split(':', 1)
            try:
                d[k.strip()] = np.array([float(x) for x in vals.split()], dtype=np.float64)
            except ValueError:
                pass
    return d

def parse_date_from_name(basename: str) -> str:
    m = re.search(r'(20\d{2}_\d{2}_\d{2})', basename)
    if not m:
        raise ValueError(f"Date not found in name: {basename}")
    return m.group(1)

def parse_cam_id_from_name(basename: str) -> int:
    m = re.search(r'_(0[23])\.png$', basename)
    return int(m.group(1)) if m else 2

def load_calibs_by_date(calib_root: str, date_str: str, cam_id: int):
    date_dir = os.path.join(calib_root, date_str)
    cam_txt  = os.path.join(date_dir, 'calib_cam_to_cam.txt')
    velo_txt = os.path.join(date_dir, 'calib_velo_to_cam.txt')
    if not (os.path.exists(cam_txt) and os.path.exists(velo_txt)):
        raise FileNotFoundError(f"Missing calib files under {date_dir}")

    cam_kv  = read_kv_txt(cam_txt)
    velo_kv = read_kv_txt(velo_txt)

    P_key = f'P_rect_0{cam_id}'
    if P_key not in cam_kv:
        raise KeyError(f"{P_key} not in {cam_txt}")
    P  = cam_kv[P_key].reshape(3,4)
    R0 = (cam_kv['R_rect_00'] if 'R_rect_00' in cam_kv else cam_kv['R0_rect']).reshape(3,3)

    if 'Tr' in velo_kv:
        Tr = velo_kv['Tr'].reshape(3,4)
    else:
        if 'R' not in velo_kv or 'T' not in velo_kv:
            raise KeyError(f"Tr or (R,T) not in {velo_txt}")
        R = velo_kv['R'].reshape(3,3); T = velo_kv['T'].reshape(3,1)
        Tr = np.hstack([R, T])

    return P.astype(np.float64), R0.astype(np.float64), Tr.astype(np.float64)

# -------------------- N-lines degrade --------------------

def ring_id_from_xyz(x, y, z):
    elev = np.arctan2(z, np.hypot(x, y))
    r = np.round((elev - np.deg2rad(-24.9)) / np.deg2rad(26.9) * 63.0).astype(np.int32)
    return np.clip(r, 0, 63)

def select_mask_for_N_lines(rings, N):
    assert 64 % N == 0, "N must divide 64"
    keep = np.arange(0, 64, 64 // N, dtype=np.int32)
    return np.isin(rings, keep)

def degrade_sparse_to_Nlines_m(sparse_m: np.ndarray,
                               P: np.ndarray,
                               R0_rect: np.ndarray,
                               Tr_velo_to_cam: np.ndarray,
                               N: int) -> np.ndarray:
    mask = sparse_m > 0
    if not np.any(mask):
        return np.zeros_like(sparse_m, dtype=np.float32)
    v, u = np.where(mask); Z = sparse_m[v, u].astype(np.float64)
    fx, fy = P[0,0], P[1,1]; cx, cy = P[0,2], P[1,2]
    X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy

    T = np.eye(4, dtype=np.float64); T[:3,:4] = Tr_velo_to_cam
    Trect = np.eye(4, dtype=np.float64); Trect[:3,:3] = R0_rect
    T_cam2velo = np.linalg.inv(Trect @ T)
    X_cam  = np.column_stack([X, Y, Z, np.ones_like(Z)])
    X_velo = (T_cam2velo @ X_cam.T).T[:, :3]

    rings = ring_id_from_xyz(X_velo[:,0], X_velo[:,1], X_velo[:,2])
    keep  = select_mask_for_N_lines(rings, N)

    out = np.zeros_like(sparse_m, dtype=np.float32)
    out[v[keep], u[keep]] = sparse_m[v[keep], u[keep]].astype(np.float32)
    return out

# -------------------- Poisson (unweighted) --------------------

def build_Laplacian(H: int, W: int) -> sp.csr_matrix:
    N = H * W
    main = np.full(N, 4, np.float32)
    off  = np.full(N, -1, np.float32)
    A = sp.diags([main, off, off, off, off], [0, -1, +1, -W, +W], format='csr')
    # block-row wrap 제거
    for i in range(H):
        L = i * W; R = i * W + (W - 1)
        if L - 1 >= 0: A[L, L-1] = 0; A[L-1, L] = 0
        if R + 1 < N: A[R, R+1] = 0; A[R+1, R] = 0
    return A

def get_A_base(h: int, w: int, cache_dir: str) -> sp.csr_matrix:
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"A_base_{h}x{w}.npz")
    if os.path.exists(fn): return sp.load_npz(fn)
    A = build_Laplacian(h, w); sp.save_npz(fn, A); return A

def _border_mask(h: int, w: int, thick: int = 1) -> np.ndarray:
    t = max(0, int(thick))
    m = np.zeros((h, w), dtype=bool)
    if t == 0: return m
    t = min(t, h//2, w//2)
    m[:t, :] = True; m[-t:, :] = True; m[:, :t] = True; m[:, -t:] = True
    return m

def _expand_anchors_local_avg(depth_m: np.ndarray, mask_sparse: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """희소 앵커를 반경 r로 확장하고 새로 추가된 고정값은 국소 평균으로 채움."""
    if radius <= 0:
        return depth_m, mask_sparse
    r = int(radius)
    k = (2*r+1, 2*r+1)
    sum_nb = cv2.blur((depth_m*mask_sparse.astype(np.float32)).astype(np.float32), k)
    cnt_nb = cv2.blur(mask_sparse.astype(np.float32), k)
    fill   = sum_nb / (cnt_nb + 1e-6)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv2.dilate(mask_sparse.astype(np.uint8), se) > 0
    new_mask = dil
    new_depth = depth_m.copy()
    new_depth[new_mask & (~mask_sparse)] = fill[new_mask & (~mask_sparse)]
    return new_depth, new_mask

def poisson_complete(
    sparse_m: np.ndarray,
    estim_gray_0_255: np.ndarray,
    A_base: sp.csr_matrix,
    *,
    border_policy: str = "none",   # 'none' | 'zero' | 'est'
    border_thick: int = 0,
    div_gauss_sigma: float = 0.0,
    div_mul: float = 1.0,
    div_auto: bool = False,
    div_target_frac: float = 0.25,
    anchor_dilate: int = 0
) -> np.ndarray:
    """
    순수 Poisson:
      A_uu x_u = b_u - A_uk v_k
    b = gain * div(∇I_norm), I_norm = I/255
    known set k: (sparse ⊕ anchor_dilate) ∪ border(정책)
    v_k: sparse/anchor -> LiDAR/지역평균, border -> 정책값
    """
    H, W = estim_gray_0_255.shape
    I = estim_gray_0_255.astype(np.float32)

    if div_gauss_sigma > 0:
        k = int(max(3, 2*round(div_gauss_sigma*3)+1))
        I = cv2.GaussianBlur(I, (k, k), div_gauss_sigma)

    # ∇I_norm, div
    I_norm = I / 255.0
    gx = cv2.Sobel(I_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I_norm, cv2.CV_32F, 0, 1, ksize=3)
    div = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)

    # --- gain (auto or fixed)
    if div_auto:
        valid = sparse_m > 0
        med_depth = float(np.median(sparse_m[valid])) if np.any(valid) else 10.0
        med_div   = float(np.median(np.abs(div))) + 1e-6
        gain = (div_target_frac * med_depth) / med_div
    else:
        gain = float(div_mul)
    b_full = (div * gain).reshape(-1)

    # --- anchors: sparse (+ optional dilate)
    mask_sparse = (sparse_m > 0)
    anchor_depth, anchor_mask = _expand_anchors_local_avg(sparse_m, mask_sparse, radius=anchor_dilate)

    # --- border policy
    if border_policy.lower() == "none":
        mask_known = anchor_mask
        v_known_img = anchor_depth
    else:
        mb = _border_mask(H, W, thick=border_thick)
        mask_known = (anchor_mask | mb)
        if border_policy.lower() == "zero":
            v_known_img = np.where(anchor_mask, anchor_depth, 0.0)
        elif border_policy.lower() == "est":
            v_known_img = np.where(anchor_mask, anchor_depth, I_norm)  # 주의: 단위 다름
        else:
            raise ValueError(f"Unknown border_policy: {border_policy}")

    idx_k = np.nonzero(mask_known.reshape(-1))[0]
    idx_u = np.nonzero(~mask_known.reshape(-1))[0]
    if idx_u.size == 0:
        return v_known_img.astype(np.float32)

    A_uu = A_base[idx_u][:, idx_u]
    A_uk = A_base[idx_u][:, idx_k]

    v_flat = v_known_img.reshape(-1).astype(np.float32)
    b_u  = b_full[idx_u] - A_uk.dot(v_flat[idx_k])
    x_u  = spsolve(A_uu, b_u).astype(np.float32)

    x = np.empty(H*W, np.float32)
    x[idx_k] = v_flat[idx_k]
    x[idx_u] = x_u
    out = x.reshape(H, W)

    max_gt = float(sparse_m.max())
    return np.clip(out, 0.0, max_gt) if max_gt > 0 else out

# -------------------- Main pipeline --------------------

def main():
    ap = argparse.ArgumentParser("Generate 8/32-lines sparse & pseudo via Poisson(using sparse+est), with reversed-JET-near-purple viz")
    ap.add_argument("--ds-root", required=True, help="kitti_k100_dataset/train (flat)")
    ap.add_argument("--calib-root", required=True, help="KITTI calib root organized by date folders")

    ap.add_argument("--n", nargs="+", type=int, default=[8, 32], help="N-lines list (each must divide 64)")
    ap.add_argument("--scale", type=float, default=256.0, help="u16 depth scale for IO")
    ap.add_argument("--require-w", type=int, default=1242)
    ap.add_argument("--require-h", type=int, default=375)

    # Poisson 옵션 / 안정화
    ap.add_argument("--border-policy", choices=["none","zero","est"], default="none",
                    help="경계 고정: none(기본) | zero | est(단위 불일치에 주의)")
    ap.add_argument("--border-thick", type=int, default=0, help="경계 두께(픽셀, border-policy!=none일 때)")
    ap.add_argument("--div-gauss", type=float, default=0.0, help="div 계산 전 estimation 가우시안 블러 sigma")
    ap.add_argument("--div-mul", type=float, default=1.0, help="div 스케일(고정 게인)")
    ap.add_argument("--div-auto", action="store_true", help="자동 게인 사용(깊이 스케일에 맞춤)")
    ap.add_argument("--div-target-frac", type=float, default=0.25,
                    help="자동 게인 시 목표 스케일: median(depth) * frac")
    ap.add_argument("--anchor-dilate", type=int, default=0,
                    help="희소 앵커 확장 반경(r픽셀). 새 앵커 값은 지역 평균으로 채움.")
    ap.add_argument("--cache-dir", default="", help="라플라시안 캐시 위치(미지정 시 ds_root/.poisson_cache)")

    # 8‑라인 자동 강화
    ap.add_argument("--robust8", action="store_true",
                    help="8line에 대해 div-gauss≥1.0, anchor-dilate≥2, div-auto 활성화")

    # VIZ
    ap.add_argument("--save-viz", action="store_true", help="Save near-purple reversed-JET visualizations")
    ap.add_argument("--viz-root", default="", help="Default: <ds-root>/viz")
    ap.add_argument("--viz-max-m", type=float, default=80.0)
    ap.add_argument("--jet-purple-bins", type=int, default=64,
                    help="reversed JET 상단에서 보라로 치환할 bin 개수(기본 48)")
    ap.add_argument("--jet-purple-gamma", type=float, default=2.0,
                    help="보라 블렌딩 감마(near에 보라 집중, 기본 2.0)")

    args = ap.parse_args()

    ds_root = Path(args.ds_root)
    sparse_dir = ds_root / "sparse"
    estim_dir  = ds_root / "estim"

    sparse_map = {os.path.basename(p): p for p in glob.glob(str(sparse_dir / "*.png"))}
    estim_map  = {os.path.basename(p): p for p in glob.glob(str(estim_dir  / "*.png"))}
    names = sorted(set(sparse_map) & set(estim_map))
    if not names:
        raise RuntimeError("No matching basenames between sparse and estim.")
    print(f"[INFO] samples={len(names)}  (scale={args.scale})")

    out_dirs, viz_dirs = {}, {}
    for N in args.n:
        assert 64 % N == 0, f"N must divide 64 (got {N})"
        out_dirs[f"sparse{N}"] = ds_root / f"sparse{N}"
        out_dirs[f"pseudo{N}"] = ds_root / f"pseudo{N}"
        out_dirs[f"sparse{N}"].mkdir(parents=True, exist_ok=True)
        out_dirs[f"pseudo{N}"].mkdir(parents=True, exist_ok=True)

    if args.save_viz:
        viz_root = Path(args.viz_root) if args.viz_root else (ds_root / "viz")
        for N in args.n:
            viz_dirs[f"sparse{N}"] = viz_root / f"sparse{N}"
            viz_dirs[f"pseudo{N}"] = viz_root / f"pseudo{N}"
            viz_dirs[f"sparse{N}"].mkdir(parents=True, exist_ok=True)
            viz_dirs[f"pseudo{N}"].mkdir(parents=True, exist_ok=True)
        print(f"[INFO] viz root = {viz_root} (viz_max_m={args.viz_max_m})")

    # LUT 준비 (한 번 생성해서 재사용)
    pr_lut = build_reversed_jet_with_purple_near(
        top_bins=args.jet_purple_bins, gamma=args.jet_purple_gamma
    )

    cache_dir = args.cache_dir if args.cache_dir else str(ds_root / ".poisson_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    calib_cache: Dict[Tuple[str,int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    req_h, req_w = int(args.require_h), int(args.require_w)
    A_base = get_A_base(req_h, req_w, cache_dir=cache_dir)

    pbar = tqdm(names, desc="Build 8/32-lines via Poisson + viz (near=purple -> far=red)", unit="img")
    for base in pbar:
        sp_path = sparse_map[base]
        es_path = estim_map[base]

        if read_hw(sp_path) != (req_h, req_w):
            pbar.set_postfix_str("skip size mismatch (sparse)"); continue
        if read_hw(es_path) != (req_h, req_w):
            pbar.set_postfix_str("skip size mismatch (estim)"); continue

        date_str = parse_date_from_name(base)
        cam_id   = parse_cam_id_from_name(base)
        key = (date_str, cam_id)
        if key not in calib_cache:
            P, R0, Tr = load_calibs_by_date(str(args.calib_root), date_str, cam_id)
            calib_cache[key] = (P, R0, Tr)
        else:
            P, R0, Tr = calib_cache[key]

        sparse_m = load_depth_m_u16(sp_path, args.scale)
        estim_g  = load_estim8_gray(es_path)

        for N in args.n:
            # 8‑라인 자동 보강
            div_gauss_here = args.div_gauss
            anchor_dilate  = args.anchor_dilate
            div_auto_here  = args.div_auto
            div_frac_here  = args.div_target_frac  # <-- NOTE: remove typo in final code (see below)
            if args.robust8 and N == 8:
                div_gauss_here = max(div_gauss_here, 1.0)
                anchor_dilate  = max(anchor_dilate, 2)
                div_auto_here  = True
                div_frac_here  = max(div_frac_here, 0.25)

            spN = degrade_sparse_to_Nlines_m(sparse_m, P, R0, Tr, N)

            psN = poisson_complete(
                spN, estim_g, A_base,
                border_policy=args.border_policy,
                border_thick=args.border_thick,
                div_gauss_sigma=div_gauss_here,
                div_mul=args.div_mul,
                div_auto=div_auto_here,
                div_target_frac=div_frac_here,
                anchor_dilate=anchor_dilate
            )

            save_depth_u16(str(out_dirs[f"sparse{N}"] / base), spN, args.scale)
            save_depth_u16(str(out_dirs[f"pseudo{N}"] / base), psN, args.scale)

            if args.save_viz:
                save_viz_png(str(viz_dirs[f"sparse{N}"] / base), spN, args.viz_max_m, pr_lut)
                save_viz_png(str(viz_dirs[f"pseudo{N}"] / base), psN, args.viz_max_m, pr_lut)

    print("[DONE] Generated:", ", ".join(out_dirs.keys()), "under", ds_root)
    if args.save_viz:
        roots = {str(p.parent) for p in viz_dirs.values()}
        print("[DONE] Viz saved under:", ", ".join(sorted(roots)))

if __name__ == "__main__":
    main()
