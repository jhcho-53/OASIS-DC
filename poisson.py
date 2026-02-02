#!/usr/bin/env python3
# eval_pseudo_single.py  (no-save)
import os, sys, argparse, glob, time, re
import numpy as np
from PIL import Image

# ---- import poisson_gpu (프로젝트 경로에 맞게) ----
from models.module import poisson_gpu

# ========================== 공통 유틸 ==========================
def _norm_key_from_path(path: str) -> str:
    """KITTI val_selection_cropped 스타일: 중간 토큰 제거 후 공통 키 생성."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'_(image|groundtruth_depth|velodyne_raw)_', '_', stem, flags=re.IGNORECASE)
    stem = re.sub(r'__+', '_', stem)
    return stem

def _index_dir_by_norm_key(dir_path: str, exts, recursive: bool = False):
    """지정 확장자를 스캔해 {정규화된 stem: full_path} 매핑 생성."""
    files, m = [], {}
    pattern_prefix = "**/*" if recursive else "*"
    for e in exts:
        files.extend(glob.glob(os.path.join(dir_path, f"{pattern_prefix}{e}"), recursive=recursive))
    for p in files:
        if os.path.isdir(p):  # 디렉토리는 스킵
            continue
        m[_norm_key_from_path(p)] = p
    return m

def _read_any_depth_est(path):
    """est/rgb_da용: npy/npz/이미지 모두 지원. float32 반환(상대 깊이)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr.squeeze() if arr.ndim == 3 else arr
    if ext == ".npz":
        z = np.load(path); arr = None
        for k in ["arr_0","rel","depth","D","d"]:
            if k in z: arr = z[k]; break
        if arr is None: arr = list(z.values())[0]
        arr = np.array(arr).astype(np.float32)
        return arr.squeeze() if arr.ndim == 3 else arr
    with Image.open(path) as im:
        im = im.convert("I") if im.mode not in ["I","I;16","I;16B","F","L"] else im
        return np.array(im).astype(np.float32)

def _read_kitti_depth_png(path):
    """KITTI: 16-bit PNG, depth[m] = val/256, 0=invalid."""
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.uint16)
    return (arr.astype(np.float32) / 256.0)

def _read_nyu_depth_mm(path):
    """NYUv2: depth_inpainted_mm (보통 mm). m로 변환."""
    with Image.open(path) as im:
        arr = np.array(im)
    arr = arr.astype(np.float32)
    if arr.max() > 50.0:  # mm로 판단되면 m로 변환
        arr = arr / 1000.0
    return arr

def _resize_like(arr, HW, is_float=True):
    H, W = HW
    mode = Image.BILINEAR if is_float else Image.NEAREST
    im = Image.fromarray(arr)
    im = im.resize((W, H), mode)
    return np.array(im)

def _minmax01(x, mask=None, eps=1e-6):
    v = x if mask is None else x[mask > 0]
    xmin, xmax = (np.min(v), np.max(v)) if (mask is not None and v.size>0) else (np.min(x), np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - xmin) / (xmax - xmin + eps)
    return np.clip(y.astype(np.float32), 0.0, 1.0)

def _corr_sign_pearson(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.mean(a * b))

def _auto_flip(E01, DL, ML, thr=0.0):
    """라이다/샘플 기준 상관 부호로 near/far 자동 뒤집기."""
    yy, xx = np.where(ML > 0)
    if len(yy) < 10:
        return E01
    e = E01[yy, xx].astype(np.float64)
    d = DL[yy, xx].astype(np.float64)
    corr = _corr_sign_pearson(e, d)
    return (1.0 - E01) if corr < -abs(thr) else E01

def _affine_calibrate(E01, DL, ML, dmax):
    """E01 ↦ a*E01+b (anchors=DL/dmax) → [0,1] 클립 → m 스케일."""
    yy, xx = np.where(ML > 0)
    if len(yy) < 10:
        return (E01 * dmax).astype(np.float32)
    x = E01[yy, xx].reshape(-1,1).astype(np.float64)
    y = (DL[yy, xx] / dmax).reshape(-1,1).astype(np.float64)
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b0 = float(theta[0,0]), float(theta[1,0])
    Em01 = np.clip(a*E01 + b0, 0.0, 1.0).astype(np.float32)
    return Em01 * float(dmax)

def _sample_sparse_from_dense(D_gt_m, n=500, seed=0):
    """NYU: GT에서 n개 샘플을 뽑아 sparse anchors 생성."""
    rng = np.random.default_rng(seed)
    yy, xx = np.where(D_gt_m > 0)
    if yy.size == 0:
        H, W = D_gt_m.shape
        return np.zeros((H,W), np.float32), np.zeros((H,W), np.uint8)
    sel = rng.choice(yy.size, size=min(n, yy.size), replace=False)
    y, x = yy[sel], xx[sel]
    DL = np.zeros_like(D_gt_m, np.float32)
    ML = np.zeros_like(D_gt_m, np.uint8)
    DL[y, x] = D_gt_m[y, x]; ML[y, x] = 1
    return DL, ML

# ========================== KITTI 평가 (no-save) ==========================
def evaluate_kitti(root, dmax=80.0, cg_tol=1e-4, cg_maxiter=400, cg_init="est",
                   cg_device="cuda:0", clip_to_max_gt=False,
                   recursive=False):
    est_dir = os.path.join(root, "est")
    vel_dir = os.path.join(root, "velodyne_raw")
    gt_dir  = os.path.join(root, "groundtruth_depth")
    if not (os.path.isdir(est_dir) and os.path.isdir(vel_dir) and os.path.isdir(gt_dir)):
        raise FileNotFoundError("KITTI: root/est, root/velodyne_raw, root/groundtruth_depth 모두 있어야 합니다.")

    est_map = _index_dir_by_norm_key(est_dir, [".npy",".npz",".png",".tif",".tiff",".exr"], recursive=recursive)
    vel_map = _index_dir_by_norm_key(vel_dir, [".png"], recursive=recursive)
    gt_map  = _index_dir_by_norm_key(gt_dir,  [".png"], recursive=recursive)

    keys = sorted(set(est_map) & set(vel_map) & set(gt_map))
    if not keys:
        raise RuntimeError("KITTI: 공통 stem 없음 (파일명 규칙/확장자/경로를 확인하세요)")

    mae_list, rmse_list = [], []
    for i, stem in enumerate(keys, 1):
        E_raw = _read_any_depth_est(est_map[stem])   # relative
        DL    = _read_kitti_depth_png(vel_map[stem]) # sparse [m]
        GT    = _read_kitti_depth_png(gt_map[stem])  # GT [m]

        H, W = DL.shape
        if E_raw.shape != (H, W):
            E_raw = _resize_like(E_raw, (H, W), is_float=True).astype(np.float32)
        ML = (DL > 0).astype(np.uint8)

        # prior 정렬
        norm_mask = (GT > 0).astype(np.uint8)
        if norm_mask.sum() < 10:
            norm_mask = ML
        E01 = _minmax01(E_raw, mask=norm_mask)
        E01 = _auto_flip(E01, DL, ML)
        Em  = _affine_calibrate(E01, DL, ML, dmax=dmax)

        # GPU Poisson
        P, st = poisson_gpu(
            sparse_m=DL.astype(np.float32), est_m=Em.astype(np.float32),
            tol=float(cg_tol), maxiter=int(cg_maxiter),
            device=str(cg_device), init=str(cg_init),
            clip_to_max_gt=bool(clip_to_max_gt)
        )

        valid = (GT > 0)
        if valid.sum() == 0:
            print(f"[{i}/{len(keys)}] {stem}: GT valid=0 → skip")
            continue
        diff = (P - GT)[valid]
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff*diff)))
        iters = st.get("cg_iters", None); tsec = st.get("time_sec", None)
        tstr = f"{tsec:.3f}s" if isinstance(tsec, (int,float)) else "-"
        print(f"[{i:04d}/{len(keys)}] {stem}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"| cg_iters={iters if iters is not None else '-'}  time={tstr}")
        mae_list.append(mae); rmse_list.append(rmse)

    if mae_list:
        print("\n=== KITTI Poisson(GPU) Pseudo-depth Evaluation ===")
        print(f"Frames: {len(mae_list)}")
        print(f"MAE : {np.mean(mae_list):.4f} m")
        print(f"RMSE: {np.sqrt(np.mean(np.square(rmse_list))):.4f} m")
    else:
        print("유효한 평가 샘플이 없습니다.")

# ========================== NYUv2 평가 (no-save) ==========================
def evaluate_nyu(root, dmax=10.0, cg_tol=1e-4, cg_maxiter=400, cg_init="est",
                 cg_device="cuda:0", n_sparse=500, sparse_seed=0,
                 recursive=False):
    est_dir = os.path.join(root, "rgb_MiDAS")
    gt_dir  = os.path.join(root, "depth_inpainted_mm")
    if not (os.path.isdir(est_dir) and os.path.isdir(gt_dir)):
        raise FileNotFoundError("NYUv2: root/rgb_da, root/depth_inpainted_mm 모두 있어야 합니다.")

    est_map = _index_dir_by_norm_key(est_dir, [".npy",".npz",".png",".tif",".tiff",".exr"], recursive=recursive)
    gt_map  = _index_dir_by_norm_key(gt_dir,  [".png",".tif",".tiff",".npy",".npz"], recursive=recursive)

    keys = sorted(set(est_map) & set(gt_map))
    if not keys:
        raise RuntimeError("NYUv2: 공통 stem 없음 (파일명 규칙/확장자/경로를 확인하세요)")

    mae_list, rmse_list = [], []
    for i, stem in enumerate(keys, 1):
        E_raw = _read_any_depth_est(est_map[stem])   # relative
        GT    = _read_nyu_depth_mm(gt_map[stem])     # dense [m]

        H, W = GT.shape
        if E_raw.shape != (H, W):
            E_raw = _resize_like(E_raw, (H, W), is_float=True).astype(np.float32)

        DL, ML = _sample_sparse_from_dense(GT, n=n_sparse, seed=sparse_seed)

        # prior 정렬
        norm_mask = (GT > 0).astype(np.uint8)
        E01 = _minmax01(E_raw, mask=norm_mask)
        E01 = _auto_flip(E01, DL, ML)
        Em  = _affine_calibrate(E01, DL, ML, dmax=dmax)

        # GPU Poisson
        P, st = poisson_gpu(
            sparse_m=DL.astype(np.float32), est_m=Em.astype(np.float32),
            tol=float(cg_tol), maxiter=int(cg_maxiter),
            device=str(cg_device), init=str(cg_init),
            clip_to_max_gt=False
        )

        valid = (GT > 0)
        if valid.sum() == 0:
            print(f"[{i}/{len(keys)}] {stem}: GT valid=0 → skip")
            continue
        diff = (P - GT)[valid]
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff*diff)))
        iters = st.get("cg_iters", None); tsec = st.get("time_sec", None)
        tstr = f"{tsec:.3f}s" if isinstance(tsec, (int,float)) else "-"
        print(f"[{i:04d}/{len(keys)}] {stem}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"| cg_iters={iters if iters is not None else '-'}  time={tstr}")
        mae_list.append(mae); rmse_list.append(rmse)

    if mae_list:
        print("\n=== NYUv2 Poisson(GPU) Pseudo-depth Evaluation ===")
        print(f"Frames: {len(mae_list)}")
        print(f"MAE : {np.mean(mae_list):.4f} m")
        print(f"RMSE: {np.sqrt(np.mean(np.square(rmse_list))):.4f} m")
    else:
        print("유효한 평가 샘플이 없습니다.")

# ============================ CLI =============================
def main():
    ap = argparse.ArgumentParser(description="Pseudo-depth Poisson(GPU) evaluation for KITTI / NYUv2 (no-save)")
    ap.add_argument("--dataset", type=str, required=True, choices=["kitti","nyu"])
    ap.add_argument("--root", type=str, required=True, help="dataset root")
    ap.add_argument("--dmax", type=float, default=None, help="KITTI=80.0, NYUv2=10.0 (기본값 자동)")
    # CG/Poisson 설정
    ap.add_argument("--cg-tol", type=float, default=1e-4)
    ap.add_argument("--cg-maxiter", type=int, default=400)
    ap.add_argument("--cg-init", type=str, default="est", choices=["est","zero"])
    ap.add_argument("--cg-device", type=str, default="cuda:0")
    ap.add_argument("--clip-to-max-gt", action="store_true")
    # NYU sparsity
    ap.add_argument("--n-sparse", type=int, default=500, help="NYUv2 sparse 샘플 수")
    ap.add_argument("--sparse-seed", type=int, default=0)
    # 기타
    ap.add_argument("--recursive", action="store_true", help="하위 폴더까지 검색")
    args = ap.parse_args()

    dmax = args.dmax if args.dmax is not None else (80.0 if args.dataset == "kitti" else 10.0)

    if args.dataset == "kitti":
        evaluate_kitti(
            root=args.root, dmax=dmax,
            cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter, cg_init=args.cg_init,
            cg_device=args.cg_device, clip_to_max_gt=args.clip_to_max_gt,
            recursive=args.recursive
        )
    else:
        evaluate_nyu(
            root=args.root, dmax=dmax,
            cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter, cg_init=args.cg_init,
            cg_device=args.cg_device, n_sparse=args.n_sparse, sparse_seed=args.sparse_seed,
            recursive=args.recursive
        )

if __name__ == "__main__":
    main()
