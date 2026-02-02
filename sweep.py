# eval_pseudo_sweep.py
import os, sys, argparse, glob, time, re, csv
import numpy as np
from PIL import Image
from itertools import product
from collections import Counter

# ---- import poisson_gpu (환경 경로에 맞게) ----
from models.module import poisson_gpu

# ======================== 스윕 설정 (A만) ========================
GRID = {
    "norm_mask":   ["gt", "union"],          # A1
    "flip_method": ["pearson", "spearman"],  # A2
    "flip_thr":    [0.0],                    # A2
    "affine_mode": ["ols", "irls"],          # A3
    "affine_log":  [False],                  # A3
    "irls_iters":  [3],                      # A3
    "huber_delta": [1.0],                    # A3 (미사용)
}
# ========================= 고정 설정 ===========================
CG_TOL     = float(os.environ.get("POISSON_TOL", 1e-4))
CG_MAXITER = int(os.environ.get("POISSON_MAXITER", 400))
CG_INIT    = os.environ.get("POISSON_INIT", "est")      # "est" | "zero"
CG_DEVICE  = os.environ.get("POISSON_DEVICE", "cuda:0") # "cuda:0" | "cpu"

N_SPARSE_NYU = 500
SPARSE_SEED  = 0
# ===============================================================

# -------------------- 공통: 파일 인덱싱/로드 --------------------
def _norm_key_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # 중간 토큰 제거 (val_selection_cropped 규칙)
    stem = re.sub(r'_(image|groundtruth_depth|velodyne_raw)_', '_', stem, flags=re.IGNORECASE)
    stem = re.sub(r'__+', '_', stem)
    return stem

def _index_dir_by_norm_key(dir_path: str, exts, recursive=True):
    files, m = [], {}
    for e in exts:
        pattern = "**/*" + e if recursive else "*" + e
        files.extend(glob.glob(os.path.join(dir_path, pattern), recursive=recursive))
    for p in files:
        if os.path.isdir(p):  #폴더 건너뛰기
            continue
        m[_norm_key_from_path(p)] = p
    return m

def _read_any_depth_est(path):
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
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.uint16)
    return (arr.astype(np.float32) / 256.0)  # meters

def _read_nyu_depth_mm(path):
    with Image.open(path) as im:
        arr = np.array(im)
    arr = arr.astype(np.float32)
    if arr.max() > 50.0:  # mm -> m
        arr = arr / 1000.0
    return arr

def _resize_like(arr, HW, is_float=True):
    H, W = HW
    mode = Image.BILINEAR if is_float else Image.NEAREST
    im = Image.fromarray(arr)
    im = im.resize((W, H), mode)
    return np.array(im)

# -------------------- pseudo 전처리(공통) --------------------
def _box3_once(x: np.ndarray) -> np.ndarray:
    pad = 1
    xp = np.pad(x, ((pad,pad),(pad,pad)), mode="reflect").astype(np.float32)
    out = (
        xp[0:-2,0:-2] + xp[0:-2,1:-1] + xp[0:-2,2:] +
        xp[1:-1,0:-2] + xp[1:-1,1:-1] + xp[1:-1,2:] +
        xp[2:  ,0:-2] + xp[2:  ,1:-1] + xp[2:  ,2:]
    ) / 9.0
    return out.astype(np.float32)

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

def _corr_sign_spearman(a, b):
    def rank(v):
        o = np.argsort(v)
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(1, v.size+1, dtype=np.float64)
        return r
    ra, rb = rank(a), rank(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float(np.mean(ra * rb))

def _auto_flip(E01, DL, ML, method="pearson", thr=0.0):
    yy, xx = np.where(ML > 0)
    if len(yy) < 10:
        return E01
    e = E01[yy, xx].astype(np.float64)
    d = DL[yy, xx].astype(np.float64)
    if method == "spearman":
        corr = _corr_sign_spearman(e, d)
    elif method == "pearson":
        corr = _corr_sign_pearson(e, d)
    else:
        corr = 1.0
    return (1.0 - E01) if corr < -abs(thr) else E01

def _choose_norm_mask(policy: str, GT: np.ndarray, ML: np.ndarray):
    if policy == "gt":
        return (GT > 0).astype(np.uint8)
    if policy == "lidar":
        return (ML > 0).astype(np.uint8)
    if policy == "union":
        return ((GT > 0) | (ML > 0)).astype(np.uint8)
    return None  # "all"

def _affine_calibrate(E01, DL, ML, dmax, mode="ols", irls_iters=3, huber_delta=1.0, use_log=False):
    yy, xx = np.where(ML > 0)
    if len(yy) < 10:
        Em01 = E01
    else:
        x = E01[yy, xx].reshape(-1,1).astype(np.float64)
        y = (DL[yy, xx] / dmax).reshape(-1,1).astype(np.float64)
        if use_log:
            y = np.log(np.clip(y, 1e-6, 1.0))
        A = np.concatenate([x, np.ones_like(x)], axis=1)
        if mode == "irls":
            w = np.ones((A.shape[0],1), dtype=np.float64)
            for _ in range(int(irls_iters)):
                Aw, yw = A*w, y*w
                theta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
                r = (A @ theta - y)
                c = 1.345 * (np.median(np.abs(r)) + 1e-9)
                w = 1.0 / np.maximum(1.0, np.abs(r)/c)
        elif mode == "huber":
            w = np.ones((A.shape[0],1), dtype=np.float64); delta = float(huber_delta)
            for _ in range(5):
                Aw, yw = A*w, y*w
                theta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
                r = (A @ theta - y)
                w = 1.0 / np.maximum(1.0, np.abs(r)/delta)
        else:  # OLS
            theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b0 = float(theta[0,0]), float(theta[1,0])
        Em01 = np.clip(a*E01 + b0, 0.0, 1.0).astype(np.float32)
        if use_log:
            Em01 = np.clip(np.exp(Em01), 0.0, 1.0).astype(np.float32)
    return Em01.astype(np.float32) * float(dmax)

# -------------------- 진단 유틸 --------------------
def _debug_keys(title, m, k=5):
    ks = list(m.keys())
    print(f"[DEBUG] {title}: {len(ks)} files")
    for s in ks[:k]:
        print(f"  - {s}")

def _diagnose_intersection(est_map, vel_map, gt_map):
    est_k, vel_k, gt_k = set(est_map), set(vel_map), set(gt_map)
    inter = est_k & vel_k & gt_k
    if inter:
        return inter
    print("\n[DIAG] 교집합 비어 있음 → 폴더별 key 요약")
    _debug_keys("est", est_map)
    _debug_keys("vel", vel_map)
    _debug_keys("gt ", gt_map)
    # 어디가 빠졌는지 Top-5 보여주기
    miss_est = (vel_k & gt_k) - est_k
    miss_vel = (est_k & gt_k) - vel_k
    miss_gt  = (est_k & vel_k) - gt_k
    print(f"[DIAG] missing in est: {list(miss_est)[:5]}")
    print(f"[DIAG] missing in vel: {list(miss_vel)[:5]}")
    print(f"[DIAG] missing in gt : {list(miss_gt)[:5]}")
    return inter

# -------------------- KITTI 한 조합 평가 --------------------
def evaluate_once_kitti(root, dmax, cfg) -> dict:
    # est 폴더가 없으면 image 폴더를 est로 사용 (모노 prior가 image/에 있을 때)
    est_dir = os.path.join(root, "est")
    if not os.path.isdir(est_dir):
        alt = os.path.join(root, "image")
        if os.path.isdir(alt):
            est_dir = alt
    vel_dir = os.path.join(root, "velodyne_raw")
    gt_dir  = os.path.join(root, "groundtruth_depth")
    if not (os.path.isdir(est_dir) and os.path.isdir(vel_dir) and os.path.isdir(gt_dir)):
        raise FileNotFoundError("KITTI: root 하단에 est(or image) / velodyne_raw / groundtruth_depth 필요")

    est_map = _index_dir_by_norm_key(est_dir, [".npy",".npz",".png",".tif",".tiff",".exr"], recursive=True)
    vel_map = _index_dir_by_norm_key(vel_dir, [".png",".npy",".npz"], recursive=True)
    gt_map  = _index_dir_by_norm_key(gt_dir,  [".png",".npy",".npz"], recursive=True)

    common = sorted(_diagnose_intersection(est_map, vel_map, gt_map))
    if not common:
        raise RuntimeError("KITTI: 공통 stem 없음 (위 DIAG 로그 참고)")

    mae_list, rmse_list, t_list = [], [], []
    t0 = time.time()
    for stem in common:
        E_raw = _read_any_depth_est(est_map[stem])
        DL    = _read_kitti_depth_png(vel_map[stem])   # sparse [m] (PNG 권장)
        GT    = _read_kitti_depth_png(gt_map[stem])    # GT [m]    (PNG 권장)
        H, W = DL.shape
        if E_raw.shape != (H, W):
            E_raw = _resize_like(E_raw, (H, W), is_float=True).astype(np.float32)
        ML = (DL > 0).astype(np.uint8)

        nm  = _choose_norm_mask(cfg["norm_mask"], GT, ML)
        E01 = _minmax01(E_raw, mask=nm)
        E01 = _auto_flip(E01, DL, ML, method=cfg["flip_method"], thr=float(cfg["flip_thr"]))
        Em  = _affine_calibrate(E01, DL, ML, dmax=dmax,
                                mode=cfg["affine_mode"], irls_iters=int(cfg["irls_iters"]),
                                huber_delta=float(cfg["huber_delta"]), use_log=bool(cfg["affine_log"]))

        P, st = poisson_gpu(
            sparse_m=DL.astype(np.float32), est_m=Em.astype(np.float32),
            tol=CG_TOL, maxiter=CG_MAXITER, device=CG_DEVICE, init=CG_INIT,
            clip_to_max_gt=False
        )
        valid = (GT > 0)
        if valid.sum() == 0:
            continue
        diff = (P - GT)[valid]
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff*diff)))
        mae_list.append(mae); rmse_list.append(rmse)
        t_list.append(float(st.get("time_sec", 0.0)))

    return {
        "frames": len(mae_list),
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "rmse": float(np.mean(rmse_list)) if mae_list else float("nan"),
        "time_avg": float(np.mean(t_list)) if t_list else float("nan"),
        "elapsed": float(time.time() - t0),
    }

# -------------------- NYUv2 한 조합 평가 --------------------
def _sample_sparse_from_dense(D_gt_m, n=500, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.where(D_gt_m > 0)
    if yy.size == 0:
        H, W = D_gt_m.shape
        return np.zeros((H,W), np.float32), np.zeros((H,W), np.uint8)
    sel = rng.choice(yy.size, size=min(n, yy.size), replace=False)
    y, x = yy[sel], xx[sel]
    DL = np.zeros_like(D_gt_m, np.float32); ML = np.zeros_like(D_gt_m, np.uint8)
    DL[y, x] = D_gt_m[y, x]; ML[y, x] = 1
    return DL, ML

def evaluate_once_nyu(root, dmax, cfg) -> dict:
    est_dir, gt_dir = os.path.join(root, "rgb_da"), os.path.join(root, "depth_inpainted_mm")
    if not (os.path.isdir(est_dir) and os.path.isdir(gt_dir)):
        raise FileNotFoundError("NYUv2: root 하단에 rgb_da / depth_inpainted_mm 필요")

    est_map = _index_dir_by_norm_key(est_dir, [".npy",".npz",".png",".tif",".tiff",".exr"], recursive=True)
    gt_map  = _index_dir_by_norm_key(gt_dir,  [".png",".tif",".tiff",".npy",".npz"], recursive=True)
    common = sorted(set(est_map).intersection(gt_map))
    if not common:
        # 진단
        _debug_keys("nyu est", est_map)
        _debug_keys("nyu gt ", gt_map)
        raise RuntimeError("NYUv2: 공통 stem 없음")

    mae_list, rmse_list, t_list = [], [], []
    t0 = time.time()
    for stem in common:
        E_raw = _read_any_depth_est(est_map[stem])     # 상대 깊이
        GT    = _read_nyu_depth_mm(gt_map[stem])       # dense GT [m]
        H, W = GT.shape
        if E_raw.shape != (H, W):
            E_raw = _resize_like(E_raw, (H, W), is_float=True).astype(np.float32)

        DL, ML = _sample_sparse_from_dense(GT, n=N_SPARSE_NYU, seed=SPARSE_SEED)

        nm  = _choose_norm_mask(cfg["norm_mask"], GT, ML)
        E01 = _minmax01(E_raw, mask=nm)
        E01 = _auto_flip(E01, DL, ML, method=cfg["flip_method"], thr=float(cfg["flip_thr"]))
        Em  = _affine_calibrate(E01, DL, ML, dmax=dmax,
                                mode=cfg["affine_mode"], irls_iters=int(cfg["irls_iters"]),
                                huber_delta=float(cfg["huber_delta"]), use_log=bool(cfg["affine_log"]))

        P, st = poisson_gpu(
            sparse_m=DL.astype(np.float32), est_m=Em.astype(np.float32),
            tol=CG_TOL, maxiter=CG_MAXITER, device=CG_DEVICE, init=CG_INIT,
            clip_to_max_gt=False
        )
        valid = (GT > 0)
        if valid.sum() == 0:
            continue
        diff = (P - GT)[valid]
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff*diff)))
        mae_list.append(mae); rmse_list.append(rmse)
        t_list.append(float(st.get("time_sec", 0.0)))

    return {
        "frames": len(mae_list),
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "rmse": float(np.mean(rmse_list)) if mae_list else float("nan"),
        "time_avg": float(np.mean(t_list)) if t_list else float("nan"),
        "elapsed": float(time.time() - t0),
    }

# -------------------- 스윕 루프 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="KITTI or NYUv2 root")
    ap.add_argument("--dataset", type=str, required=True, choices=["kitti","nyu"])
    ap.add_argument("--dmax", type=float, default=None, help="KITTI=80.0, NYU=10.0")
    ap.add_argument("--out-csv", type=str, required=True, help="스윕 요약표 저장 경로")
    args = ap.parse_args()

    dmax = args.dmax if args.dmax is not None else (80.0 if args.dataset == "kitti" else 10.0)

    keys = list(GRID.keys()); vals = [GRID[k] for k in keys]
    total = 1
    for v in vals: total *= len(v)
    print(f"[Sweep] dataset={args.dataset} dmax={dmax}  total combinations={total}")

    rows = []
    for combo in product(*vals):
        cfg = {k: v for k, v in zip(keys, combo)}
        if args.dataset == "kitti":
            res = evaluate_once_kitti(args.root, dmax, cfg)
        else:
            res = evaluate_once_nyu(args.root, dmax, cfg)
        row = {"dataset": args.dataset, "dmax": dmax, **cfg, **res}
        rows.append(row)
        print(row)

    # CSV 저장 (A 관련 키 + 지표만)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "dataset","dmax",
        "norm_mask","flip_method","flip_thr","affine_mode","affine_log","irls_iters","huber_delta",
        "frames","mae","rmse","time_avg","elapsed"
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[CSV] saved to {args.out_csv}")

if __name__ == "__main__":
    main()
