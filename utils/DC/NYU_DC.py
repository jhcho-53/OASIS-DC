import os
import glob
import csv
import argparse
from pathlib import Path

import h5py
import numpy as np
import cv2
from tqdm import tqdm

import scipy.sparse as sp
import scipy.sparse.linalg as spla

# =========================
# H5 로딩 유틸
# =========================
IMAGE_KEYS = ["image", "rgb", "img", "color", "images"]
DEPTH_KEYS = ["depth", "depths", "rawDepth", "rawDepths", "gt", "D"]

def read_h5_rgb_depth(h5_path: Path):
    rgb = None
    depth = None
    with h5py.File(str(h5_path), 'r') as f:
        # depth
        for k in DEPTH_KEYS:
            if k in f:
                depth = f[k][...]
                break
        if depth is None:
            for k in f.keys():
                arr = f[k][...]
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
                    depth = arr
                    break
        if depth is None:
            raise KeyError(f"No depth found in {h5_path}. Keys={list(f.keys())}")
        depth = depth.astype(np.float32)

        # rgb (optional)
        for k in IMAGE_KEYS:
            if k in f:
                img = f[k][...]
                rgb = img
                break
        if rgb is not None:
            if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            if rgb.dtype != np.uint8:
                scale = 255.0 if np.max(rgb) <= 1.0 else 1.0
                rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return rgb, depth  # depth in meters

# =========================
# 전처리: resize / center crop
# =========================
def center_crop(arr, w, h):
    H, W = arr.shape[:2]
    x0 = max(0, (W - w) // 2)
    y0 = max(0, (H - h) // 2)
    return arr[y0:y0+h, x0:x0+w]

def apply_resize_crop(rgb, depth, resize_wh=None, crop_wh=None):
    if resize_wh is not None:
        w, h = resize_wh
        if rgb is not None:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    if crop_wh is not None:
        w, h = crop_wh
        if rgb is not None:
            rgb = center_crop(rgb, w, h)
        depth = center_crop(depth, w, h)
    return rgb, depth

def cg_compat(A, b, x0=None, maxiter=1000, tol=1e-5):
    """SciPy 버전별 cg 인자 불일치(tol vs rtol/atol) 호환 래퍼."""
    import inspect
    sig = inspect.signature(spla.cg)
    params = sig.parameters
    # 신형 SciPy: rtol/atol 사용
    if "rtol" in params:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, rtol=tol, atol=0.0)
    # 구형 SciPy: tol 사용
    else:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, tol=tol)
    
# =========================
# 희소 깊이 샘플링
# =========================
def sample_sparse_depth(gt_depth, n=500, max_depth=10.0, seed=None):
    rng = np.random.default_rng(seed)
    valid = (gt_depth > 0) & np.isfinite(gt_depth)
    if max_depth is not None:
        valid &= (gt_depth <= max_depth)
    idx = np.where(valid)
    num_valid = len(idx[0])
    mask = np.zeros_like(gt_depth, dtype=bool)
    if num_valid > 0:
        k = min(n, num_valid)
        sel = rng.choice(num_valid, size=k, replace=False)
        mask[idx[0][sel], idx[1][sel]] = True
    sparse = np.zeros_like(gt_depth, dtype=np.float32)
    sparse[mask] = gt_depth[mask]
    return sparse, mask

# =========================
# 라플라시안(4-이웃) 캐시
# =========================
_LAPLACIAN_CACHE = {}
def grid_laplacian(H, W, dtype=np.float64):
    key = (H, W, dtype)
    if key in _LAPLACIAN_CACHE:
        return _LAPLACIAN_CACHE[key]
    N = H * W

    ids = np.arange(N)
    has_left  = (ids % W) != 0
    has_right = (ids % W) != (W - 1)
    has_up    = ids >= W
    has_down  = ids < (N - W)

    deg = has_left.astype(np.float64) + has_right.astype(np.float64) + \
          has_up.astype(np.float64) + has_down.astype(np.float64)

    h = np.ones(N - 1, dtype=np.float64)
    h[np.arange(N - 1) % W == (W - 1)] = 0.0  # 행 경계 단절
    off_pm1 = -h
    v = -np.ones(N - W, dtype=np.float64)

    L = sp.diags(
        diagonals=[deg, off_pm1, off_pm1, v, v],
        offsets=[0, 1, -1, W, -W],
        shape=(N, N),
        format='csr',
        dtype=dtype
    )
    _LAPLACIAN_CACHE[key] = L
    return L

# =========================
# Screened Poisson: (L + λM) z = λ M d
# =========================
def poisson_complete(sparse, mask, lam=10.0, maxiter=1000, tol=1e-5):
    H, W = sparse.shape
    if mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    L = grid_laplacian(H, W, dtype=np.float64)  # SPD
    M_diag = mask.reshape(-1).astype(np.float64)
    d = sparse.reshape(-1).astype(np.float64)

    A = L.tolil(copy=True)
    A.setdiag(A.diagonal() + lam * M_diag)
    A = A.tocsr()

    b = lam * (M_diag * d)
    x0 = d.copy()

    x, info = cg_compat(A, b, x0=x0, maxiter=maxiter, tol=tol)
    if info != 0:  # CG 실패 시 폴백
        x = spla.spsolve(A, b)
    dense = x.reshape(H, W).astype(np.float32)
    return dense

# =========================
# 지표 계산 + 누적기
# =========================
def dc_metrics(pred, gt, max_depth=10.0):
    valid = (gt > 0) & np.isfinite(gt)
    if max_depth is not None:
        valid &= (gt <= max_depth)
    if not np.any(valid):
        return dict(RMSE=np.nan, MAE=np.nan, iRMSE=np.nan, iMAE=np.nan, REL=np.nan), (0,)*6

    p = pred[valid]
    g = gt[valid]
    e = p - g

    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))

    invp = 1.0 / np.clip(p, 1e-6, None)
    invg = 1.0 / np.clip(g, 1e-6, None)
    ie   = invp - invg
    irmse = float(np.sqrt(np.mean(ie**2)))
    imae  = float(np.mean(np.abs(ie)))

    rel = float(np.mean(np.abs(e) / g))

    # 픽셀 단위 누적에 쓰일 통계(합/카운트)
    se_sum   = float(np.sum(e**2))
    ae_sum   = float(np.sum(np.abs(e)))
    sie_sum  = float(np.sum(ie**2))
    aie_sum  = float(np.sum(np.abs(ie)))
    rel_sum  = float(np.sum(np.abs(e) / g))
    n_valid  = int(g.size)

    return dict(RMSE=rmse, MAE=mae, iRMSE=irmse, iMAE=imae, REL=rel), (se_sum, ae_sum, sie_sum, aie_sum, rel_sum, n_valid)

class PixelAverager:
    def __init__(self):
        self.se = 0.0; self.ae = 0.0; self.sie = 0.0; self.aie = 0.0; self.rel = 0.0; self.n = 0
    def add(self, se_sum, ae_sum, sie_sum, aie_sum, rel_sum, n):
        self.se += se_sum; self.ae += ae_sum; self.sie += sie_sum; self.aie += aie_sum; self.rel += rel_sum; self.n += n
    def mean(self):
        if self.n == 0:
            return dict(RMSE=np.nan, MAE=np.nan, iRMSE=np.nan, iMAE=np.nan, REL=np.nan)
        return dict(
            RMSE=float(np.sqrt(self.se / self.n)),
            MAE=float(self.ae / self.n),
            iRMSE=float(np.sqrt(self.sie / self.n)),
            iMAE=float(self.aie / self.n),
            REL=float(self.rel / self.n),
        )

# =========================
# 저장 유틸
# =========================
def normalize_to_16bit(d):
    d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d_min = float(np.min(d)); d_max = float(np.max(d))
    rng = d_max - d_min
    if rng <= 0: return np.zeros_like(d, dtype=np.uint16)
    d_norm = (d - d_min) / rng
    return (d_norm * 65535.0).astype(np.uint16)

def make_viz(d):
    d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d_min = float(np.min(d)); d_max = float(np.max(d))
    rng = max(d_max - d_min, 1e-8)
    d8 = ((d - d_min) * (255.0 / rng)).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)  # BGR

# =========================
# 파일 수집
# =========================
def build_file_list(root: Path):
    patterns = [
        str(root / "train" / "**" / "*.h5"),
        str(root / "val"   / "**" / "*.h5"),
    ]
    fs = []
    for p in patterns:
        fs.extend(glob.glob(p, recursive=True))
    fs = [Path(f) for f in fs]
    fs.sort()
    return fs

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser("NYUv2 500-sample Screened Poisson Depth Completion + Metrics to CSV")
    ap.add_argument("--nyu-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)

    ap.add_argument("--sample-n", type=int, default=500)
    ap.add_argument("--lambda-reg", type=float, default=10.0)
    ap.add_argument("--max-depth", type=float, default=10.0)

    # 요청: center crop 진행. 기본값 304x228.
    ap.add_argument("--center-crop", nargs=2, type=int, default=[304, 228], metavar=("W", "H"))
    # 필요시 resize도 지원(옵션)
    ap.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"))

    # config는 유지하지만, npy 저장은 실제로 하지 않습니다.
    ap.add_argument("--save-sparse", action="store_true")
    ap.add_argument("--save-dense", action="store_true")

    ap.add_argument("--save-viz", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")

    ap.add_argument("--csv", type=str, default="metrics_poisson.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 저장 옵션 기본값: 별도 지정 없으면 16-bit + viz 저장
    # (npy는 강제로 저장하지 않음)
    if not (args.save_viz):
        args.save_viz = True  # 시각화 기본 on

    nyu_root = Path(args.nyu_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / args.csv

    # viz 루트 생성
    viz_root = out_root / "viz"
    viz_root.mkdir(parents=True, exist_ok=True)

    files = build_file_list(nyu_root)
    if not files:
        print(f("[ERROR] No .h5 found under {nyu_root}"))
        return

    # 누적기: 전체/스플릿별
    agg_all = PixelAverager()
    agg_split = {"train": PixelAverager(), "val": PixelAverager()}

    rows = []
    for h5_path in tqdm(files, desc="NYUv2 Poisson DC + Metrics"):
        try:
            rel = h5_path.relative_to(nyu_root)  # e.g., train/basement_0001a/00001.h5
            split = rel.parts[0] if len(rel.parts) > 0 else ""
            scene = rel.parts[1] if len(rel.parts) > 1 else ""
            base = (out_root / rel).with_suffix("")

            # 저장 경로들
            p16_path = base.parent / f"{base.name}_poisson{args.sample_n}_16bit.png"

            # viz는 루트의 viz/ 아래로 보냄 (상대 경로 유지)
            sub_rel = Path(*rel.parts[1:-1])  # scene 하위 경로
            viz_path = (viz_root / split / sub_rel / f"{base.name}_poisson{args.sample_n}_viz.png")

            # skip-existing: 16-bit와 viz 둘 다 있으면 완전 스킵
            if args.skip_existing and p16_path.exists() and (not args.save_viz or viz_path.exists()):
                # 스킵 시 해당 샘플은 CSV에도 포함되지 않습니다.
                continue

            # 1) 로드 및 전처리(센터 크롭 우선)
            rgb, depth = read_h5_rgb_depth(h5_path)
            rgb, depth = apply_resize_crop(
                rgb, depth,
                resize_wh=tuple(args.resize) if args.resize else None,
                crop_wh=tuple(args.center_crop) if args.center_crop else None
            )

            # 2) 희소 깊이(500개) 생성
            sparse, mask = sample_sparse_depth(depth, n=args.sample_n, max_depth=args.max_depth, seed=args.seed)

            # 3) Poisson completion
            dense = poisson_complete(sparse, mask, lam=args.lambda_reg)

            # 4) (옵션) 범위 제한
            if args.max_depth is not None:
                dense = np.clip(dense, 0.0, args.max_depth)

            # 5) 저장: 16-bit (기존 경로), viz (루트/viz/)
            p16_path.parent.mkdir(parents=True, exist_ok=True)
            d16 = normalize_to_16bit(dense)
            cv2.imwrite(str(p16_path), d16)

            if args.save_viz:
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                viz_img = make_viz(dense)
                cv2.imwrite(str(viz_path), viz_img)

            # 6) 지표 계산 및 누적
            metrics, sums = dc_metrics(dense, depth, max_depth=args.max_depth)
            agg_all.add(*sums)
            if split in agg_split: agg_split[split].add(*sums)

            rows.append([
                str(rel), split, scene, depth.shape[1], depth.shape[0],
                int(np.count_nonzero(dense > 0)),
                args.sample_n, args.lambda_reg, args.max_depth,
                metrics["RMSE"], metrics["MAE"], metrics["iRMSE"], metrics["iMAE"], metrics["REL"]
            ])

        except Exception as e:
            print(f"[ERROR] {h5_path}: {e}")

    # CSV 저장 (파일별 + 요약행)
    header = [
        "rel_path","split","scene","W","H","num_nonzero_pred",
        "N_sparse","lambda","max_depth",
        "RMSE","MAE","iRMSE","iMAE","REL"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

        # 요약: 전체, train, val
        mean_all = agg_all.mean()
        writer.writerow([])
        writer.writerow(["SUMMARY_ALL","","","","","",
                         args.sample_n,args.lambda_reg,args.max_depth,
                         mean_all["RMSE"],mean_all["MAE"],mean_all["iRMSE"],mean_all["iMAE"],mean_all["REL"]])
        for sp in ["train", "val"]:
            m = agg_split[sp].mean()
            writer.writerow([f"SUMMARY_{sp.upper()}","","","","","",
                             args.sample_n,args.lambda_reg,args.max_depth,
                             m["RMSE"],m["MAE"],m["iRMSE"],m["iMAE"],m["REL"]])

    print(f"[DONE] Processed {len(rows)} files.")
    print(f"[CSV] Saved metrics to: {csv_path}")
    print(f"[VIZ] Saved visualizations under: {viz_root}")

if __name__ == "__main__":
    main()
