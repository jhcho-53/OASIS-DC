# eval_poisson_from_triplet.py
# List format (space-separated, one sample per line):
#   <est_path> <sparse_path> <gt_path>
#
# Example:
#   /data/exp/est/000001.png  /data/kitti/velodyne_raw/000001.png  /data/kitti/groundtruth_depth/000001.png
#
# Usage:
#   # KITTI (dmax=80m, scale=256)
#   python eval_poisson_from_triplet.py \
#     --dataset kitti --list kitti_triplet.txt --device cuda:0 \
#     --poisson-init est --clip-to-max-gt --save-dir out/pseudo --save-format png16
#
#   # NYUv2 (dmax=10m, scale=1000)
#   python eval_poisson_depth_est.py \
#     --dataset nyuv2 --list nyu_triplet.txt --device cuda:0 \
#     --poisson-init est --save-dir out/pseudo_nyu --save-format npy

import os, argparse, numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# ===== Import your Poisson solver =====
try:
    # must provide: poisson_gpu(sparse_m, est_m, tol, maxiter, device, init, clip_to_max_gt)
    from poisson import poisson_gpu
except Exception as e:
    raise ImportError(
        "Cannot import 'poisson_gpu'. Ensure your environment exposes "
        "'from poisson import poisson_gpu' with the required signature."
    ) from e


# ===== Poisson wrapper (based on your provided function) =====
class PoissonWrapper:
    def __init__(self, dmax, tol=1e-4, maxiter=600, init="zeros", clip_to_max_gt=True,
                 auto_flip=True, est_affine=True, smooth_est=False, device="cuda:0"):
        self.cfg = type("cfg", (), {"dmax": float(dmax)})
        self.poisson_tol = float(tol)
        self.poisson_maxiter = int(maxiter)
        self.poisson_init = init
        self.poisson_clip_to_max_gt = bool(clip_to_max_gt)
        self.poisson_auto_flip = bool(auto_flip)
        self.poisson_est_affine = bool(est_affine)
        self.poisson_smooth_est = bool(smooth_est)
        self.device = device

    @staticmethod
    def _smooth3_reflect_np(e2d: np.ndarray) -> np.ndarray:
        x = np.pad(e2d, 1, mode='reflect')
        out = (x[:-2, :-2] + x[:-2, 1:-1] + x[:-2, 2:] +
               x[1:-1, :-2] + x[1:-1, 1:-1] + x[1:-1, 2:] +
               x[2:, :-2] + x[2:, 1:-1] + x[2:, 2:]) / 9.0
        return out.astype(np.float32)

    def _poisson_batch(self, DL: torch.Tensor, ML: torch.Tensor, E01: torch.Tensor):
        import numpy as _np
        B = DL.shape[0]; dmax = float(self.cfg.dmax); dev = DL.device
        dev_str = self.device
        P_list, stats_list = [], []
        for b in range(B):
            e  = E01[b,0].detach().cpu().numpy().astype(np.float32)          # [0,1]
            dl = (DL[b,0] / dmax).detach().cpu().numpy().astype(np.float32)  # ~[0,1]
            m  = (ML[b,0].detach().cpu().numpy().astype(np.float32) > 0)

            if self.poisson_auto_flip and m.sum() >= 10:
                em = e[m].reshape(-1); dm = dl[m].reshape(-1)
                if em.size > 1 and dm.size > 1:
                    corr = _np.corrcoef(em, dm)[0,1]
                    if not _np.isfinite(corr): corr = 0.0
                    if corr < 0.0: e = 1.0 - e

            if self.poisson_est_affine and m.sum() >= 10:
                x = e[m].reshape(-1,1); y = dl[m].reshape(-1,1)
                A = _np.concatenate([x, _np.ones_like(x)], axis=1)
                w = _np.ones((A.shape[0],1), dtype=_np.float32)
                for _ in range(3):
                    Aw = A * w; yw = y * w
                    theta, *_ = _np.linalg.lstsq(Aw, yw, rcond=None)
                    r = (A @ theta - y)
                    c = 1.345 * _np.median(_np.abs(r)) + 1e-6
                    w = (1.0 / _np.maximum(1.0, _np.abs(r)/c)).astype(_np.float32)
                a, b0 = float(theta[0,0]), float(theta[1,0])
                e = _np.clip(a*e + b0, 0.0, 1.0)

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


# ===== IO helpers =====
def _read_png_16bit_as_meters(path: str, scale_div: float) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # depth png일 가능성이 낮지만 안전하게 첫 채널만 사용
        arr = arr[..., 0]
    if arr.dtype not in (np.uint16, np.uint32, np.int32):
        arr = arr.astype(np.uint16)
    depth = arr.astype(np.float32) / float(scale_div)
    depth[arr == 0] = 0.0
    return depth

def _read_est_e01_from_8bit(path: str) -> np.ndarray:
    # 0~255 상대 깊이 이미지를 [0,1]로
    img = Image.open(path).convert("L")  # 강제 그레이스케일
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def _ensure_same_size(a: np.ndarray, b: np.ndarray, name_a="A", name_b="B"):
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {name_a}{a.shape} != {name_b}{b.shape}")


# ===== Dataset =====
class TripletList(Dataset):
    """
    Each line: <est_path> <sparse_path> <gt_path>
    est: 8-bit PNG (0~255), converted to E01 in [0,1]
    sparse, gt: 16-bit PNG depth images (meters scaled by divisor)
    """
    def __init__(self, list_file: str, dataset: str, sparse_div: float, gt_div: float, resize_est: bool):
        self.items = []
        with open(list_file, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                parts = ln.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid line (need 3 columns): {ln}")
                self.items.append({"est": parts[0], "sparse": parts[1], "gt": parts[2]})
        self.dataset = dataset
        self.sparse_div = float(sparse_div)
        self.gt_div = float(gt_div)
        self.resize_est = bool(resize_est)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # read gt first (reference size)
        gt = _read_png_16bit_as_meters(it["gt"], self.gt_div)
        H, W = gt.shape

        # read sparse
        sparse = _read_png_16bit_as_meters(it["sparse"], self.sparse_div)
        _ensure_same_size(sparse, gt, "sparse", "gt")

        # read est (0~255 image -> E01)
        e01 = _read_est_e01_from_8bit(it["est"])
        if e01.shape != (H, W):
            if not self.resize_est:
                raise RuntimeError(f"est shape {e01.shape} != gt shape {(H,W)}; use --resize-est.")
            # bilinear resize (keep dynamic range)
            pil = Image.fromarray((e01 * 255.0).astype(np.uint8))
            pil = pil.resize((W, H), resample=Image.BILINEAR)
            e01 = np.asarray(pil).astype(np.float32) / 255.0
            e01 = np.clip(e01, 0.0, 1.0).astype(np.float32)

        # masks
        m_sparse = (sparse > 0.0).astype(np.bool_)
        m_gt = (gt > 0.0).astype(np.bool_)

        sample = {
            "DL": sparse[None, ...].astype(np.float32),  # meters
            "ML": m_sparse[None, ...],
            "GT": gt[None, ...].astype(np.float32),      # meters
            "E01": e01[None, ...].astype(np.float32),    # [0,1]
            "valid_gt": m_gt[None, ...],
            "id": os.path.splitext(os.path.basename(it["gt"]))[0]
        }
        return sample


# ===== Metrics =====
def rmse_mae(pred_m: np.ndarray, gt_m: np.ndarray) -> tuple[float, float]:
    valid = gt_m > 0.0
    if valid.sum() == 0:
        return float("nan"), float("nan")
    diff = pred_m[valid] - gt_m[valid]
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae  = float(np.mean(np.abs(diff)))
    return rmse, mae


# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="triplet list: <est> <sparse> <gt>")
    ap.add_argument("--dataset", required=True, choices=["kitti", "nyuv2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    # scales (PNG -> meters). Default by dataset.
    ap.add_argument("--sparse-div", type=float, default=None, help="PNG scale divisor for sparse depth")
    ap.add_argument("--gt-div", type=float, default=None, help="PNG scale divisor for GT depth")
    ap.add_argument("--dmax", type=float, default=None, help="max depth in meters for E01->meters (KITTI=80, NYU=10)")
    ap.add_argument("--resize-est", action="store_true", help="resize est to GT size (bilinear)")
    # Poisson options
    ap.add_argument("--poisson-tol", type=float, default=1e-4)
    ap.add_argument("--poisson-maxiter", type=int, default=600)
    ap.add_argument("--poisson-init", type=str, default="zeros", choices=["zeros","est"])
    ap.add_argument("--clip-to-max-gt", action="store_true")
    ap.add_argument("--no-auto-flip", action="store_true")
    ap.add_argument("--no-affine", action="store_true")
    ap.add_argument("--smooth-est", action="store_true")
    # Saving
    ap.add_argument("--save-dir", default=None, help="optional dir to save pseudo depth")
    ap.add_argument("--save-format", default="png16", choices=["png16","npy"])
    ap.add_argument("--save-scale", type=float, default=None, help="for png16, meters->uint16 scale (KITTI=256, NYU=1000)")
    args = ap.parse_args()

    # defaults by dataset
    if args.dataset == "kitti":
        if args.dmax is None: args.dmax = 80.0
        if args.sparse_div is None: args.sparse_div = 256.0
        if args.gt_div is None: args.gt_div = 256.0
        if args.save_scale is None: args.save_scale = 256.0
    else:  # nyuv2
        if args.dmax is None: args.dmax = 10.0
        if args.sparse_div is None: args.sparse_div = 1000.0
        if args.gt_div is None: args.gt_div = 1000.0
        if args.save_scale is None: args.save_scale = 1000.0

    # data
    ds = TripletList(args.list, dataset=args.dataset,
                     sparse_div=args.sparse_div, gt_div=args.gt_div,
                     resize_est=args.resize_est)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, collate_fn=lambda x: x)

    device = torch.device(args.device if (torch.cuda.is_available() or "cuda" in args.device) else "cpu")
    PW = PoissonWrapper(
        dmax=args.dmax, tol=args.poisson_tol, maxiter=args.poisson_maxiter,
        init=args.poisson_init, clip_to_max_gt=args.clip_to_max_gt,
        auto_flip=(not args.no_auto_flip), est_affine=(not args.no_affine),
        smooth_est=args.smooth_est, device=args.device
    )

    # accumulators
    rmse_sum, mae_sum, n_count = 0.0, 0.0, 0

    # prepare saver
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        assert args.save_format in ("png16","npy")

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluating ({args.dataset})"):
            # stack
            DL  = torch.stack([torch.from_numpy(b["DL"])  for b in batch], dim=0).to(device=device, dtype=torch.float32)
            ML  = torch.stack([torch.from_numpy(b["ML"])  for b in batch], dim=0).to(device=device)
            GT  = torch.stack([torch.from_numpy(b["GT"])  for b in batch], dim=0).to(device=device, dtype=torch.float32)
            E01 = torch.stack([torch.from_numpy(b["E01"]) for b in batch], dim=0).to(device=device, dtype=torch.float32)
            ids = [b["id"] for b in batch]

            # Poisson
            P, _ = PW._poisson_batch(DL=DL, ML=ML, E01=E01)  # meters

            # metrics
            for i in range(P.shape[0]):
                pred = P[i,0].detach().cpu().numpy()
                gt   = GT[i,0].detach().cpu().numpy()
                rmse, mae = rmse_mae(pred, gt)
                if not np.isnan(rmse):
                    rmse_sum += rmse; mae_sum += mae; n_count += 1

                # save
                if save_dir:
                    out_id = ids[i]
                    if args.save_format == "npy":
                        np.save(os.path.join(save_dir, f"{out_id}.npy"), pred.astype(np.float32))
                    else:
                        # png16 with provided scale
                        arr = (np.clip(pred, 0, args.dmax) * args.save_scale + 0.5).astype(np.uint16)
                        Image.fromarray(arr).save(os.path.join(save_dir, f"{out_id}.png"))

    if n_count == 0:
        print("No valid samples evaluated.")
        return

    print("=== Evaluation (valid GT pixels only) ===")
    print(f"RMSE [m]: {rmse_sum / n_count:.4f}")
    print(f"MAE  [m]: {mae_sum  / n_count:.4f}")


if __name__ == "__main__":
    main()
