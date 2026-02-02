# utils/NYU_IMG/dataset.py
import os, glob, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utils.NYU.dataset_utils import _random_pick_entries, _resolve_estimation_dir_only, _collect_samples_from_dir_fast
# -------------------- 파일 후보 --------------------
RGB_CANDIDATES = ["rgb.png", "image.png", "img.png", "color.png"]
GT_CANDIDATES_MM16 = ["depth_mm16.png", "gt_mm16.png"]  # 16-bit, mm, invalid=0

# estimation(3번: viz) 우선
EST_CANDIDATES = [
    "*_viz.png", "*_da_*_viz.png", "*_vit*_viz.png",
    "*_16bit.png", "*_mm16.png", "*.png", "*.npy", "*.tiff", "*.tif"
]

CV2_CMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet":     cv2.COLORMAP_JET,
    "turbo":   cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
    "magma":   cv2.COLORMAP_MAGMA  if hasattr(cv2, "COLORMAP_MAGMA") else cv2.COLORMAP_INFERNO,
    "plasma":  cv2.COLORMAP_PLASMA if hasattr(cv2, "COLORMAP_PLASMA") else cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
}

# -------------------- 로딩 유틸 --------------------
def _imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _read_mm16_to_meters(path: str, scale_mm: float = 1000.0) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: raise FileNotFoundError(path)
    if d.dtype != np.uint16: d = d.astype(np.uint16)
    m = d.astype(np.float32) / float(scale_mm)
    m[~np.isfinite(m)] = 0.0; m[m < 0] = 0.0
    return m

def _get_cmap_lut_bgr(cmap_name: str) -> np.ndarray:
    code = CV2_CMAPS.get(cmap_name.lower(), cv2.COLORMAP_INFERNO)
    idx = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    lut = cv2.applyColorMap(idx, code)  # (256,1,3) BGR
    return lut.reshape(256, 3).astype(np.float32)

def _invert_viz_to_uint8(viz_bgr: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    H, W = viz_bgr.shape[:2]
    lut = _get_cmap_lut_bgr(cmap_name)  # (256,3)
    pts = viz_bgr.reshape(-1, 3).astype(np.float32)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(lut)
        _, idx = tree.query(pts, k=1, workers=-1)
        return idx.astype(np.uint8).reshape(H, W)
    except Exception:
        out = np.empty((H * W,), dtype=np.uint8)
        step = 200_000
        for s in range(0, pts.shape[0], step):
            p = pts[s:s+step]
            dif = p[:, None, :] - lut[None, :, :]
            dsq = np.sum(dif * dif, axis=2)
            out[s:s+step] = np.argmin(dsq, axis=1).astype(np.uint8)
        return out.reshape(H, W)

def _read_viz_to_meters(path: str, dmax: float, cmap_name: str = "inferno") -> np.ndarray:
    viz = cv2.imread(path, cv2.IMREAD_COLOR)
    if viz is None: raise FileNotFoundError(path)
    idx8 = _invert_viz_to_uint8(viz, cmap_name=cmap_name)
    return (idx8.astype(np.float32) / 255.0) * float(dmax)

# -------------------- 전처리 --------------------
def _center_crop_array(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = arr.shape[:2]; y0 = max(0, (H - h)//2); x0 = max(0, (W - w)//2)
    return arr[y0:y0+h, x0:x0+w, ...] if arr.ndim == 3 else arr[y0:y0+h, x0:x0+w]

# -------------------- 희소 샘플링(500) --------------------
def _make_sparse_from_gt(gt: torch.Tensor, n: int = 500, seed: Optional[int] = None):
    assert gt.dim() == 3 and gt.shape[0] == 1
    H, W = gt.shape[-2:]
    g = gt[0].cpu().numpy()
    valid = np.where((g > 0) & np.isfinite(g))
    DL = torch.zeros_like(gt); ML = torch.zeros_like(gt)
    if valid[0].size == 0: return DL, ML
    rng = np.random.default_rng(seed)
    k = min(n, valid[0].size)
    sel = rng.choice(valid[0].size, size=k, replace=False)
    ys, xs = valid[0][sel], valid[1][sel]
    dl = np.zeros((H, W), np.float32); dl[ys, xs] = g[ys, xs]
    ml = np.zeros((H, W), np.float32); ml[ys, xs] = 1.0
    DL[0] = torch.from_numpy(dl); ML[0] = torch.from_numpy(ml)
    return DL, ML

def _make_sparse_from_gt_deterministic(gt: torch.Tensor, n: int, key: Tuple):
    seed = abs(hash(key)) % (2**32)
    return _make_sparse_from_gt(gt, n=n, seed=seed)

# -------------------- Screened Poisson --------------------
_LAPLACIAN_CACHE: Dict[Tuple[int,int], sp.csr_matrix] = {}
def _grid_laplacian(H: int, W: int) -> sp.csr_matrix:
    key = (H, W)
    if key in _LAPLACIAN_CACHE: return _LAPLACIAN_CACHE[key]
    N = H * W; ids = np.arange(N)
    has_left  = (ids % W) != 0; has_right = (ids % W) != (W - 1)
    has_up    = ids >= W;       has_down  = ids < (N - W)
    deg = has_left.astype(np.float64) + has_right.astype(np.float64) + has_up.astype(np.float64) + has_down.astype(np.float64)
    h = np.ones(N - 1, dtype=np.float64); h[np.arange(N - 1) % W == (W - 1)] = 0.0
    v = -np.ones(N - W, dtype=np.float64)
    L = sp.diags([deg, -h, -h, v, v], [0, 1, -1, W, -W], shape=(N, N), format="csr", dtype=np.float64)
    _LAPLACIAN_CACHE[key] = L; return L

def _cg_compat(A, b, x0=None, maxiter=1000, tol=1e-5):
    import inspect
    sig = inspect.signature(spla.cg)
    if "rtol" in sig.parameters:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, rtol=tol, atol=0.0)
    else:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, tol=tol)

def _poisson_complete_single(E: np.ndarray, DL: np.ndarray, ML: np.ndarray,
                             lam: float, iters: int, hard: bool, dmax: float) -> np.ndarray:
    H, W = E.shape; L = _grid_laplacian(H, W)
    e = E.astype(np.float64).reshape(-1); d = DL.astype(np.float64).reshape(-1); m = ML.astype(np.float64).reshape(-1)
    if hard:
        A = L.tolil(copy=True); b = L.dot(e)
        idx = np.where(m > 0.5)[0]
        for i in idx:
            A.rows[i] = [i]; A.data[i] = [1.0]; b[i] = d[i]
        A = A.tocsr()
    else:
        A = L.tolil(copy=True); A.setdiag(A.diagonal() + lam * m); A = A.tocsr()
        b = L.dot(e) + lam * (m * d)
    x0 = e.copy()
    x, info = _cg_compat(A, b, x0=x0, maxiter=int(iters), tol=1e-5)
    if info != 0: x = spla.spsolve(A, b)
    z = x.reshape(H, W).astype(np.float32)
    if dmax and dmax > 0: z = np.clip(z, 0.0, float(dmax))
    z[~np.isfinite(z)] = 0.0
    return z

def poisson_complete(E: torch.Tensor, DL: torch.Tensor, ML: torch.Tensor,
                     lam: float, iters: int, hard: bool, dmax: float) -> torch.Tensor:
    assert E.dim() == 4 and E.shape == DL.shape == ML.shape
    B, _, H, W = E.shape; outs = []
    for b in range(B):
        z = _poisson_complete_single(E[b,0].cpu().numpy(), DL[b,0].cpu().numpy(), ML[b,0].cpu().numpy(),
                                     lam=lam, iters=iters, hard=hard, dmax=dmax)
        outs.append(torch.from_numpy(z).unsqueeze(0))
    return torch.stack(outs, dim=0)

# -------------------- Split Builder --------------------
def _collect_samples_from_dir(split_dir: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not os.path.isdir(split_dir): return items
    for root, _, _ in os.walk(split_dir):
        gt = None
        for c in GT_CANDIDATES_MM16:
            p = os.path.join(root, c)
            if os.path.isfile(p): gt = p; break
        if gt is None: continue
        rgb = None
        for c in RGB_CANDIDATES:
            p = os.path.join(root, c)
            if os.path.isfile(p): rgb = p; break
        rel = os.path.relpath(root, split_dir).replace("\\", "/")
        items.append({"img": rgb, "gt": gt, "id": rel})
    return sorted(items, key=lambda x: x["id"])

def _resolve_estimation(nyu: Dict[str, Any], split: str, item: Dict[str, str]) -> Optional[str]:
    mono_root = nyu.get("mono_root", None)
    if not mono_root: return None
    cand_dirs = [
        os.path.join(mono_root, "viz", split, item["id"]),
        os.path.join(mono_root, split, item["id"]),
        os.path.join(mono_root, split, os.path.dirname(item["id"]) or ""),
    ]
    cands = []
    for base in cand_dirs:
        for pat in EST_CANDIDATES:
            cands += glob.glob(os.path.join(base, pat))
    if not cands:
        for pat in EST_CANDIDATES:
            cands += glob.glob(os.path.join(mono_root, "**", item["id"] + pat), recursive=True)
    cands = [c for c in cands if os.path.isfile(c)]
    return sorted(cands)[0] if cands else None

def build_oneshot_from_nyu(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    random_pick=True 이면 전수 스캔 없이 무작위로 K개만 골라 entries를 만든다.
    random_pick=False 이면 기존의 스캔 방식(혹은 fast scan)을 사용.
    """
    nyu = cfg["nyu"]
    ks  = cfg.get("kshot", {})
    K_train = int(ks.get("K_train", 1))
    K_val   = ks.get("K_val", None)
    seed    = int(ks.get("seed", 1))

    random_pick = bool(nyu.get("random_pick", False))
    tries       = int(nyu.get("random_tries", 200))
    require_rgb = bool(nyu.get("require_rgb", False))

    if random_pick:
        # 무작위로 K_train, K_val만 뽑기 (전체 스캔 없음)
        train_sel = _random_pick_entries(nyu, "train", K_train, seed=seed,
                                         tries_per_pick=tries, require_rgb=require_rgb)
        if not train_sel:
            raise RuntimeError("Random pick failed for train split. Increase random_tries or check folder structure.")
        if (K_val is None):
            # 검증 전체 스캔은 피하려면, 합리적인 개수로 뽑아 쓰세요. 기본은 1장.
            K_val_eff = 1
        else:
            K_val_eff = int(K_val)
        val_sel = _random_pick_entries(nyu, "val", K_val_eff, seed=seed+1,
                                       tries_per_pick=tries, require_rgb=require_rgb)
        if not val_sel:
            raise RuntimeError("Random pick failed for val split. Increase random_tries or check folder structure.")
        return {"train": train_sel, "val": val_sel}

    # ---- random_pick=False: 기존 방식(전체/빠른 스캔)으로 수집 ----
    # 당신이 이전에 넣어둔 fast scan/early-stop/caching 버전을 호출하세요.
    # 예: 기존 함수명이 _collect_samples_from_dir_fast 라면 아래처럼.
    train_items = _collect_samples_from_dir_fast(nyu.get("train_dir", os.path.join(nyu["root"], "train")))
    val_items   = _collect_samples_from_dir_fast(nyu.get("val_dir",   os.path.join(nyu["root"], "val")))
    if not train_items: raise RuntimeError("No samples found in train.")
    if not val_items:   raise RuntimeError("No samples found in val.")

    # K_train, K_val만 샘플링
    random.seed(seed)
    train_sel = random.sample(train_items, k=K_train) if len(train_items) > K_train else train_items
    if (K_val is None) or (int(K_val) >= len(val_items)):
        val_sel = val_items
    else:
        val_sel = random.sample(val_items, k=int(K_val))

    # estimation 주입
    for it in train_sel: it["mono"] = _resolve_estimation_dir_only(nyu, "train", it["id"])
    for it in val_sel:   it["mono"] = _resolve_estimation_dir_only(nyu, "val",   it["id"])
    return {"train": train_sel, "val": val_sel}

# -------------------- GT 덤프 --------------------
def save_depth_png16_with_scale(path: str, depth_m: np.ndarray, scale_mm: float = 1000.0):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0.0; d[d < 0] = 0.0
    mm = np.clip(np.round(d * scale_mm), 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mm)

def save_jet(path: str, depth_m: np.ndarray, dmax: float):
    d = depth_m.astype(np.float32).copy()
    if dmax and dmax > 0: d = np.clip(d, 0.0, float(dmax))
    valid = (d > 0) & np.isfinite(d)
    if np.any(valid): vmin, vmax = float(d[valid].min()), float(d[valid].max())
    else: vmin, vmax = 0.0, 1.0
    vmax = vmin + 1.0 if (vmax - vmin) < 1e-6 else vmax
    d8 = ((np.clip(d, vmin, vmax) - vmin) * (255.0 / (vmax - vmin))).astype(np.uint8)
    jet = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, jet)

def dump_oneshot_gt_images(splits: Dict[str, List[dict]], cfg: Dict[str, Any], dmax: float):
    ks = cfg.get("kshot", {}); nyu = cfg["nyu"]
    save_dir = ks.get("save_dir", "runs_nyu_1shot")
    out_dir  = os.path.join(save_dir, "oneshot_dump"); os.makedirs(out_dir, exist_ok=True)
    scale_mm = float(cfg.get("dump", {}).get("scale_mm", 1000.0))
    ch = int(nyu.get("crop_h", 228)); cw = int(nyu.get("crop_w", 304))
    do_crop = (ch is not None and cw is not None)

    for ent in splits["train"]:
        rid = ent.get("id", os.path.basename(os.path.dirname(ent["gt"])))
        gt_m = _read_mm16_to_meters(ent["gt"], scale_mm=scale_mm)
        if do_crop: gt_m = _center_crop_array(gt_m, ch, cw)
        save_depth_png16_with_scale(os.path.join(out_dir, f"{rid}_gt16.png"), gt_m, scale_mm=scale_mm)
        save_jet(os.path.join(out_dir, f"{rid}_gt_jet.png"), gt_m, dmax=dmax)

# -------------------- Dataset --------------------
class NYUH5OneShotDataset(Dataset):
    """
    entries: [{"img": <rgb or None>, "gt": <mm16>, "id": rid, "mono": <viz or None>}, ...]
    반환: I(uint8 3xHxW), DL(1xHxW), ML(1xHxW), P(1xHxW), E_norm(1xHxW), GT(1xHxW)
    """
    def __init__(self, entries: List[dict], dmax: float,
                 mono_scale: Optional[float],
                 sparse_scale: Optional[float],
                 fix_sparse: Optional[bool],
                 poisson_lam: float, poisson_iters: int, poisson_hard: bool,
                 crop_hw: Optional[Tuple[int,int]] = (228, 304),
                 est_cmap: str = "inferno"):
        super().__init__()
        self.entries = entries
        self.dmax = float(dmax)
        self.mono_scale = mono_scale  # (viz 사용 시 미사용)
        self.sparse_scale = sparse_scale
        self.fix_sparse = bool(fix_sparse) if fix_sparse is not None else False
        self.p_lam = float(poisson_lam); self.p_iters = int(poisson_iters); self.p_hard = bool(poisson_hard)
        self.crop_hw = crop_hw; self.est_cmap = est_cmap
        self._sparse_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self): return len(self.entries)

    def __getitem__(self, i: int):
        ent = self.entries[i]
        # GT & RGB
        GT_np = _read_mm16_to_meters(ent["gt"], scale_mm=1000.0)
        H, W = GT_np.shape
        if ent.get("img") and os.path.isfile(ent["img"]):
            rgb = _imread_rgb(ent["img"])
        else:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # crop
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            rgb = _center_crop_array(rgb, ch, cw)
            GT_np = _center_crop_array(GT_np, ch, cw)

        I  = torch.from_numpy(rgb.transpose(2,0,1).copy())          # (3,H,W) uint8
        GT = torch.from_numpy(GT_np).unsqueeze(0).float()           # (1,H,W)

        # Sparse
        key = (ent["gt"], *(self.crop_hw if self.crop_hw else (-1,-1)))
        if self.fix_sparse:
            if key in self._sparse_cache: DL, ML = self._sparse_cache[key]
            else:
                DL, ML = _make_sparse_from_gt_deterministic(GT, 500, key)
                self._sparse_cache[key] = (DL.clone(), ML.clone())
        else:
            DL, ML = _make_sparse_from_gt(GT, 500)
        if self.sparse_scale is not None: DL = DL * float(self.sparse_scale)

        # Estimation from viz (3번)
        E = torch.zeros_like(GT)
        if ent.get("mono") and os.path.isfile(ent["mono"]):
            viz = cv2.imread(ent["mono"], cv2.IMREAD_COLOR)
            if viz is not None:
                if self.crop_hw is not None: viz = _center_crop_array(viz, ch, cw)
                idx8 = _invert_viz_to_uint8(viz, cmap_name=self.est_cmap)
                e_np = (idx8.astype(np.float32) / 255.0) * self.dmax
                E = torch.from_numpy(e_np).unsqueeze(0).float()

        # Poisson completion
        P = poisson_complete(E.unsqueeze(0), DL.unsqueeze(0), ML.unsqueeze(0),
                             lam=self.p_lam, iters=self.p_iters, hard=self.p_hard, dmax=self.dmax)[0]

        E_norm = torch.clamp(E / self.dmax, 0.0, 1.0)
        return I, DL, ML, P, E_norm, GT
