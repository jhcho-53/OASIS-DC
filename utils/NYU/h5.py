# utils/NYU/h5.py
import os
from typing import Optional, Tuple, List

import numpy as np
import h5py
import os, json, argparse, math, random, csv
from typing import Dict, Any, Tuple, List, Optional
from .datset import _as_chw_uint8

# dataset key candidates
RGB_KEYS      = ["rgb", "image", "images", "color", "colors"]
DEPTH_KEYS    = ["gt", "depth", "depths", "gt_depth", "ground_truth"]
DL_KEYS       = ["dl", "sparse_depth", "sparse_dl", "lidar_depth"]
ML_KEYS       = ["ml", "mask", "sparse_mask", "valid_mask"]
SPARSE_X_KEYS = ["sparse_x", "x"]
SPARSE_Y_KEYS = ["sparse_y", "y"]
SPARSE_D_KEYS = ["sparse_d", "d", "depth_values"]
SPARSE_BUNDLE = ["sparse", "points"]  # (N,K,3) with (x,y,d) order
ID_KEYS       = ["ids", "id", "names", "name"]

def _h5_find_first(f: h5py.File, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in f:
            return k
    for k in keys:
        if "/" in k:
            grp, ds = k.split("/", 1)
            if grp in f and ds in f[grp]:
                return k
    return None

def _h5_get(f: h5py.File, key: str):
    if "/" in key:
        grp, ds = key.split("/", 1)
        return f[grp][ds]
    return f[key]

def _h5_count_records(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        rgb_k = _h5_find_first(f, RGB_KEYS)
        dep_k = _h5_find_first(f, DEPTH_KEYS)
        for k in [rgb_k, dep_k]:
            if k is None: 
                continue
            ds = _h5_get(f, k)
            if ds.ndim >= 3:
                return ds.shape[0]  # (N, ...)
            else:
                return 1            # 2D single sample
    raise RuntimeError(f"Cannot infer record count from {h5_path}.")

def _h5_ids_or_index(h5_path: str, N: int) -> List[str]:
    base = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        id_k = _h5_find_first(f, ID_KEYS)
        if id_k is None:
            if N == 1:
                return [base]
            return [str(i).zfill(5) for i in range(N)]
        ds = _h5_get(f, id_k)
        out = []
        for i in range(N):
            v = ds[i]
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="ignore"))
            elif hasattr(v, "astype"):
                try:
                    out.append(str(v.astype(str)))
                except Exception:
                    out.append(str(v))
            else:
                out.append(str(v))
        return out

def _h5_read_record(h5_path: str, idx: int, dmax: float,
                    sparse_scale_mm: Optional[float]) -> Tuple[np.ndarray,np.ndarray,Optional[np.ndarray],Optional[np.ndarray]]:
    """
    Returns: rgb_chw(uint8), gt_hw(float32 meters), DL_hw(float32 meters or None), ML_hw(uint8 {0,1} or None)
    Handles both single-sample(2D) and multi-sample(3D/4D) datasets.
    """
    def _take_sample_2d(ds, idx, expect_hw=None):
        arr = None
        if ds.ndim <= 2:
            arr = np.array(ds[()], dtype=np.float32)
        else:
            ok = False
            for axis in range(ds.ndim):
                slicer = [slice(None)]*ds.ndim
                slicer[axis] = idx
                a = np.array(ds[tuple(slicer)])
                a = np.squeeze(a)
                if a.ndim == 2:
                    arr = a.astype(np.float32, copy=False)
                    ok = True
                    break
            if not ok:
                a = np.array(ds[idx])
                a = np.squeeze(a)
                if a.ndim == 2:
                    arr = a.astype(np.float32, copy=False)
                elif (expect_hw is not None) and (a.ndim == 1) and (a.size == expect_hw[0]*expect_hw[1]):
                    arr = a.reshape(expect_hw).astype(np.float32, copy=False)
                else:
                    raise RuntimeError(f"Cannot extract 2D slice from dataset with shape {ds.shape}")
        return arr

    with h5py.File(h5_path, "r") as f:
        # RGB
        k_rgb = _h5_find_first(f, RGB_KEYS)
        if k_rgb is None:
            raise RuntimeError(f"[{h5_path}] RGB dataset not found.")
        ds_rgb = _h5_get(f, k_rgb)

        if ds_rgb.ndim <= 2:
            rgb_raw = np.array(ds_rgb[()])
        elif ds_rgb.ndim == 3:
            rgb_raw = np.array(ds_rgb[()]) if 3 in ds_rgb.shape else np.array(ds_rgb[idx])
        else:
            found = False
            for axis in range(ds_rgb.ndim):
                slicer = [slice(None)]*ds_rgb.ndim
                slicer[axis] = idx
                a = np.array(ds_rgb[tuple(slicer)])
                if a.ndim in (2,3):
                    rgb_raw = a; found = True; break
            if not found:
                rgb_raw = np.array(ds_rgb[idx])
        rgb = _as_chw_uint8(rgb_raw)
        H, W = rgb.shape[1], rgb.shape[2]

        # GT (meters)
        k_gt = _h5_find_first(f, DEPTH_KEYS)
        if k_gt is None:
            raise RuntimeError(f"[{h5_path}] GT depth dataset not found.")
        ds_gt = _h5_get(f, k_gt)
        gt = _take_sample_2d(ds_gt, idx, expect_hw=(H, W))
        gt = np.clip(gt, 0.0, dmax)

        # Sparse (DL/ML)
        DL_np, ML_np = None, None
        k_dl = _h5_find_first(f, DL_KEYS)
        if k_dl is not None:
            ds_dl = _h5_get(f, k_dl)
            dl = _take_sample_2d(ds_dl, idx, expect_hw=(H, W))
            if sparse_scale_mm is not None:
                dl = dl / float(sparse_scale_mm)
            dl = np.clip(dl, 0.0, dmax)
            DL_np = dl

            k_ml = _h5_find_first(f, ML_KEYS)
            if k_ml is not None:
                ds_ml = _h5_get(f, k_ml)
                ml = _take_sample_2d(ds_ml, idx, expect_hw=(H, W)).astype(np.float32)
                ML_np = (ml > 0).astype(np.uint8)
            else:
                ML_np = (dl > 0).astype(np.uint8)
        else:
            # points(x,y,d) (선택 구현 필요 시 여기 추가)
            pass

        return rgb, gt, DL_np, ML_np

def build_oneshot_from_nyu(cfg: Dict[str,Any]) -> Dict[str, List[dict]]:
    """
    - nyu.h5_train / nyu.h5_val 에서 샘플 개수 N 추정
    - K_train/K_val에 맞춰 인덱스 샘플링(oneshot은 보통 K_train=1)
    - 각 엔트리에 mono(estimation) 경로를 nyu.mono_fmt로 주입
    """
    nyu = cfg["nyu"]
    root = nyu["root"]
    # 허용: h5_* 또는 t5_*에 .h5 경로가 들어있을 때
    h5_train = nyu.get("h5_train") or nyu.get("t5_train")
    h5_val   = nyu.get("h5_val")   or nyu.get("t5_val")
    if not (h5_train and os.path.isfile(h5_train)):
        raise RuntimeError("nyu.h5_train not found or not a file.")
    if not (h5_val and os.path.isfile(h5_val)):
        raise RuntimeError("nyu.h5_val not found or not a file.")

    mono_fmt = nyu.get("mono_fmt", "")

    N_tr = _h5_count_records(h5_train)
    N_va = _h5_count_records(h5_val)
    ids_tr = _h5_ids_or_index(h5_train, N_tr)
    ids_va = _h5_ids_or_index(h5_val,   N_va)

    ks = cfg.get("kshot", {})
    K_train = int(ks.get("K_train", 1))
    K_val   = ks.get("K_val", None)
    seed    = int(ks.get("seed", 1))
    random.seed(seed)

    idx_tr_all = list(range(N_tr))
    idx_va_all = list(range(N_va))
    idx_tr = random.sample(idx_tr_all, k=min(K_train, N_tr))
    if (K_val is None) or (int(K_val) >= N_va):
        idx_va = idx_va_all
    else:
        idx_va = random.sample(idx_va_all, k=min(int(K_val), N_va))

    def _mk_entries(h5_path: str, idx_list: List[int], ids: List[str]) -> List[dict]:
        ents = []
        base = os.path.splitext(os.path.basename(h5_path))[0]
        for i in idx_list:
            rid = ids[i] if i < len(ids) else f"{base}_{i:05d}"
            mono = mono_fmt.format(root=root, id=rid) if mono_fmt else None
            ents.append({"h5": h5_path, "idx": i, "id": rid, "mono": mono})
        return ents