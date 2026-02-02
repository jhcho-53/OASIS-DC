import os, json, argparse, math, random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import h5py
import cv2
import os, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

# ==================== HDF5 loader ====================
# 후보 키들
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


def _collect_samples_from_dir_fast(
    split_dir: str,
    stop_after: Optional[int] = None,
    require_rgb: bool = False,
    gt_names: Optional[List[str]] = None,
    rgb_names: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    depth_mm16.png(또는 지정된 GT 이름들)만 재귀 글롭하여 샘플 디렉터리를 빠르게 수집합니다.
    - 전수 os.walk 대비 훨씬 적은 I/O 로드
    - stop_after 를 지정하면 해당 개수만 모은 뒤 즉시 종료(얼리-스톱)
    - require_rgb=True 이면 RGB가 없는 샘플은 제외
    - 반환: [{"img": <rgb or None>, "gt": <gt_path>, "id": <split_dir 기준 상대경로>}, ...]

    Args:
        split_dir:   train 또는 val 디렉터리의 루트 경로
        stop_after:  이 개수만 수집하면 중단 (None이면 전부 수집)
        require_rgb: True면 rgb 후보 파일이 없는 샘플은 제외
        gt_names:    GT 파일명 후보 리스트 (기본: ["depth_mm16.png","gt_mm16.png"])
        rgb_names:   RGB 파일명 후보 리스트 (기본: ["rgb.png","image.png","img.png","color.png"])

    Note:
        - 상위 모듈에서 GT_CANDIDATES_MM16 / RGB_CANDIDATES 를 이미 정의했다면 그대로 재사용합니다.
        - 중복 디렉터리는 제거합니다(동일 샘플 폴더에 GT 후보가 여러 개 있어도 1회만 추가).
    """
    if not os.path.isdir(split_dir):
        return []

    # 상위에서 정의된 상수 재사용 (없으면 기본값)
    if gt_names is None:
        gt_names = globals().get("GT_CANDIDATES_MM16", ["depth_mm16.png", "gt_mm16.png"])
    if rgb_names is None:
        rgb_names = globals().get("RGB_CANDIDATES", ["rgb.png", "image.png", "img.png", "color.png"])

    # 패턴 만들기: <split_dir>/**/<gt_name>
    patterns = [os.path.join(split_dir, "**", name) for name in gt_names]

    items: List[Dict[str, str]] = []
    seen_dirs = set()

    for pat in patterns:
        # iglob은 제너레이터이므로 메모리 사용이 작고 빠릅니다.
        for gt_path in glob.iglob(pat, recursive=True):
            # 파일 유효성
            if not os.path.isfile(gt_path):
                continue

            sample_dir = os.path.dirname(gt_path)
            if sample_dir in seen_dirs:
                continue  # 같은 디렉터리에서 다른 GT 후보가 발견되어도 중복 추가 금지

            # RGB 후보 중 첫 번째를 채택 (없으면 None)
            rgb_path = None
            for rn in rgb_names:
                p = os.path.join(sample_dir, rn)
                if os.path.isfile(p):
                    rgb_path = p
                    break

            if require_rgb and (rgb_path is None):
                continue

            rel = os.path.relpath(sample_dir, split_dir).replace("\\", "/")
            items.append({"img": rgb_path, "gt": gt_path, "id": rel})
            seen_dirs.add(sample_dir)

            # 목표 개수 수집했으면 즉시 종료
            if stop_after is not None and len(items) >= int(stop_after):
                # 정렬하여 반환(일관된 순서 보장)
                items.sort(key=lambda x: x["id"])
                return items

    # 전체 순회 종료 후 정렬
    items.sort(key=lambda x: x["id"])
    return items

def _as_chw_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Accepts (H,W,3) or (3,H,W) or (H,W) or (H,W,1) or (1,H,W)
    Returns (3,H,W) uint8
    """
    if arr.ndim == 2:  # (H,W) gray
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3 and arr.shape[-1] == 3:  # (H,W,3)
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] == 3:   # (3,H,W)
        pass
    elif arr.ndim == 3 and arr.shape[-1] == 1:  # (H,W,1)
        g = arr[..., 0]
        arr = np.stack([g, g, g], axis=0)
    elif arr.ndim == 3 and arr.shape[0] == 1:   # (1,H,W)
        g = arr[0]
        arr = np.stack([g, g, g], axis=0)
    else:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a_max = float(a.max()) if a.size else 1.0
        if a_max <= 1.0 + 1e-6:
            a = np.clip(a * 255.0, 0, 255)
        else:
            a = np.clip(a, 0, 255)
        arr = a.astype(np.uint8)
    return arr

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
                return 1            # 2D 단일 샘플
    raise RuntimeError(f"Cannot infer record count from {h5_path}. Expected one of {RGB_KEYS} or {DEPTH_KEYS}.")

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
    단일 샘플(2D)과 다중 샘플(3D/4D) 모두 처리.
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
        # ---------- RGB ----------
        k_rgb = _h5_find_first(f, RGB_KEYS)
        if k_rgb is None:
            raise RuntimeError(f"[{h5_path}] RGB dataset not found. Tried {RGB_KEYS}")
        ds_rgb = _h5_get(f, k_rgb)

        if ds_rgb.ndim <= 2:
            rgb_raw = np.array(ds_rgb[()])
        elif ds_rgb.ndim == 3:
            if 3 in ds_rgb.shape:
                rgb_raw = np.array(ds_rgb[()])
            else:
                rgb_raw = np.array(ds_rgb[idx])
        else:
            found = False
            for axis in range(ds_rgb.ndim):
                slicer = [slice(None)]*ds_rgb.ndim
                slicer[axis] = idx
                a = np.array(ds_rgb[tuple(slicer)])
                if a.ndim in (2,3):
                    rgb_raw = a
                    found = True
                    break
            if not found:
                rgb_raw = np.array(ds_rgb[idx])
        rgb = _as_chw_uint8(rgb_raw)          # (3,H,W)
        H, W = rgb.shape[1], rgb.shape[2]

        # ---------- GT (meters assumed in NYUv2 H5) ----------
        k_gt = _h5_find_first(f, DEPTH_KEYS)
        if k_gt is None:
            raise RuntimeError(f"[{h5_path}] GT depth dataset not found. Tried {DEPTH_KEYS}")
        ds_gt = _h5_get(f, k_gt)
        gt = _take_sample_2d(ds_gt, idx, expect_hw=(H, W))
        gt = np.clip(gt, 0.0, dmax)

        # ---------- Sparse (DL/ML or points) ----------
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
            # points(x,y,d) 형태
            k_bundle = _h5_find_first(f, SPARSE_BUNDLE)
            if k_bundle is not None:
                pts = _h5_get(f, k_bundle)[idx]  # (K,3) or empty
                if pts.ndim == 1 and pts.size == 0:
                    DL_np = np.zeros((H,W), dtype=np.float32)
                    ML_np = np.zeros((H,W), dtype=np.uint8)
                else:
                    x = pts[:,0].astype(np.int32)
                    y = pts[:,1].astype(np.int32)
                    d = pts[:,2].astype(np.float32)
                    if sparse_scale_mm is not None:
                        d = d / float(sparse_scale_mm)
                    x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1)
                    d = np.clip(d, 0.0, dmax)
                    DL_np = np.zeros((H,W), dtype=np.float32)
                    ML_np = np.zeros((H,W), dtype=np.uint8)
                    DL_np[y, x] = d
                    ML_np[y, x] = 1
            else:
                kx = _h5_find_first(f, SPARSE_X_KEYS)
                ky = _h5_find_first(f, SPARSE_Y_KEYS)
                kd = _h5_find_first(f, SPARSE_D_KEYS)
                if (kx is not None) and (ky is not None) and (kd is not None):
                    x = np.array(_h5_get(f, kx)[idx]).astype(np.int32).ravel()
                    y = np.array(_h5_get(f, ky)[idx]).astype(np.int32).ravel()
                    d = np.array(_h5_get(f, kd)[idx]).astype(np.float32).ravel()
                    if sparse_scale_mm is not None:
                        d = d / float(sparse_scale_mm)
                    x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1)
                    d = np.clip(d, 0.0, dmax)
                    DL_np = np.zeros((H,W), dtype=np.float32)
                    ML_np = np.zeros((H,W), dtype=np.uint8)
                    DL_np[y, x] = d
                    ML_np[y, x] = 1

        return rgb, gt, DL_np, ML_np
    
# ==================== Center crop helpers ====================
def _center_crop_slices(H: int, W: int, out_h: int, out_w: int):
    if out_h > H or out_w > W:
        raise RuntimeError(f"Crop size ({out_w}x{out_h}) exceeds input ({W}x{H}).")
    top  = (H - out_h) // 2
    left = (W - out_w) // 2
    return slice(top, top + out_h), slice(left, left + out_w)

def _center_crop_tensor(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    x: (C,H,W) or (1,H,W) or (H,W)
    """
    if x.dim() == 2:
        H, W = x.shape
        ys, xs = _center_crop_slices(H, W, out_h, out_w)
        return x[ys, xs]
    elif x.dim() == 3:
        C, H, W = x.shape
        ys, xs = _center_crop_slices(H, W, out_h, out_w)
        return x[:, ys, xs]
    else:
        raise RuntimeError(f"Expected 2D/3D tensor, got {tuple(x.shape)}")
    
# ==================== Image/Depth I/O helpers ====================
def _read_rgb_path(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # H,W,3
    return torch.from_numpy(arr).permute(2,0,1).contiguous()  # 3,H,W

def _read_u16(path: str) -> Tuple[np.ndarray, dict]:
    img = Image.open(path)
    text = dict(img.text) if hasattr(img, "text") else {}
    arr = np.array(img)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr, text

def _u16_to_meters(u16: np.ndarray, text: dict, dmax: float, default_scale_mm: Optional[float]) -> np.ndarray:
    # 1) PNG tEXt('scale','offset') → 2) default scale(mm) → 3) 0..65535↔0..dmax
    if ("scale" in text) or ("offset" in text):
        s = float(text.get("scale", "1000.0"))
        t = float(text.get("offset", "0.0"))
        depth = (u16.astype(np.float32) - t) / max(1e-12, s)
        return np.clip(depth, 0.0, dmax)
    if default_scale_mm is not None:
        depth = u16.astype(np.float32) / float(default_scale_mm)
        return np.clip(depth, 0.0, dmax)
    depth = (u16.astype(np.float32) / 65535.0) * dmax
    return np.clip(depth, 0.0, dmax)

def _read_float_or_u16_to_meters(path: str, dmax: float, scale_mm: Optional[float]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)  # meters assumed
        return np.clip(arr, 0.0, dmax)
    u16, text = _read_u16(path)
    return _u16_to_meters(u16, text, dmax=dmax, default_scale_mm=scale_mm)

# ==================== Save GT PNG16 with scale ====================
def save_depth_png16_with_scale(path: str, depth_m: np.ndarray, scale_mm: float = 1000.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    u16 = np.clip(d * scale_mm + 0.5, 0, 65535).astype(np.uint16)
    im = Image.fromarray(u16, mode="I;16")
    meta = PngInfo()
    meta.add_text("scale", f"{scale_mm}")
    meta.add_text("offset", "0.0")
    im.save(path, pnginfo=meta)

import hashlib

def _stable_int_seed(*parts) -> int:
    """입력 튜플로부터 프로세스 독립적인 고정 시드를 만든다."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7fffffff

def _make_sparse_from_gt_deterministic(GT_m: torch.Tensor, N: int, key_tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GT_m: (1,H,W) meters
    key_tuple: (h5_path, idx, crop_h, crop_w) 같이 샘플을 유일하게 식별하는 튜플
    """
    H, W = GT_m.shape[-2:]
    valid = (GT_m > 0.0).view(-1)                    # (H*W,)
    idx_all = valid.nonzero(as_tuple=False).view(-1) # 유효 픽셀 인덱스
    if idx_all.numel() == 0:
        ML = torch.zeros((1, H, W), dtype=torch.float32, device=GT_m.device)
        DL = torch.zeros_like(GT_m)
        return DL, ML

    # 고정 시드 생성 → 항상 같은 순서의 퍼뮤테이션
    seed = _stable_int_seed(*key_tuple, H, W)
    g = torch.Generator(device=GT_m.device)
    g.manual_seed(seed)

    sel = idx_all[torch.randperm(idx_all.numel(), generator=g)[:min(N, idx_all.numel())]]
    ML = torch.zeros((H*W,), dtype=torch.float32, device=GT_m.device)
    ML[sel] = 1.0
    ML = ML.view(1, H, W)

    DL = GT_m * ML
    return DL, ML


def _has_file(dirpath: str, names: List[str]) -> Optional[str]:
    """dirpath 안에 names 중 존재하는 첫 파일의 경로를 반환. 없으면 None."""
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.isfile(p):
            return p
    return None

def _list_subdirs(path: str) -> List[str]:
    try:
        return [os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        return []

def _resolve_estimation_dir_only(nyu: Dict[str,Any], split: str, rel_id: str) -> Optional[str]:
    """
    글롭 없이 해당 샘플 폴더 내에서만 탐색.
    우선순위: mono_root/viz/<split>/<id>/ -> mono_root/<split>/<id>/ -> (없으면 None)
    내부 파일 목록만 보고 *_viz.png, *_16bit.png, *.npy 등을 1개 선택.
    """
    mono_root = nyu.get("mono_root", None)
    if not mono_root:
        return None
    candidates = []
    for base in [
        os.path.join(mono_root, "viz", split, rel_id),
        os.path.join(mono_root, split, rel_id),
    ]:
        if os.path.isdir(base):
            try:
                files = os.listdir(base)
            except Exception:
                files = []
            # 1) *_viz.png 최우선
            vizs = [f for f in files if f.lower().endswith(".png") and "viz" in f.lower()]
            vizs.sort()
            if vizs:
                return os.path.join(base, vizs[0])
            # 2) 그 외 후보들
            exts = (".png", ".npy", ".tif", ".tiff")
            others = [f for f in files if f.lower().endswith(exts)]
            others.sort()
            if others:
                return os.path.join(base, others[0])
    return None

def _random_pick_entries(nyu: Dict[str,Any], split: str, k: int,
                         seed: int = 1, tries_per_pick: int = 200,
                         require_rgb: bool = False) -> List[Dict[str,str]]:
    """
    전체 스캔 없이 무작위로 샘플 k개를 고른다.
    - train_dir(or val_dir) 바로 아래의 'scene' 폴더 목록만 한 번 읽고,
    - 무작위 scene -> 무작위 sample 폴더로 내려가서 depth_mm16.png 존재 여부만 체크.
    - 필요하면 한 단계 더 내려가 시도 (scene 바로 아래가 sample 구조가 아닐 수도 있으므로).
    """
    split_dir = nyu.get(f"{split}_dir", os.path.join(nyu["root"], split))
    rng = random.Random(seed)

    scenes = _list_subdirs(split_dir)
    if not scenes:
        # train_dir 자체가 sample 디렉터리들일 수도 있음
        scenes = [split_dir]

    picks: List[Dict[str,str]] = []
    seen_ids = set()

    for _ in range(max(1, k)):
        found = False
        for _try in range(max(1, tries_per_pick)):
            scene = rng.choice(scenes)
            # level-1 후보
            level1 = _list_subdirs(scene)
            # sample 후보 후보군
            cand_dirs = level1 if level1 else [scene]

            sample_dir = rng.choice(cand_dirs)
            # depth 체크
            gt_path = _has_file(sample_dir, GT_CANDIDATES_MM16)
            if gt_path is None:
                # 한 단계 더 내려가 시도 (scene/sample/<id>/depth_mm16.png 형태 대응)
                level2 = _list_subdirs(sample_dir)
                if level2:
                    sample_dir2 = rng.choice(level2)
                    gt_path = _has_file(sample_dir2, GT_CANDIDATES_MM16)
                    if gt_path is not None:
                        sample_dir = sample_dir2

            if gt_path is None:
                continue

            rgb_path = _has_file(sample_dir, RGB_CANDIDATES)
            if require_rgb and (rgb_path is None):
                continue

            rel = os.path.relpath(sample_dir, split_dir).replace("\\", "/")
            if rel in seen_ids:
                continue

            picks.append({
                "img": rgb_path, "gt": gt_path, "id": rel
            })
            seen_ids.add(rel)
            found = True
            break

        if not found:
            break  # 더 못 찾으면 중단

    # estimation 경로는 선택된 샘플에 대해서만 해석(빠름)
    for it in picks:
        it["mono"] = _resolve_estimation_dir_only(nyu, split, it["id"])

    return picks