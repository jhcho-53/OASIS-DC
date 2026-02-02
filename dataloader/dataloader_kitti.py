# dataloader_kitti.py
import os, random, hashlib
from typing import Callable, Dict, List, Optional, Tuple, Sequence
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------- reproducibility
def seed_all(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

# --------------------------- I/O
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
            # KITTI 16-bit PNG는 mm 단위가 일반적 → m로 변환
            return (np.array(im, dtype=np.uint16).astype(np.float32) / 1000.0)
        return np.array(im, dtype=np.uint8).astype(np.float32)  # 이미 m로 저장된 8bit일 수도 있음

# --------------------------- geometry (KITTI: 기본 원본 해상도 사용)
def _to_tensor_rgb(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2,0,1).contiguous().float() / 255.0

def _keep_ratio_resize_rgb(img_u8: np.ndarray, target_short: int = 384) -> np.ndarray:
    H, W = img_u8.shape[:2]
    scale = target_short / min(H, W)
    newH, newW = int(round(H*scale)), int(round(W*scale))
    return np.array(Image.fromarray(img_u8).resize((newW, newH), Image.BILINEAR), dtype=np.uint8)

# --------------------------- sparse handling
def subsample_sparse_from_nonzero(depth_sparse_m: np.ndarray, n: Optional[int], seed: int, tag: str):
    """KITTI는 velodyne_raw가 이미 sparse 투영임. 원하면 n개로 서브샘플."""
    H, W = depth_sparse_m.shape
    ys, xs = np.where(depth_sparse_m > 0)
    sp = np.zeros((H,W), np.float32)
    mk = np.zeros((H,W), np.uint8)
    if len(ys) == 0:
        return sp, mk, np.zeros((0,2), dtype=np.int32)
    if n is None or n <= 0 or n >= len(ys):
        sp[ys, xs] = depth_sparse_m[ys, xs]; mk[ys, xs] = 1
        return sp, mk, np.stack([ys, xs], 1).astype(np.int32)
    rng = np.random.default_rng(seed ^ _stable_hash(tag))
    sel = rng.choice(len(ys), n, replace=False)
    y, x = ys[sel], xs[sel]
    sp[y, x] = depth_sparse_m[y, x]; mk[y, x] = 1
    return sp, mk, np.stack([y, x], 1).astype(np.int32)

# --------------------------- E_norm utils
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
    """RGB 컬러맵 → DL과의 상관 최대인 스칼라 선택."""
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

# --------------------------- dataset (KITTI)
class KITTIDepthForMCProp(Dataset):
    """
    list line (flexible):
      (A) <rel_id>
          -> image/<rel_id>, velodyne_raw/<rel_id>, groundtruth_depth/<rel_id>, est/<rel_id or ext-variant>
      (B) <image_rel> <gt_rel>
          -> velodyne_raw/<same_rel>, est/<same_rel or ext-variant>
      (C) <image_rel> <gt_rel> [<id>] [<est_rel>] [<sparse_rel>]
    """
    def __init__(
        self,
        root: str,
        list_file: str,
        seed: int = 0,
        emode: str = "precomputed",
        foundation_infer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pre_mono_dir: str = "est",
        image_dir: str = "image",
        sparse_dir: str = "velodyne_raw",
        gt_dir: str = "groundtruth_depth",
        pre_mono_exts: Sequence[str] = (".png",".npy",".npz",".tif",".tiff"),
        strict_precomputed: bool = True,
        normalize_on_valid_gt: bool = True,
        limit_sparse_to_n: Optional[int] = None,  # None이면 velodyne_raw 그대로, 정수면 그 수로 서브샘플
        foundation_short: int = 384,              # foundation_infer 입력 크기(짧은 변)
    ):
        super().__init__()
        self.root = root
        self.items = self._read_list(list_file, image_dir, gt_dir, pre_mono_dir, sparse_dir)
        self.seed = int(seed)
        self.emode = emode
        self.f_infer = foundation_infer
        self.pre_mono_dir = pre_mono_dir
        self.pre_mono_exts = tuple(pre_mono_exts)
        self.strict_precomputed = strict_precomputed
        self.normalize_on_valid_gt = normalize_on_valid_gt
        self.limit_sparse_to_n = limit_sparse_to_n
        self.foundation_short = int(foundation_short)
        self.cache_mono: Dict[str, np.ndarray] = {}

    def _read_list(self, p: str, image_dir: str, gt_dir: str, est_dir: str, sparse_dir: str) -> List[Tuple[str,str,str,Optional[str],Optional[str]]]:
        items = []
        with open(p, "r") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"): continue
                cols = s.split()
                # (A) one column: rel_id
                if len(cols) == 1:
                    rel = cols[0]
                    img_rel = os.path.join(image_dir, rel)
                    gt_rel  = os.path.join(gt_dir, rel)
                    est_rel = os.path.join(est_dir, rel)      # 실제 확장은 _resolve_precomputed_path에서 탐색
                    sp_rel  = os.path.join(sparse_dir, rel)
                    sid = os.path.splitext(os.path.basename(rel))[0]
                    items.append((img_rel, gt_rel, sid, est_rel, sp_rel))
                # (B) two columns: image_rel, gt_rel
                elif len(cols) == 2:
                    img_rel, gt_rel = cols[0], cols[1]
                    # 동일 상대경로를 sparse/est에도 사용
                    # 만약 상위 폴더명이 이미 포함되어 있으면 그대로 사용
                    def _maybe(prefix, rel):
                        return rel if rel.startswith(prefix + os.sep) else os.path.join(prefix, rel)
                    est_rel = _maybe(est_dir, os.path.relpath(gt_rel, gt_dir)) if gt_rel.startswith(gt_dir) else os.path.join(est_dir, gt_rel)
                    sp_rel  = _maybe(sparse_dir, os.path.relpath(gt_rel, gt_dir)) if gt_rel.startswith(gt_dir) else os.path.join(sparse_dir, gt_rel)
                    sid = os.path.splitext(os.path.basename(img_rel))[0]
                    items.append((img_rel, gt_rel, sid, est_rel, sp_rel))
                else:
                    # (C) >=3: image_rel gt_rel id [est_rel] [sparse_rel]
                    img_rel, gt_rel, sid = cols[0], cols[1], cols[2]
                    est_rel = cols[3] if len(cols) >= 4 else os.path.join(est_dir, os.path.relpath(gt_rel, gt_dir)) if gt_rel.startswith(gt_dir) else None
                    sp_rel  = cols[4] if len(cols) >= 5 else os.path.join("velodyne_raw", os.path.relpath(gt_rel, gt_dir)) if gt_rel.startswith(gt_dir) else None
                    items.append((img_rel, gt_rel, sid, est_rel, sp_rel))
        return items

    def __len__(self): return len(self.items)

    def _resolve_precomputed_path(self, est_rel_hint: Optional[str]) -> Optional[str]:
        """
        est_rel_hint가 파일이면 그대로 사용.
        없다면 같은 stem으로 pre_mono_exts 중 존재하는 것을 est/ 아래에서 탐색.
        """
        if est_rel_hint is not None:
            p = os.path.join(self.root, est_rel_hint)
            if os.path.exists(p): return p
        # hint가 없거나 파일이 없을 때, est 디렉토리 하위에서 stem에 맞춰 탐색
        # image/<...>/<stem>.png  →  est/<...>/<stem>.(png|npy|...)
        # est_rel_hint가 None인 경우는 items 만들 때 기본값을 넣지 못했을 수 있으므로 스킵
        return None

    # --- E_norm 생성
    def _load_precomputed_rel_raw(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
            if arr.ndim == 3: arr = arr[0] if arr.shape[0] in (1,3) else arr.squeeze()
            return arr.astype(np.float32)
        if ext == ".npz":
            z = np.load(path)
            for k in ["arr_0","rel","depth","D","d"]:
                if k in z:
                    arr = z[k]; break
            else:
                arr = list(z.values())[0]
            if arr.ndim == 3: arr = arr[0] if arr.shape[0] in (1,3) else arr.squeeze()
            return arr.astype(np.float32)
        with Image.open(path) as im:
            if im.mode in ["I;16","I;16B","I"]:
                return np.array(im, dtype=np.uint16).astype(np.float32)
            if im.mode in ["L","LA"]:
                return np.array(im, dtype=np.uint8).astype(np.float32)
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _compute_E_norm_precomputed(self, pre_path: str, base_size_hw: Tuple[int,int],
                                    D_gt: np.ndarray, DL: np.ndarray, ML: np.ndarray) -> np.ndarray:
        raw = self._load_precomputed_rel_raw(pre_path)  # HxW or HxWx3
        Ht, Wt = base_size_hw
        # est를 RGB해상도에 맞춤
        if raw.ndim == 2:
            if raw.shape != (Ht, Wt):
                raw = np.array(Image.fromarray(raw).resize((Wt,Ht), Image.BILINEAR if raw.dtype!=np.uint16 else Image.NEAREST))
            Er = raw.astype(np.float32)
        else:
            if raw.shape[:2] != (Ht, Wt):
                raw = np.array(Image.fromarray(raw).resize((Wt,Ht), Image.BILINEAR))
            Er = _best_gray_from_rgb(raw.astype(np.uint8), DL, ML) * 255.0  # 0~1 → 0~255
        mask = (D_gt > 0) if self.normalize_on_valid_gt else None
        E01 = _minmax01(Er, mask=mask)
        E01 = _orient_by_sparse(E01, DL, ML)
        return E01.astype(np.float32)

    def _compute_E_norm(self, mode: str, I_rgb_u8: np.ndarray, D_gt: np.ndarray, DL: np.ndarray, ML: np.ndarray,
                        est_rel_hint: Optional[str], sid: str):
        if sid in self.cache_mono:
            return self.cache_mono[sid]
        H, W = I_rgb_u8.shape[:2]
        if mode == "precomputed":
            # 우선 hint 사용, 없으면 est/<same_rel> 경로를 추정
            p = None
            if est_rel_hint is not None:
                p = os.path.join(self.root, est_rel_hint)
                if not os.path.exists(p): p = None
            if p is None:
                # image 경로와 동일한 상대경로가 est에도 있다고 가정
                # (items 생성 시 est_rel_hint 넣었으면 더 정확)
                # 실패해도 strict_precomputed가 True면 에러
                # 파일 확장자 대체 탐색은 생략(사용자가 est_rel_hint로 넘기는 것을 권장)
                pass
            if p is None:
                if self.strict_precomputed:
                    raise FileNotFoundError(f"[KITTIDepthForMCProp] precomputed est not found for id={sid}.")
                E01 = np.zeros((H,W), np.float32)
                self.cache_mono[sid] = E01; return E01
            E01 = self._compute_E_norm_precomputed(p, (H,W), D_gt, DL, ML)
            self.cache_mono[sid] = E01; return E01

        elif mode == "foundation":
            if self.f_infer is None:
                raise ValueError("[KITTIDepthForMCProp] foundation_infer required for emode='foundation'.")
            rgb_kr = _keep_ratio_resize_rgb(I_rgb_u8, self.foundation_short)
            I_kr = _to_tensor_rgb(rgb_kr).unsqueeze(0)  # (1,3,h,w)
            with torch.no_grad():
                rel = self.f_infer(I_kr)  # (1,Hm,Wm) 상대깊이
                Er_full = rel.detach().cpu().float().numpy()[0]
            Er = np.array(Image.fromarray(Er_full.astype(np.float32)).resize((W,H), Image.BILINEAR))
            E01 = _minmax01(Er, mask=(D_gt>0) if self.normalize_on_valid_gt else None)
            E01 = _orient_by_sparse(E01, DL, ML)
            self.cache_mono[sid] = E01; return E01

        elif mode == "zeros":
            E01 = np.zeros((H,W), np.float32)
            self.cache_mono[sid] = E01; return E01

        else:
            raise ValueError(f"Unknown emode: {mode}")

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        img_rel, gt_rel, sid, est_rel, sp_rel = self.items[i]

        rgb = _read_rgb(os.path.join(self.root, img_rel))
        D_gt_full = _read_depth(os.path.join(self.root, gt_rel))
        DL_full = _read_depth(os.path.join(self.root, sp_rel)) if sp_rel is not None else np.zeros_like(D_gt_full, np.float32)

        # KITTI: 원본 해상도 그대로 사용
        H, W = rgb.shape[:2]
        # valid GT
        valid = (D_gt_full > 0).astype(np.uint8)

        # 필요시 sparse를 n개로 서브샘플
        if self.limit_sparse_to_n is not None and self.limit_sparse_to_n > 0:
            DL, ML, idx = subsample_sparse_from_nonzero(DL_full, self.limit_sparse_to_n, self.seed, sid)
        else:
            DL = DL_full.astype(np.float32)
            ML = (DL_full > 0).astype(np.uint8)
            yy, xx = np.where(ML > 0)
            idx = np.stack([yy, xx], 1).astype(np.int32) if len(yy) > 0 else np.zeros((0,2), dtype=np.int32)

        # E_norm
        E01 = self._compute_E_norm(self.emode, rgb, D_gt_full, DL, ML, est_rel, sid)

        return {
            "I": _to_tensor_rgb(rgb),                 # (3,H,W) [0,1]
            "DL": torch.from_numpy(DL)[None],         # (1,H,W) metric sparse
            "ML": torch.from_numpy(ML).bool().float()[None],
            "E_norm": torch.from_numpy(E01)[None],    # (1,H,W) [0,1]
            "D_gt": torch.from_numpy(D_gt_full)[None],# (1,H,W)
            "meta": {
                "id": sid,
                "crop_box": torch.tensor((0,0,H,W), dtype=torch.int32),  # no crop
                "orig_size": torch.tensor((H,W), dtype=torch.int32),
                "valid_mask": torch.from_numpy(valid).bool(),
                "sparse_indices": torch.from_numpy(idx.astype(np.int64)),
                "mono_path": os.path.join(self.root, est_rel) if est_rel is not None else None
            }
        }

# --------------------------- builders
def build_kitti_dataloaders(
    root: str,
    train_list: str,
    test_list: str,
    seed: int = 0,
    emode: str = "precomputed",
    foundation_infer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pre_mono_dir: str = "est",
    image_dir: str = "image",
    sparse_dir: str = "velodyne_raw",
    gt_dir: str = "groundtruth_depth",
    pre_mono_exts: Sequence[str] = (".png",".npy",".npz",".tif",".tiff"),
    strict_precomputed: bool = True,
    normalize_on_valid_gt: bool = True,
    limit_sparse_to_n: Optional[int] = None,
    foundation_short: int = 384,
):
    """
    Returns: train_loader, test_loader
    - train_list / test_list: 각 줄은 (A) rel_id 또는 (B) image_rel gt_rel 또는 (C) image_rel gt_rel id [est_rel] [sparse_rel]
    """
    seed_all(seed)

    def _winit(worker_id:int):
        wseed = seed + worker_id
        np.random.seed(wseed); random.seed(wseed); torch.manual_seed(wseed)

    train_ds = KITTIDepthForMCProp(
        root=root, list_file=train_list, seed=seed,
        emode=emode, foundation_infer=foundation_infer,
        pre_mono_dir=pre_mono_dir, image_dir=image_dir, sparse_dir=sparse_dir, gt_dir=gt_dir,
        pre_mono_exts=pre_mono_exts, strict_precomputed=strict_precomputed,
        normalize_on_valid_gt=normalize_on_valid_gt,
        limit_sparse_to_n=limit_sparse_to_n, foundation_short=foundation_short
    )
    test_ds = KITTIDepthForMCProp(
        root=root, list_file=test_list, seed=seed,
        emode=emode, foundation_infer=foundation_infer,
        pre_mono_dir=pre_mono_dir, image_dir=image_dir, sparse_dir=sparse_dir, gt_dir=gt_dir,
        pre_mono_exts=pre_mono_exts, strict_precomputed=strict_precomputed,
        normalize_on_valid_gt=normalize_on_valid_gt,
        limit_sparse_to_n=limit_sparse_to_n, foundation_short=foundation_short
    )

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=_winit, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=_winit
    )
    return train_loader, test_loader
