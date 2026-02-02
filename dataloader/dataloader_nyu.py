# mcprop_dataloader.py  (PNG precomputed-ready for MCPropNet)
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
            return (np.array(im, dtype=np.uint16).astype(np.float32) / 1000.0)  # mm->m
        return np.array(im, dtype=np.uint8).astype(np.float32)  # assume meters if 8-bit

# --------------------------- geometry
def _resize_to_320x240(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    if (H, W) == (480, 640):
        resample = Image.BILINEAR if img.ndim == 3 else Image.NEAREST
        return np.array(Image.fromarray(img).resize((320, 240), resample))
    return img

def _center_crop(img: np.ndarray, size_hw: Tuple[int,int]) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    Hc, Wc = size_hw
    H, W = img.shape[:2]
    if H < Hc or W < Wc:
        scale = max(Hc/H, Wc/W)
        newW, newH = int(round(W*scale)), int(round(H*scale))
        resample = Image.BILINEAR if img.ndim == 3 else Image.NEAREST
        img = np.array(Image.fromarray(img).resize((newW, newH), resample))
        H, W = img.shape[:2]
    top, left = (H - Hc)//2, (W - Wc)//2
    if img.ndim == 3:
        out = img[top:top+Hc, left:left+Wc, :]
    else:
        out = img[top:top+Hc, left:left+Wc]
    return out, (top, left, Hc, Wc)

def _to_tensor_rgb(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2,0,1).contiguous().float() / 255.0

def _keep_ratio_resize_rgb(img_u8: np.ndarray, target_short: int = 256) -> np.ndarray:
    H, W = img_u8.shape[:2]
    scale = target_short / min(H, W)
    newH, newW = int(round(H*scale)), int(round(W*scale))
    return np.array(Image.fromarray(img_u8).resize((newW, newH), Image.BILINEAR), dtype=np.uint8)

# --------------------------- sparse sampling
def sample_sparse_points(depth_m: np.ndarray, n: int, seed: int, tag: str):
    H, W = depth_m.shape
    ys, xs = np.where(depth_m > 0)
    sp = np.zeros((H,W), np.float32)
    mk = np.zeros((H,W), np.uint8)
    if len(ys) == 0:
        return sp, mk, np.zeros((0,2), dtype=np.int32)
    n_eff = min(n, len(ys))
    rng = np.random.default_rng(seed ^ _stable_hash(tag))
    sel = rng.choice(len(ys), n_eff, replace=False)
    y, x = ys[sel], xs[sel]
    sp[y, x] = depth_m[y, x]; mk[y, x] = 1
    return sp, mk, np.stack([y,x], 1).astype(np.int32)

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
    """rgb: HxWx3 uint8 -> choose scalar map best correlated (abs) with DL on ML."""
    rgbf = rgb.astype(np.float32) / 255.0
    R, G, B = rgbf[...,0], rgbf[...,1], rgbf[...,2]
    luma = 0.299*R + 0.587*G + 0.114*B
    vmax = np.max(rgbf, axis=2)          # HSV V (근사)
    cand = [("luma", luma), ("v", vmax), ("r", R), ("g", G), ("b", B)]
    yy, xx = np.where(ML > 0)
    if len(yy) < 5:
        return luma.astype(np.float32)
    best, best_val = luma, -1.0
    d = DL[yy, xx].astype(np.float64); d = (d - d.mean()) / (d.std() + 1e-9)
    for name, a in cand:
        e = a[yy, xx].astype(np.float64)
        e = (e - e.mean()) / (e.std() + 1e-9)
        val = abs(float(np.mean(e * d)))
        if val > best_val:
            best_val, best = val, a
    return best.astype(np.float32)

# --------------------------- dataset (precomputed PNG-ready)
class NYUv2ForMCProp(Dataset):
    """
    list line:
      <rgb_rel> <depth_rel> [<id>] [<mono_rel_precomputed>]  # mono가 없으면 pre_mono_dir에서 자동 탐색
    """
    def __init__(
        self,
        root: str,
        list_file: str,
        seed: int = 0,
        n_sparse: int = 500,
        emode: str = "precomputed",
        foundation_infer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        midas_short: int = 256,
        pre_mono_dir: Optional[str] = "mono_rel",        # 동일 root 내부
        pre_mono_exts: Sequence[str] = (".png",".npy",".npz",".tif",".tiff"),
        strict_precomputed: bool = True,
        normalize_on_valid_gt: bool = True,
    ):
        super().__init__()
        self.root = root
        self.items = self._read_list(list_file)
        self.seed = int(seed)
        self.n_sparse = int(n_sparse)
        self.emode = emode
        self.f_infer = foundation_infer
        self.midas_short = int(midas_short)
        self.pre_mono_dir = pre_mono_dir
        self.pre_mono_exts = tuple(pre_mono_exts)
        self.strict_precomputed = strict_precomputed
        self.normalize_on_valid_gt = normalize_on_valid_gt
        self.cache_sparse = {}
        self.cache_mono = {}

    def _read_list(self, p: str) -> List[Tuple[str,str,str,Optional[str]]]:
        items = []
        with open(p, "r") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"): continue
                cols = s.split()
                if len(cols) < 2: raise ValueError(f"Bad line: {s}")
                rgb_rel, dep_rel = cols[0], cols[1]
                sid = cols[2] if len(cols) >= 3 else os.path.splitext(os.path.basename(rgb_rel))[0]
                mono_rel = cols[3] if len(cols) >= 4 else None
                items.append((rgb_rel, dep_rel, sid, mono_rel))
        return items

    def __len__(self): return len(self.items)

    def _resolve_precomputed_path(self, rgb_rel: str, mono_rel_col: Optional[str]) -> Optional[str]:
        if mono_rel_col is not None:
            p = os.path.join(self.root, mono_rel_col)
            return p if os.path.exists(p) else None
        if self.pre_mono_dir is None:
            return None
        # images/train/0001.png  ->  <pre_mono_dir>/train/0001.<ext>, 또는 <pre_mono_dir>/0001.<ext>
        parts = rgb_rel.split(os.sep)
        sub_after_top = "/".join(parts[1:-1])
        stem = os.path.splitext(os.path.basename(rgb_rel))[0]
        cands = []
        for ext in self.pre_mono_exts:
            if sub_after_top:
                cands.append(os.path.join(self.root, self.pre_mono_dir, sub_after_top, f"{stem}{ext}"))
            cands.append(os.path.join(self.root, self.pre_mono_dir, f"{stem}{ext}"))
        for p in cands:
            if os.path.exists(p): return p
        return None

    # --- RGB/Depth pipelines
    def _rgb_pipeline(self, rgb_u8: np.ndarray):
        rgb_u8 = _resize_to_320x240(rgb_u8)
        rgb_c, crop_box = _center_crop(rgb_u8, (228,304))
        I = _to_tensor_rgb(rgb_c)
        rgb_kr = _keep_ratio_resize_rgb(rgb_u8, self.midas_short)  # API 유지
        I_kr = _to_tensor_rgb(rgb_kr)
        return I, I_kr, crop_box, rgb_u8

    def _depth_pipeline(self, depth_m: np.ndarray, crop_box):
        depth_m = _resize_to_320x240(depth_m)
        top, left, Hc, Wc = crop_box
        D_gt = depth_m[top:top+Hc, left:left+Wc].astype(np.float32)
        valid = (D_gt > 0).astype(np.uint8)
        return D_gt, valid

    def _sparse(self, D_gt: np.ndarray, sid: str):
        if sid in self.cache_sparse:
            return self.cache_sparse[sid]
        DL, ML, idx = sample_sparse_points(D_gt, self.n_sparse, self.seed, sid)
        self.cache_sparse[sid] = (DL, ML, idx)
        return DL, ML, idx

    # --- precomputed loader (PNG/NPY/NPZ)
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
                    arr = z[k]; 
                    break
            else:
                arr = list(z.values())[0]
            if arr.ndim == 3: arr = arr[0] if arr.shape[0] in (1,3) else arr.squeeze()
            return arr.astype(np.float32)
        # images (.png/.tif/.tiff)
        with Image.open(path) as im:
            if im.mode in ["I;16","I;16B","I"]:         # 16-bit gray
                return np.array(im, dtype=np.uint16).astype(np.float32)
            if im.mode in ["L","LA"]:                   # 8-bit gray
                return np.array(im, dtype=np.uint8).astype(np.float32)
            # RGB/RGBA -> 반환은 HxWx3 uint8 (나중에 DL,ML 보고 스칼라로 압축)
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _compute_E_norm_precomputed(self, pre_path: str, crop_box, base_320x240: np.ndarray,
                                    D_gt: np.ndarray, DL: np.ndarray, ML: np.ndarray) -> np.ndarray:
        raw = self._load_precomputed_rel_raw(pre_path)  # HxW or HxWx3
        top, left, Hc, Wc = crop_box
        # 사이즈 정렬(→320x240), 동일 크롭
        if raw.ndim == 2:
            Ht, Wt = base_320x240.shape[:2]
            if raw.shape != (Ht, Wt):
                raw = np.array(Image.fromarray(raw).resize((Wt,Ht), Image.BILINEAR if raw.dtype!=np.uint16 else Image.NEAREST))
            Er_c = raw[top:top+Hc, left:left+Wc].astype(np.float32)
        else:
            Ht, Wt = base_320x240.shape[:2]
            if raw.shape[:2] != (Ht, Wt):
                raw = np.array(Image.fromarray(raw).resize((Wt,Ht), Image.BILINEAR))
            Er_c_rgb = raw[top:top+Hc, left:left+Wc, :].astype(np.uint8)
            # RGB 컬러맵 → 스칼라 자동 복원(상관 최대 후보 선택)
            Er_c = _best_gray_from_rgb(Er_c_rgb, DL, ML) * 255.0  # 0~1 -> 0~255 스케일로 맞춤

        # [0,1] 정규화(유효 GT 기준 권장)
        mask = (D_gt > 0) if self.normalize_on_valid_gt else None
        E01 = _minmax01(Er_c, mask=mask)
        # near-far 자동 교정
        E01 = _orient_by_sparse(E01, DL, ML)
        return E01.astype(np.float32)

    def _compute_E_norm(self, mode: str, I_kr: torch.Tensor, crop_box, base_320x240: np.ndarray,
                        D_gt: np.ndarray, DL: np.ndarray, ML: np.ndarray, mono_rel_col: Optional[str],
                        rgb_rel: str, sid: str):
        if sid in self.cache_mono: 
            return self.cache_mono[sid]

        if mode == "precomputed":
            p = self._resolve_precomputed_path(rgb_rel, mono_rel_col)
            if p is None:
                if self.strict_precomputed:
                    raise FileNotFoundError(f"[NYUv2ForMCProp] precomputed mono not found for id={sid}.")
                # fallback
                Hc, Wc = D_gt.shape
                E01 = np.zeros((Hc,Wc), np.float32)
                self.cache_mono[sid] = E01
                return E01
            E01 = self._compute_E_norm_precomputed(p, crop_box, base_320x240, D_gt, DL, ML)
            self.cache_mono[sid] = E01
            return E01

        elif mode == "foundation":
            if self.f_infer is None:
                raise ValueError("[NYUv2ForMCProp] foundation_infer required for emode='foundation'.")
            with torch.no_grad():
                rel = self.f_infer(I_kr)  # (1,Hm,Wm)
                Er_full = rel.detach().cpu().float().numpy()[0]
            Ht, Wt = base_320x240.shape[:2]
            Er = np.array(Image.fromarray(Er_full.astype(np.float32)).resize((Wt,Ht), Image.BILINEAR))
            top, left, Hc, Wc = crop_box
            Er_c = Er[top:top+Hc, left:left+Wc]
            E01 = _minmax01(Er_c, mask=(D_gt>0) if self.normalize_on_valid_gt else None)
            E01 = _orient_by_sparse(E01, DL, ML)
            self.cache_mono[sid] = E01
            return E01

        elif mode == "zeros":
            Hc, Wc = D_gt.shape
            E01 = np.zeros((Hc,Wc), np.float32)
            self.cache_mono[sid] = E01
            return E01

        else:
            raise ValueError(f"Unknown emode: {mode}")

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rgb_rel, dep_rel, sid, mono_rel = self.items[i]

        rgb = _read_rgb(os.path.join(self.root, rgb_rel))
        dep = _read_depth(os.path.join(self.root, dep_rel))

        I, I_kr, crop_box, base_320x240 = self._rgb_pipeline(rgb)
        D_gt, valid = self._depth_pipeline(dep, crop_box)
        DL, ML, idx = self._sparse(D_gt, sid)

        E01 = self._compute_E_norm(self.emode, I_kr, crop_box, base_320x240, D_gt, DL, ML,
                                   mono_rel, rgb_rel, sid)

        return {
            "I": I,                                     # (3,228,304) [0,1]
            "DL": torch.from_numpy(DL)[None],          # (1,228,304) metric sparse
            "ML": torch.from_numpy(ML).bool().float()[None],
            "E_norm": torch.from_numpy(E01)[None],     # (1,228,304) [0,1]
            "D_gt": torch.from_numpy(D_gt)[None],      # (1,228,304)
            "meta": {
                "id": sid,
                "crop_box": torch.tensor(crop_box, dtype=torch.int32),
                "orig_size": torch.tensor(rgb.shape[:2], dtype=torch.int32),
                "valid_mask": torch.from_numpy(valid).bool(),
                "sparse_indices": torch.from_numpy(idx.astype(np.int64)),
                "mono_path": self._resolve_precomputed_path(rgb_rel, mono_rel)
            }
        }

# --------------------------- builders
def build_mcprop_dataloaders(
    root: str,
    train_list: str,
    test_list: str,
    seed: int = 0,
    shots: Optional[int] = None,
    n_sparse: int = 500,
    emode: str = "precomputed",
    foundation_infer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    midas_short: int = 256,
    pre_mono_dir: Optional[str] = "mono_rel",
    pre_mono_exts: Sequence[str] = (".png",".npy",".npz",".tif",".tiff"),
    strict_precomputed: bool = True,
    normalize_on_valid_gt: bool = True,
):
    seed_all(seed)

    def make_few_shot_split(list_file: str, out_file: str, k: int, seed: int) -> List[str]:
        with open(list_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        if k > len(lines): raise ValueError(f"shots={k} > dataset size={len(lines)}")
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(lines), k, replace=False).tolist())
        chosen = [lines[i] for i in sel]
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f: f.write("\n".join(chosen))
        return chosen

    if shots is not None:
        kfile = os.path.splitext(train_list)[0] + f"_{shots}shot_seed{seed}.txt"
        if not os.path.exists(kfile):
            make_few_shot_split(train_list, kfile, shots, seed)
        train_use = kfile
    else:
        train_use = train_list

    train_ds = NYUv2ForMCProp(
        root, train_use, seed=seed, n_sparse=n_sparse,
        emode=emode, foundation_infer=foundation_infer, midas_short=midas_short,
        pre_mono_dir=pre_mono_dir, pre_mono_exts=pre_mono_exts,
        strict_precomputed=strict_precomputed, normalize_on_valid_gt=normalize_on_valid_gt
    )
    test_ds = NYUv2ForMCProp(
        root, test_list, seed=seed, n_sparse=n_sparse,
        emode=emode, foundation_infer=foundation_infer, midas_short=midas_short,
        pre_mono_dir=pre_mono_dir, pre_mono_exts=pre_mono_exts,
        strict_precomputed=strict_precomputed, normalize_on_valid_gt=normalize_on_valid_gt
    )

    def _winit(worker_id:int):
        wseed = seed + worker_id
        np.random.seed(wseed); random.seed(wseed); torch.manual_seed(wseed)

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
