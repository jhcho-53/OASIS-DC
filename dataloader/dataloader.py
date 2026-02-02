import os
import glob
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import random

# ----- Utils -----

def _exists(p: Optional[str]) -> Optional[str]:
    return p if (p and os.path.exists(p)) else None

def _first_existing(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _read_rgb(path: str) -> torch.Tensor:
    """RGB: [3,H,W], float32 in [0,1]"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t

def _read_depth(path: str, scale_if_uint: float = 256.0) -> torch.Tensor:
    """
    Read depth-like PNG → [1,H,W] float32.
    - 정수형(u8/u16/u32): 값/scale_if_uint
    - float형: 그대로
    - 3채널 PNG(그레이를 3채널로 저장/컬러맵): 단일 채널로 축소
      * 세 채널이 동일하면 첫 채널 사용
      * 아니면 luma(Y) = 0.299R + 0.587G + 0.114B
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                # convert to perceived luminance
                arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(arr.dtype)
        else:
            arr = arr[..., 0]

    if arr.dtype.kind in ("u", "i"):
        depth = arr.astype(np.float32) / (scale_if_uint if scale_if_uint else 1.0)
    else:
        depth = arr.astype(np.float32)
    return torch.from_numpy(depth).unsqueeze(0)  # [1,H,W]

def _make_mask(depth: torch.Tensor) -> torch.Tensor:
    return (depth > 0).to(torch.float32)

def _resize_like(t: torch.Tensor, size_hw: Tuple[int, int], is_depth: bool) -> torch.Tensor:
    """t: [C,H,W] → resize to size_hw=(H,W). depth/mask use nearest, rgb uses bilinear."""
    assert t.ndim == 3, f"Expected [C,H,W], got {tuple(t.shape)}"
    mode = "nearest" if is_depth else "bilinear"
    t4 = t.unsqueeze(0)
    if mode == "bilinear":
        out = F.interpolate(t4, size=size_hw, mode=mode, align_corners=False).squeeze(0)
    else:
        out = F.interpolate(t4, size=size_hw, mode=mode).squeeze(0)
    return out

# ----- Simple transforms operating on dict(sample) -----

class Compose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, sample: Dict) -> Dict:
        for op in self.ops:
            sample = op(sample)
        return sample

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5): self.p = p
    def __call__(self, s: Dict) -> Dict:
        if random.random() < self.p:
            for k in ["rgb", "sparse", "gt", "mask_sparse", "mask_gt", "pseudo", "mask_pseudo", "mono", "mask_mono"]:
                if k in s and isinstance(s[k], torch.Tensor):
                    s[k] = torch.flip(s[k], dims=[2])  # flip width (W)
        return s

class ColorJitterLite:
    """RGB tensor jitter: brightness/contrast/saturation (hue 생략)"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1):
        self.b = brightness; self.c = contrast; self.s = saturation
    def __call__(self, s: Dict) -> Dict:
        if "rgb" not in s: return s
        x = s["rgb"]
        # brightness
        b = 1.0 + (random.uniform(-self.b, self.b))
        x = torch.clamp(x * b, 0.0, 1.0)
        # contrast
        if self.c > 1e-6:
            mean = x.mean(dim=(1,2), keepdim=True)
            c = 1.0 + (random.uniform(-self.c, self.c))
            x = torch.clamp((x - mean) * c + mean, 0.0, 1.0)
        # saturation
        if self.s > 1e-6 and x.shape[0] >= 3:
            gray = x.mean(dim=0, keepdim=True)
            sat = 1.0 + (random.uniform(-self.s, self.s))
            x = torch.clamp((x - gray) * sat + gray, 0.0, 1.0)
        s["rgb"] = x
        return s

class NormalizeRGB:
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
    def __call__(self, s: Dict) -> Dict:
        if "rgb" in s and isinstance(s["rgb"], torch.Tensor):
            s["rgb"] = (s["rgb"] - self.mean) / self.std
        return s

class ResizeTo:
    """Resize all modalities to fixed HxW."""
    def __init__(self, size_hw: Tuple[int,int]): self.size_hw = (int(size_hw[0]), int(size_hw[1]))
    def __call__(self, s: Dict) -> Dict:
        H, W = self.size_hw
        for k in ["rgb", "sparse", "gt", "mask_sparse", "mask_gt", "pseudo", "mask_pseudo", "mono", "mask_mono"]:
            if k in s and isinstance(s[k], torch.Tensor):
                is_depth = (k.startswith("mask") or k in ["sparse","gt","pseudo","mono"])
                s[k] = _resize_like(s[k], (H,W), is_depth=is_depth)
                # 마스크는 이진 유지
                if k.startswith("mask"):
                    s[k] = (s[k] > 0.5).float()
        return s

class AlignToRGBSize:
    """RGB의 (H,W)에 나머지 모달리티를 정렬"""
    def __call__(self, s: Dict) -> Dict:
        if "rgb" not in s or not isinstance(s["rgb"], torch.Tensor):
            return s
        H, W = s["rgb"].shape[1], s["rgb"].shape[2]
        for k in ["sparse","gt","mask_sparse","mask_gt","pseudo","mask_pseudo","mono","mask_mono"]:
            if k in s and isinstance(s[k], torch.Tensor) and s[k].shape[-2:] != (H,W):
                is_depth = (k.startswith("mask") or k in ["sparse","gt","pseudo","mono"])
                s[k] = _resize_like(s[k], (H,W), is_depth=is_depth)
                if k.startswith("mask"):
                    s[k] = (s[k] > 0.5).float()
        return s

class DepthClamp:
    """
    Clamp depths:
      - metric depth류(sparse/gt/pseudo)는 [0, max_depth]
      - mono(E)는 기본적으로 [0,1] (mono_range=(0,1)); 필요시 None으로 꺼도 됨
    """
    def __init__(self, max_depth: float = 80.0, mono_range: Optional[Tuple[float,float]] = (0.0, 1.0)):
        self.md = float(max_depth)
        self.mono_range = mono_range
    def __call__(self, s: Dict) -> Dict:
        # metric depths
        for k in ["sparse","gt","pseudo"]:
            if k in s and isinstance(s[k], torch.Tensor):
                s[k] = torch.clamp(s[k], 0.0, self.md)
        # mono range (normalized)
        if "mono" in s and isinstance(s["mono"], torch.Tensor):
            if self.mono_range is not None:
                lo, hi = self.mono_range
                s["mono"] = torch.clamp(s["mono"], float(lo), float(hi))
        # Recompute masks (0 invalid)
        for k, mname in [("sparse","mask_sparse"),("gt","mask_gt"),("pseudo","mask_pseudo"),("mono","mask_mono")]:
            if k in s and isinstance(s[k], torch.Tensor):
                s[mname] = _make_mask(s[k])
        return s

def default_transforms(train: bool,
                       resize_hw: Optional[Tuple[int,int]] = None,
                       max_depth: float = 80.0,
                       hflip_p: float = 0.5,
                       use_color_jitter: bool = True,
                       mono_is_normalized: bool = True,
                       normalize_rgb: bool = False,
                       align_to_rgb_if_no_resize: bool = True) -> Compose:
    """
    - RGB: [0,1] (Normalize는 옵션)
    - mono: [0,1]을 가정(mono_is_normalized=True일 때 강제 클램프)
    - 깊이류(sparse/gt/pseudo): metric [0,max_depth] 클램프
    - 크기: resize_hw가 주어지면 고정 리사이즈, 아니면 RGB 크기에 정렬(옵션)
    """
    ops = []
    if train:
        if hflip_p > 0: ops.append(RandomHorizontalFlip(p=hflip_p))
        if use_color_jitter: ops.append(ColorJitterLite(0.2, 0.2, 0.1))
    # 깊이/mono 범위 보정 및 마스크 재계산
    mono_range = (0.0, 1.0) if mono_is_normalized else None
    ops.append(DepthClamp(max_depth=max_depth, mono_range=mono_range))
    # 리사이즈 또는 정렬
    if resize_hw is not None:
        ops.append(ResizeTo(resize_hw))
    elif align_to_rgb_if_no_resize:
        ops.append(AlignToRGBSize())
    # (선택) RGB 정규화
    if normalize_rgb:
        ops.append(NormalizeRGB())
    return Compose(ops)

# ----- Indexing (shared with Dataset) -----

def index_items(root: str, split: str = "train") -> List[Dict]:
    """
    Build item list from shot root that mirrors KITTI-DC.
    Expected:
      - data_depth_annotated/<split>/.../proj_depth/groundtruth/image_02/<frame>.png
      - data_depth_velodyne/<split>/.../proj_depth/velodyne_raw/image_02/<frame>.png
      - RGB candidates tried in order (kitti_raw*, data_rgb*).
      - Optional extras:
          pseudo_depth_map/<split>/.../proj_depth/velodyne_raw/image_02/<frame>.png
          kitti_raw_da/<split>/.../proj_depth/image_02/<frame>.png
    """
    pat = os.path.join(root, "data_depth_annotated", split, "**",
                       "proj_depth", "groundtruth", "image_02", "*.png")
    gt_files = sorted(glob.glob(pat, recursive=True))
    items = []
    for g in gt_files:
        parts = os.path.normpath(g).split(os.sep)
        if split not in parts or "proj_depth" not in parts:
            continue
        i_split = parts.index(split)
        i_proj  = parts.index("proj_depth")
        drive = os.path.join(*parts[i_split+1:i_proj])
        frame = os.path.splitext(os.path.basename(g))[0]

        sparse = os.path.join(root, "data_depth_velodyne", split, drive,
                              "proj_depth", "velodyne_raw", "image_02", f"{frame}.png")

        rgb = _first_existing([
            os.path.join(root, "kitti_raw", split, drive, "proj_depth", "image_02", f"{frame}.png"),
            os.path.join(root, "data_rgb", split, drive, "image_02", "data", f"{frame}.png"),
            os.path.join(root, "data_rgb", split, drive, "image_02", f"{frame}.png"),
            os.path.join(root, "kitti_raw", split, drive, "image_02", "data", f"{frame}.png"),
            os.path.join(root, "kitti_raw", split, drive, "image_02", f"{frame}.png"),
        ])

        pseudo = os.path.join(root, "pseudo_depth_map", split, drive,
                              "proj_depth", "velodyne_raw", "image_02", f"{frame}.png")
        mono   = os.path.join(root, "kitti_raw_da", split, drive,
                              "proj_depth", "image_02", f"{frame}.png")

        rec = {
            "id": f"{drive}/{frame}",
            "drive": drive, "frame": frame,
            "gt": _exists(g),
            "sparse": _exists(sparse),
            "rgb": _exists(rgb),
            "pseudo": _exists(pseudo),
            "mono": _exists(mono),
        }
        if rec["gt"] and rec["sparse"] and rec["rgb"]:
            items.append(rec)

    items.sort(key=lambda x: x["id"])
    return items

# ----- Dataset -----

class KittiDepthShotDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        include_pseudo: Union[str, bool] = "auto",  # "auto" | True | False
        include_mono: Union[str, bool] = "auto",    # "auto" | True | False
        depth_scale: float = 256.0,
        pseudo_scale: Optional[float] = None,
        mono_scale: Optional[float] = None,
        transform = None,
        items: Optional[List[Dict]] = None,
        force_use_pseudo: Optional[bool] = None,
        force_use_mono: Optional[bool] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.depth_scale = float(depth_scale)
        self.pseudo_scale = float(depth_scale if pseudo_scale is None else pseudo_scale)
        self.mono_scale = float(255.0 if mono_scale is None else mono_scale)  # mono는 기본 0..255 → 0..1 추천
        self.transform = transform

        self.items = index_items(self.root, split) if items is None else items
        if not self.items:
            raise RuntimeError(f"No samples under {self.root} (split={split}).")

        def all_exist(key: str) -> bool:
            return all(bool(it.get(key)) for it in self.items)

        # Extras flags
        if force_use_pseudo is not None:
            self.use_pseudo = force_use_pseudo
        else:
            if include_pseudo == "auto": self.use_pseudo = all_exist("pseudo")
            elif include_pseudo is True:
                self.use_pseudo = True
                self.items = [it for it in self.items if it.get("pseudo")]
                if not self.items:
                    raise RuntimeError("include_pseudo=True but no items have pseudo.")
            else:
                self.use_pseudo = False

        if force_use_mono is not None:
            self.use_mono = force_use_mono
        else:
            if include_mono == "auto": self.use_mono = all_exist("mono")
            elif include_mono is True:
                self.use_mono = True
                self.items = [it for it in self.items if it.get("mono")]
                if not self.items:
                    raise RuntimeError("include_mono=True but no items have mono.")
            else:
                self.use_mono = False

        print(f"[KittiDepthShotDataset] root={self.root} split={split} n={len(self.items)} "
              f"| pseudo={'on' if self.use_pseudo else 'off'} "
              f"| mono={'on' if self.use_mono else 'off'}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        sample = {
            "id": it["id"], "drive": it["drive"], "frame": it["frame"],
            "paths": {k: it[k] for k in ("rgb","sparse","gt","pseudo","mono")}
        }
        rgb = _read_rgb(it["rgb"])
        sparse = _read_depth(it["sparse"], scale_if_uint=self.depth_scale)
        gt = _read_depth(it["gt"], scale_if_uint=self.depth_scale)
        sample.update({
            "rgb": rgb,
            "sparse": sparse, "mask_sparse": _make_mask(sparse),
            "gt": gt,         "mask_gt": _make_mask(gt),
        })
        if self.use_pseudo and it.get("pseudo"):
            pseudo = _read_depth(it["pseudo"], scale_if_uint=self.pseudo_scale)
            sample["pseudo"] = pseudo
            sample["mask_pseudo"] = _make_mask(pseudo)
        if self.use_mono and it.get("mono"):
            mono = _read_depth(it["mono"], scale_if_uint=self.mono_scale)
            sample["mono"] = mono
            sample["mask_mono"] = _make_mask(mono)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

# ----- Collate -----

def collate_pad_to_max(batch: List[Dict]) -> Dict:
    out: Dict = {
        "id": [b["id"] for b in batch],
        "drive": [b["drive"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "paths": [b["paths"] for b in batch],
    }
    H_max = max(b["rgb"].shape[1] for b in batch)
    W_max = max(b["rgb"].shape[2] for b in batch)

    def _to_3d(t: torch.Tensor) -> torch.Tensor:
        # 기대 [C,H,W]; 2D/4D 방어
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 4:
            if t.shape[0] == 1:
                t = t.squeeze(0)
            elif t.shape[-1] in (1, 3):
                t = t.permute(0, 3, 1, 2)
                t = t.squeeze(0)
        assert t.ndim == 3, f"collate expects [C,H,W], got {tuple(t.shape)}"
        return t

    def _pad(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        t = _to_3d(t)
        _, h, w = t.shape
        return F.pad(t, (0, W - w, 0, H - h))

    def _stack(key: str) -> Optional[torch.Tensor]:
        if key not in batch[0]: return None
        return torch.stack([_pad(b[key], H_max, W_max) for b in batch], dim=0)

    for key in ["rgb","sparse","gt","mask_sparse","mask_gt","pseudo","mask_pseudo","mono","mask_mono"]:
        if key in batch[0]:
            out[key] = _stack(key)
    return out

# ----- Split + Loader helpers -----

def _deterministic_split(n: int, val_ratio: float = 0.3, split_seed: int = 42) -> Tuple[List[int], List[int]]:
    if n <= 1:
        return [0], [0]  # 1-shot: same sample in train & val
    idx = list(range(n))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(idx)
    train_count = int(np.floor(n * (1.0 - val_ratio)))
    train_count = max(1, min(train_count, n - 1))  # ensure both non-empty
    train_idx = idx[:train_count]
    val_idx = idx[train_count:]
    if len(val_idx) == 0:  # safety
        val_idx = [idx[-1]]
        train_idx = idx[:-1]
    return train_idx, val_idx

def make_train_val_loaders(
    shot_root: str,
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    include_pseudo: Union[str,bool] = "auto",
    include_mono: Union[str,bool] = "auto",
    depth_scale: float = 256.0,
    pseudo_scale: Optional[float] = None,
    mono_scale: Optional[float] = None,
    val_ratio: float = 0.3,
    split_seed: int = 42,
    # transforms
    resize_hw: Optional[Tuple[int,int]] = None,  # e.g., (352,1216)
    hflip_p: float = 0.5,
    use_color_jitter: bool = True,
    mono_is_normalized: bool = True,
    normalize_rgb: bool = False,
    align_to_rgb_if_no_resize: bool = True,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    root = os.path.abspath(shot_root)
    # 1) index once and resolve extras policy on full set
    full_items = index_items(root, split)
    if not full_items:
        raise RuntimeError(f"No items in {root} split={split}")

    # Resolve extras flags on the full set for consistency
    def _all_exist(items: List[Dict], key: str) -> bool:
        return all(bool(it.get(key)) for it in items)

    if include_pseudo == "auto": use_pseudo = _all_exist(full_items, "pseudo")
    elif include_pseudo is True: use_pseudo = True
    else: use_pseudo = False

    if include_mono == "auto": use_mono = _all_exist(full_items, "mono")
    elif include_mono is True: use_mono = True
    else: use_mono = False

    # 2) split indices (1-shot ⇒ same in train/val)
    n = len(full_items)
    train_idx, val_idx = _deterministic_split(n, val_ratio=val_ratio, split_seed=split_seed)
    train_items = [full_items[i] for i in train_idx]
    val_items   = [full_items[i] for i in val_idx]

    # 3) datasets with respective transforms
    train_tf = default_transforms(
        train=True,  resize_hw=resize_hw, hflip_p=hflip_p,
        use_color_jitter=use_color_jitter, max_depth=float('inf') if depth_scale is None else 80.0,
        mono_is_normalized=mono_is_normalized, normalize_rgb=normalize_rgb,
        align_to_rgb_if_no_resize=align_to_rgb_if_no_resize
    )
    val_tf   = default_transforms(
        train=False, resize_hw=resize_hw, hflip_p=0.0,
        use_color_jitter=False, max_depth=float('inf') if depth_scale is None else 80.0,
        mono_is_normalized=mono_is_normalized, normalize_rgb=normalize_rgb,
        align_to_rgb_if_no_resize=align_to_rgb_if_no_resize
    )

    train_ds = KittiDepthShotDataset(
        root=root, split=split,
        include_pseudo=True if use_pseudo else False,
        include_mono=True if use_mono else False,
        depth_scale=depth_scale, pseudo_scale=pseudo_scale, mono_scale=mono_scale,
        transform=train_tf, items=train_items,
        force_use_pseudo=use_pseudo, force_use_mono=use_mono,
    )
    val_ds = KittiDepthShotDataset(
        root=root, split=split,
        include_pseudo=True if use_pseudo else False,
        include_mono=True if use_mono else False,
        depth_scale=depth_scale, pseudo_scale=pseudo_scale, mono_scale=mono_scale,
        transform=val_tf, items=val_items,
        force_use_pseudo=use_pseudo, force_use_mono=use_mono,
    )

    # 4) loaders
    def _worker_init_fn(worker_id: int):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_pad_to_max, worker_init_fn=_worker_init_fn)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_pad_to_max, worker_init_fn=_worker_init_fn)

    train_pct = int(round((1.0 - val_ratio) * 100))
    print(f"[Split] total={n} → train={len(train_ds)} val={len(val_ds)} "
          f"(policy: {train_pct}:{int(val_ratio*100)})")
    return train_ds, val_ds, train_loader, val_loader

# Quick check
if __name__ == "__main__":
    root = os.environ.get("SHOT_ROOT", "")
    if not root:
        print("Set SHOT_ROOT=/media/vip/T31/datasets/Sample/1shot/S0 (or 10shot/100shot)")
        exit(0)
    train_ds, val_ds, train_dl, val_dl = make_train_val_loaders(
        shot_root=root, split="train", batch_size=2, num_workers=2,
        resize_hw=None,  # e.g., (352,1216) if your net expects fixed size
        mono_is_normalized=True, normalize_rgb=False, align_to_rgb_if_no_resize=True
    )
    b = next(iter(train_dl))
    print("Train batch keys:", list(b.keys()))
