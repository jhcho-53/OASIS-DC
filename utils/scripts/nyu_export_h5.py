import os
import glob
import argparse
from pathlib import Path

import h5py
import numpy as np
import cv2
from tqdm import tqdm

# ---------- H5에서 RGB/Depth 읽기 ----------
IMAGE_KEYS = ["image", "rgb", "img", "color", "images"]
DEPTH_KEYS = ["depth", "depths", "rawDepth", "rawDepths", "gt", "D"]

def read_h5_rgb_depth(h5_path: Path):
    """NYUv2 .h5에서 RGB(HxWx3 uint8)와 Depth(HxW float32, meter)를 반환."""
    rgb = None
    depth = None
    with h5py.File(str(h5_path), "r") as f:
        # depth 먼저
        for k in DEPTH_KEYS:
            if k in f:
                depth = f[k][...]
                break
        if depth is None:
            # fallback: 2D float 데이터셋
            for k in f.keys():
                arr = f[k][...]
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
                    depth = arr
                    break
        if depth is None:
            raise KeyError(f"No depth found in {h5_path}. Keys={list(f.keys())}")

        # rgb (있을 수도/없을 수도 있음)
        for k in IMAGE_KEYS:
            if k in f:
                img = f[k][...]
                rgb = img
                break

    depth = depth.astype(np.float32)  # meter

    if rgb is not None:
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))  # (3,H,W) -> (H,W,3)
        if rgb.dtype != np.uint8:
            scale = 255.0 if np.max(rgb) <= 1.0 else 1.0
            rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return rgb, depth

# ---------- 전처리(선택): 리사이즈/센터크롭 ----------
def center_crop(arr, w, h):
    H, W = arr.shape[:2]
    x0 = max(0, (W - w) // 2); y0 = max(0, (H - h) // 2)
    return arr[y0:y0+h, x0:x0+w]

def apply_resize_crop(rgb, depth, resize_wh=None, crop_wh=None):
    if resize_wh is not None:
        w, h = resize_wh
        if rgb is not None:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)  # 값 보존
    if crop_wh is not None:
        w, h = crop_wh
        if rgb is not None:
            rgb = center_crop(rgb, w, h)
        depth = center_crop(depth, w, h)
    return rgb, depth

# ---------- 저장 유틸 ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def depth_to_mm16(depth_m, max_depth=None):
    """
    meter -> millimeter(uint16). 유효(POSITIVE)만 변환, 나머지 0.
    max_depth>0이면 그 값(m)으로 상한 클리핑.
    """
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    if max_depth is not None and max_depth > 0:
        d = np.clip(d, 0.0, max_depth)
    d_mm = (d * 1000.0).round()
    d_mm = np.clip(d_mm, 0, 65535).astype(np.uint16)  # 65.535m까지 표현 가능
    return d_mm

def depth_to_jet(depth_m, max_depth=None):
    """
    Jet 컬러맵 시각화(BGR). 유효(>0, finite) 픽셀의 min~max(또는 max_depth)로 정규화.
    """
    d = depth_m.astype(np.float32)
    valid = (d > 0) & np.isfinite(d)
    if max_depth is not None and max_depth > 0:
        valid &= (d <= max_depth)

    if np.any(valid):
        vmin = float(np.min(d[valid]))
        vmax = float(np.max(d[valid]))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax - vmin < 1e-8:
        vmax = vmin + 1.0

    d_norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    d8 = (d_norm * 255.0).astype(np.uint8)
    jet = cv2.applyColorMap(d8, cv2.COLORMAP_JET)  # BGR
    return jet

# ---------- 파일 목록 ----------
def list_h5(root: Path):
    pats = [str(root / "train" / "**" / "*.h5"),
            str(root / "val"   / "**" / "*.h5")]
    out = []
    for p in pats:
        out.extend(glob.glob(p, recursive=True))
    return [Path(f) for f in sorted(out)]

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser("Extract RGB & Depth from NYUv2 .h5 to PNGs")
    ap.add_argument("--nyu-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W","H"))
    ap.add_argument("--center-crop", nargs=2, type=int, default=None, metavar=("W","H"))
    ap.add_argument("--max-depth", type=float, default=10.0, help="시각화/클리핑 상한(m). 음수면 무제한")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    nyu_root = Path(args.nyu_root).resolve()
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    max_depth = None if (args.max_depth is None or args.max_depth < 0) else args.max_depth
    files = list_h5(nyu_root)
    if not files:
        print(f"[ERROR] No .h5 found under {nyu_root}")
        return

    for h5_path in tqdm(files, desc="Export RGB/Depth"):
        try:
            rel = h5_path.relative_to(nyu_root)           # train/scene/.../00001.h5
            stem = rel.with_suffix("").name               # 00001
            out_dir = out_root / rel.parent / stem        # .../train/scene/.../00001/
            rgb_path = out_dir / "rgb.png"
            mm16_path = out_dir / "depth_mm16.png"
            jet_path = out_dir / "depth_jet.png"

            if args.skip_existing and rgb_path.exists() and mm16_path.exists() and jet_path.exists():
                continue

            rgb, depth = read_h5_rgb_depth(h5_path)
            rgb, depth = apply_resize_crop(
                rgb, depth,
                resize_wh=tuple(args.resize) if args.resize else None,
                crop_wh=tuple(args.center_crop) if args.center_crop else None
            )

            # 저장
            ensure_dir(out_dir)
            if rgb is not None:
                # OpenCV는 BGR, 파일명은 rgb.png
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(rgb_path), bgr)

            mm16 = depth_to_mm16(depth, max_depth=max_depth)
            cv2.imwrite(str(mm16_path), mm16)

            jet = depth_to_jet(depth, max_depth=max_depth)
            cv2.imwrite(str(jet_path), jet)

        except Exception as e:
            print(f"[ERROR] {h5_path}: {e}")

    print("[DONE] Export complete.")

if __name__ == "__main__":
    main()
