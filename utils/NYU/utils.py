# utils/NYU/utils.py
import os
from typing import Dict

import numpy as np
import torch

from utils.io_utils import save_jet  # 프로젝트 내부 util
from utils.NYU.dataset_utils import _h5_read_record, _center_crop_tensor, save_depth_png16_with_scale

def dump_oneshot_gt_images(splits: Dict[str, list], cfg: Dict, dmax: float):
    """
    train split에 뽑힌 샷(들)의 GT를 16-bit PNG(+jet)로 저장
    저장 경로: <save_dir>/oneshot_dump/<id>_gt16.png, <id>_gt_jet.png
    """
    ks  = cfg.get("kshot", {})
    nyu = cfg["nyu"]

    save_dir = ks.get("save_dir", "runs_nyu_1shot")
    out_dir  = os.path.join(save_dir, "oneshot_dump")
    os.makedirs(out_dir, exist_ok=True)

    crop_h = nyu.get("crop_h", 228)
    crop_w = nyu.get("crop_w", 304)
    do_crop = (crop_h is not None) and (crop_w is not None)
    scale_mm = float(cfg.get("dump", {}).get("scale_mm", 1000.0))
    sparse_scale = nyu.get("sparse_scale", None)

    for ent in splits["train"]:
        rid = ent.get("id", f"{ent['idx']:05d}")
        _, gt_hw, _, _ = _h5_read_record(ent["h5"], ent["idx"], dmax, sparse_scale)
        GT = torch.from_numpy(gt_hw).unsqueeze(0)
        if do_crop:
            GT = _center_crop_tensor(GT, int(crop_h), int(crop_w))
        depth_np = GT.squeeze(0).cpu().numpy()

        png_path = os.path.join(out_dir, f"{rid}_gt16.png")
        save_depth_png16_with_scale(png_path, depth_np, scale_mm=scale_mm)

        jet_path = os.path.join(out_dir, f"{rid}_gt_jet.png")
        save_jet(jet_path, GT.unsqueeze(0), dmax=dmax)


# 선택적으로 외부에서 import 하기 편하도록 export
__all__ = [
    "_as_1ch4", "_as_chw4", "_resize_like",
    "_center_crop_tensor", "_center_crop_slices",
    "_read_float_or_u16_to_meters", "save_depth_png16_with_scale",
    "_h5_read_record", "_h5_count_records", "_h5_ids_or_index",
    "dump_oneshot_gt_images",
]
