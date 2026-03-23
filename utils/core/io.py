import os
import torch
import numpy as np
from PIL import Image

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path):
    return torch.load(path, map_location='cpu')

def depth_to_uint16_mm(depth_m: np.ndarray, max_m: float = 10.0) -> np.uint16:
    x = np.clip(depth_m, 0.0, max_m) * 1000.0
    return x.astype(np.uint16)

def save_depth_png16(path_png: str, depth_m: np.ndarray, max_m: float = 10.0):
    arr16 = depth_to_uint16_mm(depth_m, max_m)
    Image.fromarray(arr16).save(path_png)