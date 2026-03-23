"""
Common I/O utilities for loading images and depth maps.
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def read_rgb(path: str) -> np.ndarray:
    """
    Read RGB image
    
    Args:
        path: Path to image file
        
    Returns:
        RGB array of shape (H,W,3) with dtype uint8
    """
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def read_depth(path: str, scale_factor: float = 1.0) -> np.ndarray:
    """
    Read depth map from various formats
    
    Args:
        path: Path to depth file (.png, .npy, .npz, .tif)
        scale_factor: Scale factor to apply (e.g., 1000.0 for mm->m conversion)
        
    Returns:
        Depth array of shape (H,W) with dtype float32
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".npy":
        return np.load(path).astype(np.float32) / scale_factor
    
    if ext == ".npz":
        data = np.load(path)
        # Try common keys for depth data
        for key in ["arr_0", "depth", "D", "d"]:
            if key in data:
                return data[key].astype(np.float32) / scale_factor
        # Fallback to first array
        return list(data.values())[0].astype(np.float32) / scale_factor
    
    # Handle image formats
    with Image.open(path) as img:
        if img.mode in ["I;16", "I;16B", "I"]:
            # 16-bit PNG (common for KITTI)
            arr = np.array(img, dtype=np.uint16).astype(np.float32)
            return arr / scale_factor
        else:
            # 8-bit or other formats
            return np.array(img, dtype=np.uint8).astype(np.float32) / scale_factor


def read_kitti_depth(path: str) -> np.ndarray:
    """Read KITTI depth (16-bit PNG in mm, convert to meters)"""
    return read_depth(path, scale_factor=1000.0)


def read_nyu_depth(path: str) -> np.ndarray:
    """Read NYU depth (16-bit PNG in mm, convert to meters)"""  
    return read_depth(path, scale_factor=1000.0)


def resize_like(arr: np.ndarray, target_shape: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """
    Resize array to match target shape
    
    Args:
        arr: Input array of shape (H,W) or (H,W,C)
        target_shape: Target (H,W)
        is_mask: Whether this is a mask (use nearest neighbor)
        
    Returns:
        Resized array
    """
    target_h, target_w = target_shape
    
    if arr.shape[:2] == (target_h, target_w):
        return arr
    
    # Convert to PIL Image
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.float32))
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8) if arr.dtype == np.uint8 else arr.astype(np.float32))
    
    # Choose resampling method
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    
    # Resize and convert back
    resized = pil_img.resize((target_w, target_h), resample)
    return np.array(resized, dtype=arr.dtype)


def minmax_normalize(arr: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """
    Min-max normalize array to [0,1]
    
    Args:
        arr: Input array
        mask: Optional mask for computing min/max only on valid regions
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized array in [0,1]
    """
    if mask is not None and mask.any():
        valid_vals = arr[mask > 0]
        vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))
    else:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
    
    if vmax - vmin < eps:
        return np.zeros_like(arr, dtype=np.float32)
    
    normalized = (arr - vmin) / (vmax - vmin + eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def orient_by_correlation(E_norm: np.ndarray, depth_sparse: np.ndarray, mask_sparse: np.ndarray) -> np.ndarray:
    """
    Orient E_norm based on correlation with sparse depth
    Flips E_norm if negative correlation is detected
    
    Args:
        E_norm: Normalized relative depth estimate [0,1]
        depth_sparse: Sparse depth values
        mask_sparse: Sparse depth mask
        
    Returns:
        Oriented E_norm
    """
    sparse_indices = np.where(mask_sparse > 0)
    if len(sparse_indices[0]) < 5:
        return E_norm
    
    e_vals = E_norm[sparse_indices].astype(np.float64)
    d_vals = depth_sparse[sparse_indices].astype(np.float64)
    
    # Standardize
    e_vals = (e_vals - e_vals.mean()) / (e_vals.std() + 1e-9)
    d_vals = (d_vals - d_vals.mean()) / (d_vals.std() + 1e-9)
    
    correlation = float(np.mean(e_vals * d_vals))
    
    # Flip if negative correlation
    if correlation < -0.1:
        return 1.0 - E_norm
    else:
        return E_norm