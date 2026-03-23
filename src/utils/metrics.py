"""
Common evaluation metrics for depth completion tasks.
"""
import torch
from typing import Dict, Any


@torch.no_grad()
def rmse(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    Root Mean Square Error
    
    Args:
        pred: (B,1,H,W) predicted depth
        gt: (B,1,H,W) ground truth depth
        valid_mask: (B,1,H,W) or (B,H,W) boolean mask
    
    Returns:
        RMSE value in meters
    """
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    
    valid = valid_mask > 0
    n_valid = valid.sum().item()
    
    if n_valid == 0:
        return float("nan")
    
    error = (pred - gt)[valid]
    mse = torch.mean(error ** 2)
    return float(torch.sqrt(mse).item())


@torch.no_grad() 
def mae(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    Mean Absolute Error
    
    Args:
        pred: (B,1,H,W) predicted depth
        gt: (B,1,H,W) ground truth depth
        valid_mask: (B,1,H,W) or (B,H,W) boolean mask
    
    Returns:
        MAE value in meters
    """
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    
    valid = valid_mask > 0
    n_valid = valid.sum().item()
    
    if n_valid == 0:
        return float("nan")
    
    error = torch.abs(pred - gt)[valid]
    return float(torch.mean(error).item())


@torch.no_grad()
def delta1(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 1.25) -> float:
    """
    Delta accuracy metric
    
    Args:
        pred: (B,1,H,W) predicted depth
        gt: (B,1,H,W) ground truth depth  
        valid_mask: (B,1,H,W) or (B,H,W) boolean mask
        threshold: threshold for delta computation (default: 1.25)
    
    Returns:
        Percentage of pixels with delta < threshold
    """
    eps = 1e-6
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    
    valid = valid_mask > 0
    
    pred_valid = torch.clamp(pred[valid], min=eps)
    gt_valid = torch.clamp(gt[valid], min=eps)
    
    if pred_valid.numel() == 0:
        return 0.0
    
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    return float((ratio < threshold).float().mean().item())


def compute_all_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute all standard depth metrics
    
    Args:
        pred: (B,1,H,W) predicted depth
        gt: (B,1,H,W) ground truth depth
        valid_mask: (B,1,H,W) or (B,H,W) boolean mask
    
    Returns:
        Dictionary with metric names and values
    """
    return {
        "rmse": rmse(pred, gt, valid_mask),
        "mae": mae(pred, gt, valid_mask), 
        "delta1": delta1(pred, gt, valid_mask)
    }


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, float]:
    """Format metrics to specified precision"""
    return {k: round(float(v), precision) for k, v in metrics.items()}