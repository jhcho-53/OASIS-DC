"""
Utilities for Poisson equation solving in depth completion.
"""
import time
import numpy as np
import torch
from typing import Dict, Tuple, Optional

try:
    from models.module import poisson_gpu
    HAS_POISSON_GPU = True
except ImportError:
    poisson_gpu = None
    HAS_POISSON_GPU = False


def poisson_fallback_jacobi(sparse_depth: np.ndarray, 
                           est_depth: np.ndarray, 
                           mask: np.ndarray,
                           max_iterations: int = 400, 
                           tolerance: float = 1e-5) -> np.ndarray:
    """
    Fallback Jacobi solver for Poisson equation
    
    Args:
        sparse_depth: Sparse depth measurements
        est_depth: Initial depth estimate 
        mask: Boolean mask for sparse points
        max_iterations: Maximum Jacobi iterations
        tolerance: Convergence tolerance
        
    Returns:
        Completed depth map
    """
    # Initialize solution with estimate
    solution = est_depth.copy()
    
    # Set boundary conditions (Dirichlet at borders + sparse points)
    known_mask = mask.copy()
    known_mask[0, :] = known_mask[-1, :] = True  # Top/bottom borders
    known_mask[:, 0] = known_mask[:, -1] = True  # Left/right borders
    
    known_values = est_depth.copy()
    sparse_indices = mask & (sparse_depth > 0)
    known_values[sparse_indices] = sparse_depth[sparse_indices]
    
    # Jacobi iterations
    for iteration in range(max_iterations):
        solution_old = solution.copy()
        
        # Update interior points (5-point stencil)
        solution = 0.25 * (
            np.roll(solution, 1, axis=0) +  # up
            np.roll(solution, -1, axis=0) + # down  
            np.roll(solution, 1, axis=1) +  # left
            np.roll(solution, -1, axis=1)   # right
        )
        
        # Enforce boundary conditions
        solution[known_mask] = known_values[known_mask]
        
        # Check convergence
        if np.mean(np.abs(solution - solution_old)) < tolerance:
            break
    
    return solution.astype(np.float32)


def solve_poisson_with_stats(sparse_depth: torch.Tensor,
                            mask_sparse: torch.Tensor,
                            est_depth: torch.Tensor,
                            dmax: float,
                            tolerance: float = 1e-5,
                            max_iterations: int = 1000,
                            device: str = "cuda",
                            auto_flip: bool = True,
                            est_affine: bool = True,
                            smooth_est: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Solve Poisson equation with preprocessing and statistics
    
    Args:
        sparse_depth: (B,1,H,W) sparse depth measurements in meters
        mask_sparse: (B,1,H,W) boolean mask for sparse points
        est_depth: (B,1,H,W) relative depth estimate in [0,1] 
        dmax: Maximum depth scale
        tolerance: Solver tolerance
        max_iterations: Maximum solver iterations
        device: Computing device
        auto_flip: Auto-orient estimate by correlation
        est_affine: Apply robust affine alignment
        smooth_est: Apply mild smoothing to estimate
        
    Returns:
        Tuple of (completed_depth, statistics_dict)
    """
    assert sparse_depth.shape[0] == 1, "Currently only supports batch size 1"
    
    # Extract arrays (B=1)
    sparse_np = sparse_depth[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_np = (mask_sparse[0, 0].detach().cpu().numpy() > 0).astype(bool)
    est_np = est_depth[0, 0].detach().cpu().numpy().astype(np.float32)
    
    # Preprocessing
    if auto_flip and mask_np.sum() >= 10:
        # Orient estimate by correlation with sparse depth
        sparse_vals = (sparse_np / dmax)[mask_np]
        est_vals = est_np[mask_np]
        
        if len(sparse_vals) > 1 and len(est_vals) > 1:
            correlation = np.corrcoef(est_vals, sparse_vals)[0, 1]
            if np.isfinite(correlation) and correlation < 0.0:
                est_np = 1.0 - est_np
    
    if est_affine and mask_np.sum() >= 10:
        # Robust affine alignment using IRLS
        x = est_np[mask_np].reshape(-1, 1)
        y = (sparse_np[mask_np] / dmax).reshape(-1, 1)
        A = np.concatenate([x, np.ones_like(x)], axis=1).astype(np.float32)
        
        # IRLS iterations
        weights = np.ones((A.shape[0], 1), dtype=np.float32)
        for _ in range(3):
            A_weighted = A * weights
            y_weighted = y * weights
            try:
                theta, *_ = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
                residuals = A @ theta - y
                c = 1.345 * np.median(np.abs(residuals)) + 1e-6
                weights = 1.0 / np.maximum(1.0, np.abs(residuals) / c)
            except np.linalg.LinAlgError:
                break
        
        if 'theta' in locals():
            a, b = float(theta[0, 0]), float(theta[1, 0])
            est_np = np.clip(a * est_np + b, 0.0, 1.0)
    
    if smooth_est:
        # Mild 3x3 smoothing with reflection padding
        from torch.nn.functional import conv2d
        est_tensor = torch.from_numpy(est_np).unsqueeze(0).unsqueeze(0)
        est_padded = torch.nn.functional.pad(est_tensor, (1, 1, 1, 1), mode='reflect')
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
        est_smoothed = conv2d(est_padded, kernel).squeeze().numpy()
        est_np = est_smoothed.astype(np.float32)
    
    # Scale estimate to metric depth
    est_metric = (est_np * dmax).astype(np.float32)
    sparse_metric = sparse_np * mask_np.astype(np.float32)
    
    # Solve Poisson equation
    start_time = time.perf_counter()
    
    if HAS_POISSON_GPU and callable(poisson_gpu):
        try:
            result, solver_stats = poisson_gpu(
                sparse_m=sparse_metric,
                est_m=est_metric,
                tol=tolerance,
                maxiter=max_iterations,
                device=device,
                init="est",
                clip_to_max_gt=False
            )
            solve_time = time.perf_counter() - start_time
            stats = {
                "solver": "cg_gpu",
                "time_sec": float(solver_stats.get("time_sec", solve_time)),
                "iterations": int(solver_stats.get("cg_iters", -1)),
                "n_sparse": int(mask_np.sum())
            }
        except Exception as e:
            # Fallback to Jacobi
            result = poisson_fallback_jacobi(sparse_metric, est_metric, mask_np, max_iterations, tolerance)
            solve_time = time.perf_counter() - start_time
            stats = {
                "solver": "jacobi_fallback",
                "time_sec": solve_time,
                "iterations": -1,
                "n_sparse": int(mask_np.sum()),
                "gpu_error": str(e)
            }
    else:
        # Jacobi fallback
        result = poisson_fallback_jacobi(sparse_metric, est_metric, mask_np, max_iterations, tolerance)
        solve_time = time.perf_counter() - start_time
        stats = {
            "solver": "jacobi_cpu",
            "time_sec": solve_time,
            "iterations": -1,
            "n_sparse": int(mask_np.sum())
        }
    
    # Convert back to tensor
    result_tensor = torch.from_numpy(result).unsqueeze(0).unsqueeze(0)
    result_tensor = result_tensor.to(sparse_depth.device, dtype=torch.float32)
    
    return result_tensor, stats


def subsample_sparse_points(depth_map: np.ndarray, 
                           n_points: int, 
                           seed: int = 0, 
                           identifier: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample sparse points from depth map
    
    Args:
        depth_map: Full depth map
        n_points: Number of points to sample
        seed: Random seed
        identifier: Identifier for deterministic sampling
        
    Returns:
        Tuple of (sparse_depth, sparse_mask)
    """
    h, w = depth_map.shape
    valid_indices = np.where(depth_map > 0)
    
    sparse_depth = np.zeros_like(depth_map, dtype=np.float32)
    sparse_mask = np.zeros_like(depth_map, dtype=np.uint8)
    
    if len(valid_indices[0]) == 0:
        return sparse_depth, sparse_mask
    
    n_available = len(valid_indices[0])
    n_sample = min(n_points, n_available) if n_points > 0 else n_available
    
    if n_sample == n_available:
        sparse_depth[valid_indices] = depth_map[valid_indices]
        sparse_mask[valid_indices] = 1
    else:
        # Deterministic sampling
        import hashlib
        hash_seed = seed ^ (abs(hash(identifier)) & 0xffffffff)
        rng = np.random.default_rng(hash_seed)
        
        selected_indices = rng.choice(n_available, n_sample, replace=False)
        selected_y = valid_indices[0][selected_indices]
        selected_x = valid_indices[1][selected_indices]
        
        sparse_depth[selected_y, selected_x] = depth_map[selected_y, selected_x]
        sparse_mask[selected_y, selected_x] = 1
    
    return sparse_depth, sparse_mask