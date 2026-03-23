"""
Poisson equation solver for depth completion.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F


def check_device(device: str = "cuda:0") -> torch.device:
    """Check and return valid device."""
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check driver/CUDA or use cpu device.")
    return torch.device(device)


@torch.no_grad()
def poisson_gpu(
    sparse_m: np.ndarray, 
    est_m: np.ndarray,
    tol: float = 1e-5, 
    maxiter: int = 1000,
    device: str = "cuda:0", 
    init: str = "est",
    clip_to_max_gt: bool = False
):
    """
    GPU-accelerated Poisson solver with Dirichlet boundary conditions.
    
    Solves: A·x = b where:
    - A: 5-point Laplacian operator (conv2d implementation)  
    - Dirichlet constraints: known = (sparse > 0) ∪ border
    - RHS: b = A@est - A@v_known (only in unknown regions)
    - CG solver with Jacobi preconditioning (diagonal = 4)
    
    Args:
        sparse_m: Sparse depth measurements (H, W)
        est_m: Initial depth estimates (H, W)
        tol: Convergence tolerance
        maxiter: Maximum CG iterations
        device: Compute device
        init: Initialization strategy ("est" or "zeros")
        clip_to_max_gt: Whether to clip result to max sparse depth
        
    Returns:
        Solved depth map (H, W)
    """
    dev = check_device(device)
    H, W = est_m.shape

    # Convert to tensors
    est = torch.from_numpy(est_m).to(dev, dtype=torch.float32)[None, None]     # (1,1,H,W)
    sparse = torch.from_numpy(sparse_m).to(dev, dtype=torch.float32)[None, None]

    # Define constraint masks
    mask_sparse = (sparse > 0)
    mask_border = torch.zeros_like(mask_sparse, dtype=torch.bool)
    mask_border[:, :, 0, :]  = True   # Top border
    mask_border[:, :, -1, :] = True   # Bottom border  
    mask_border[:, :, :, 0]  = True   # Left border
    mask_border[:, :, :, -1] = True   # Right border

    known_mask = mask_sparse | mask_border
    unknown_mask = ~known_mask

    # Set known values: sparse where available, otherwise est
    v_known = torch.where(mask_sparse, sparse, est) * known_mask.float()

    # Define 5-point Laplacian kernel
    K = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                     device=dev, dtype=torch.float32).view(1, 1, 3, 3)
    
    def lap(x): 
        return F.conv2d(x, K, padding=1)

    # Set up linear system: A·x = b
    b_full = lap(est)
    b_u = (b_full - lap(v_known)) * unknown_mask.float()

    # Initialize solution
    if init == "est":
        x = (est * unknown_mask.float()).clone()
    else:
        x = torch.zeros_like(est)

    def A_apply(xfull):
        """Apply system matrix A (only in unknown regions)"""
        xfull = xfull * unknown_mask.float()
        y = lap(xfull)
        return y * unknown_mask.float()

    # Jacobi preconditioner (diagonal = 4 for interior points)
    Minv = 0.25 * unknown_mask.float()

    # Timing setup
    use_events = (dev.type == "cuda")
    if use_events:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
    else:
        t_start = time.perf_counter()

    # Conjugate Gradient solver
    r = b_u - A_apply(x)
    z = Minv * r
    p = z.clone()
    rz_old = torch.sum(r * z)
    rhs_norm = torch.sqrt(torch.sum(b_u * b_u)) + 1e-12

    it = 0
    for it in range(1, maxiter + 1):
        Ap = A_apply(p)
        denom = torch.sum(p * Ap) + 1e-12
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        # Check convergence
        rel = torch.sqrt(torch.sum(r * r)) / rhs_norm
        if rel.item() < tol:
            break

        # Update search direction
        z = Minv * r
        rz_new = torch.sum(r * z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new

    # Timing
    if use_events:
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0
    else:
        elapsed = time.perf_counter() - t_start

    # Reconstruct full solution
    x_full = x + v_known
    
    # Optional clipping
    if clip_to_max_gt and sparse_m.max() > 0:
        x_full = torch.clamp(x_full, 0, sparse_m.max())
    
    out = x_full.squeeze().detach().cpu().numpy().astype(np.float32)
    
    return out, {"iterations": it, "elapsed": elapsed, "converged": rel.item() < tol}