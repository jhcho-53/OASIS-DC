"""
Affinity computation utilities.
"""
import torch
from typing import List


def normalize_affinity_list(affinity_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalize affinity maps for stability.
    Apply tanh activation and L1 normalization for signed CSPN style.
    
    Args:
        affinity_list: List of affinity tensors (B, k*k, H, W)
        
    Returns:
        List of normalized affinity tensors
    """
    normalized = []
    for A in affinity_list:
        A_tanh = torch.tanh(A)  # Bounded activation
        A_norm = A_tanh / (torch.sum(torch.abs(A_tanh), dim=1, keepdim=True) + 1e-8)
        normalized.append(A_norm)
    return normalized


def apply_affinity_propagation(depth, affinity_list, kernels, steps=1):
    """
    Apply affinity-based propagation to depth maps.
    
    Args:
        depth: Input depth map (B, 1, H, W)
        affinity_list: List of normalized affinity tensors
        kernels: List of kernel sizes
        steps: Number of propagation steps
        
    Returns:
        Refined depth map
    """
    current_depth = depth
    
    for step in range(steps):
        # Weighted combination of propagations from different kernel sizes
        propagated_depths = []
        
        for A, k in zip(affinity_list, kernels):
            # Unfold depth for neighbor aggregation
            import torch.nn.functional as F
            depth_patches = F.unfold(current_depth, kernel_size=k, padding=k//2)
            B, _, HW = depth_patches.shape
            H, W = current_depth.shape[-2:]
            depth_patches = depth_patches.view(B, k*k, H, W)
            
            # Affinity-weighted aggregation
            propagated = (A * depth_patches).sum(dim=1, keepdim=True)
            propagated_depths.append(propagated)
        
        # Combine propagations (simple average for now)
        if propagated_depths:
            current_depth = sum(propagated_depths) / len(propagated_depths)
    
    return current_depth