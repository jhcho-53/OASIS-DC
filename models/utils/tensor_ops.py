"""
Common tensor operations and utilities.
"""
import torch
import torch.nn.functional as F
from typing import List


def unfold_neighbors(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Unfold tensor to get neighbor patches."""
    return F.unfold(x, kernel_size=kernel_size, padding=kernel_size//2)


def normalize_affinity_list(affinity_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalize affinity maps for stability.
    Apply tanh activation and L1 normalization.
    """
    normalized = []
    for A in affinity_list:
        A_tanh = torch.tanh(A)  # Bounded activation
        A_norm = A_tanh / (torch.sum(torch.abs(A_tanh), dim=1, keepdim=True) + 1e-8)
        normalized.append(A_norm)
    return normalized


def compute_gradient_magnitude(x: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude using Sobel operators."""
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)