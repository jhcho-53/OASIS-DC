# Compatibility layer - imports from new refactored structure
from .base import TinyFeat, ResidualHead, CurvatureGen, KernelGate, AnchorHead
from .utils import normalize_affinity_list, unfold_neighbors
from .poisson import poisson_gpu

# Re-export for backward compatibility
__all__ = [
    'TinyFeat', 'ResidualHead', 'CurvatureGen', 'KernelGate', 'AnchorHead',
    'normalize_affinity_list', 'unfold_neighbors', 'poisson_gpu'
]