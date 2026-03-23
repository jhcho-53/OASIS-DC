# Compatibility layer - imports from new refactored structure

from .losses.depth_losses import scale_shift_align, lgt_affinity_invariant, fusion_quantile_loss
from .losses.gradient_losses import kmeans_masks_from_edges, edge_guided_gradient_loss
from .losses.self_distill import build_edge_representation_GS

# Re-export for backward compatibility
__all__ = [
    'scale_shift_align', 'lgt_affinity_invariant', 'fusion_quantile_loss',
    'kmeans_masks_from_edges', 'edge_guided_gradient_loss',
    'build_edge_representation_GS'
]