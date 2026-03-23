from .depth_losses import lgt_affinity_invariant, scale_shift_align
from .gradient_losses import edge_guided_gradient_loss, kmeans_masks_from_edges
from .self_distill import build_edge_representation_GS

__all__ = [
    'lgt_affinity_invariant', 'scale_shift_align',
    'edge_guided_gradient_loss', 'kmeans_masks_from_edges',
    'build_edge_representation_GS'
]