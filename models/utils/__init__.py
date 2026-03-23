"""
Utility functions for models.
"""
from .shape_utils import _as_chw4, _as_1ch4, _resize_like, _strip_extra_batch_dim
from .init_utils import init_weights, apply_weight_init, init_conv_weights, init_bn_weights
from .tensor_ops import unfold_neighbors, normalize_affinity_list, compute_gradient_magnitude

__all__ = [
    '_as_chw4', '_as_1ch4', '_resize_like', '_strip_extra_batch_dim',
    'init_weights', 'apply_weight_init', 'init_conv_weights', 'init_bn_weights',
    'unfold_neighbors', 'normalize_affinity_list', 'compute_gradient_magnitude'
]