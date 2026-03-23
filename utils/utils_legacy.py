# Compatibility layer - imports from new refactored structure

# Core utilities
from .core.metrics import AverageMeter, rmse
from .core.tensor_ops import _as_chw4, _as_1ch4, _resize_like
from .core.io import save_checkpoint, depth_to_uint16_mm, save_depth_png16

# Training utilities
from .training.checkpoints import save_ckpt
from .training.config import build_cfg_from_args, default_iters_for_shots

# Depth processing
from .depth.poisson import poisson_complete, poisson_gpu
from .depth.sampling import _sample_sparse, _make_sparse_from_gt

# Visualization
from .visualization.depth_viz import _depth_to_jet_rgb, save_depth_color, save_depth_bundle

# Re-export for backward compatibility
__all__ = [
    # Core
    'AverageMeter', 'rmse', '_as_chw4', '_as_1ch4', '_resize_like',
    'save_checkpoint', 'depth_to_uint16_mm', 'save_depth_png16',
    
    # Training
    'save_ckpt', 'build_cfg_from_args', 'default_iters_for_shots',
    
    # Depth
    'poisson_complete', 'poisson_gpu', '_sample_sparse', '_make_sparse_from_gt',
    
    # Visualization
    '_depth_to_jet_rgb', 'save_depth_color', 'save_depth_bundle'
]