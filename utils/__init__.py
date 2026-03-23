# Refactored utils package with modular structure

# Core utilities
from .core import AverageMeter, rmse, _as_chw4, _as_1ch4, _resize_like

# Training utilities  
from .training import save_ckpt, build_cfg_from_args, default_iters_for_shots

# Dataset utilities
from .datasets import KShotDataset, build_splits_from_cfg
from .datasets.nyu import read_h5_rgb_depth, center_crop
from .datasets.kitti import load_calibs_by_date, parse_date_from_path

# Depth processing
from .depth import poisson_complete, poisson_gpu, _sample_sparse, _make_sparse_from_gt

# Visualization
from .visualization import _depth_to_jet_rgb, save_depth_color, save_depth_bundle

# Losses
from .losses import (
    lgt_affinity_invariant, scale_shift_align,
    edge_guided_gradient_loss, kmeans_masks_from_edges,
    build_edge_representation_GS
)

__all__ = [
    # Core
    'AverageMeter', 'rmse', '_as_chw4', '_as_1ch4', '_resize_like',
    
    # Training
    'save_ckpt', 'build_cfg_from_args', 'default_iters_for_shots',
    
    # Datasets
    'KShotDataset', 'build_splits_from_cfg',
    'read_h5_rgb_depth', 'center_crop',
    'load_calibs_by_date', 'parse_date_from_path',
    
    # Depth
    'poisson_complete', 'poisson_gpu', '_sample_sparse', '_make_sparse_from_gt',
    
    # Visualization
    '_depth_to_jet_rgb', 'save_depth_color', 'save_depth_bundle',
    
    # Losses
    'lgt_affinity_invariant', 'scale_shift_align',
    'edge_guided_gradient_loss', 'kmeans_masks_from_edges',
    'build_edge_representation_GS'
]