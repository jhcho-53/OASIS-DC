# Compatibility layer - imports from new refactored structure

from .datasets.base import KShotDataset, build_splits_from_cfg

# Re-export for backward compatibility
__all__ = ['KShotDataset', 'build_splits_from_cfg']