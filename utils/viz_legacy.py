# Compatibility layer - imports from new refactored structure

from .visualization.depth_viz import _depth_to_jet_rgb, save_depth_color, save_depth_bundle

# Re-export for backward compatibility
__all__ = ['_depth_to_jet_rgb', 'save_depth_color', 'save_depth_bundle']