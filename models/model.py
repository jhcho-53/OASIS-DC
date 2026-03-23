# Compatibility layer for refactored models
from .architectures.oasis import OASIS_DC

# Re-export for backward compatibility and new naming
MCPropNet = OASIS_DC  # Backward compatibility alias
__all__ = ['OASIS_DC', 'MCPropNet']