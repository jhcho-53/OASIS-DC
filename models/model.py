from .architectures.oasis import OASIS_DC

# Backward compatibility alias
MCPropNet = OASIS_DC
__all__ = ['OASIS_DC', 'MCPropNet']