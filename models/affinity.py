# Compatibility layer - imports from new refactored structure  
from .affinity import HCLApproxAffinity, EllipticAffinity

# Re-export for backward compatibility
__all__ = ['HCLApproxAffinity', 'EllipticAffinity']