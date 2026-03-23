# Compatibility layer for affinity modules
from .affinity import HCLApproxAffinity, EllipticAffinity

# Backward compatibility exports
__all__ = ['HCLApproxAffinity', 'EllipticAffinity']