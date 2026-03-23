"""
Affinity computation modules for depth completion.
"""
from .hyperbolic import HCLApproxAffinity
from .elliptic import EllipticAffinity
from .utils import normalize_affinity_list, apply_affinity_propagation

__all__ = [
    'HCLApproxAffinity',
    'EllipticAffinity', 
    'normalize_affinity_list',
    'apply_affinity_propagation'
]