# Main models package
from .architectures import OASIS_DC
from .base import TinyFeat, ResidualHead, CurvatureGen, KernelGate, AnchorHead
from .affinity import HCLApproxAffinity, EllipticAffinity, normalize_affinity_list
from .poisson import poisson_gpu, ScreenedPoissonLayer
from .utils import _as_chw4, _as_1ch4, _resize_like

__all__ = [
    # Main architecture
    'OASIS_DC',
    
    # Base components
    'TinyFeat', 'ResidualHead', 'CurvatureGen', 'KernelGate', 'AnchorHead',
    
    # Affinity modules
    'HCLApproxAffinity', 'EllipticAffinity', 'normalize_affinity_list',
    
    # Poisson solvers
    'poisson_gpu', 'ScreenedPoissonLayer',
    
    # Utilities
    '_as_chw4', '_as_1ch4', '_resize_like'
]