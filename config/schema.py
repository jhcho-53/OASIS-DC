# config/schema.py
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TrainCfg:
    epochs: int
    lr: float
    save_dir: str
    tag: str

@dataclass
class LossW:
    mu_scaleinv: float
    w_lidar: float
    w_anchor_reg: float = 0.0

@dataclass
class OneShotPaths:
    rgb: str
    sparse: str
    pseudo: str
    estim: str
    gt: Optional[str] = None

@dataclass
class MCPropCfg:
    dmax: float
    steps: int
    kernels: Tuple[int,...]
    use_residual: bool
    use_sparse: bool
    # --- anchor options ---
    anchor_alpha: float               # initial α (used if not learnable)
    anchor_learnable: bool            # if True, learn α
    anchor_mode: str                  # "scalar" | "map"
    # curvature
    kappa_min: float
    kappa_max: float
    geometry: str                     # "hyperbolic" | "elliptic"
