import torch
import numpy as np

def rmse(pred, target, mask):
    diff = (pred - target) * mask
    mse = (diff**2).sum(dim=[1,2,3]) / torch.clamp(mask.sum(dim=[1,2,3]), min=1.0)
    return torch.sqrt(mse).mean().item()

class AverageMeter:
    def __init__(self): 
        self.reset()
    
    def reset(self): 
        self.tot = 0.0
        self.cnt = 0
    
    def update(self, v, n=1): 
        self.tot += v * n
        self.cnt += n
    
    @property
    def avg(self): 
        return self.tot / max(self.cnt, 1)

def dc_metrics(pred, gt, max_depth=10.0):
    valid = (gt > 0) & np.isfinite(gt)
    if max_depth is not None:
        valid &= (gt <= max_depth)
    if not np.any(valid):
        return dict(RMSE=np.nan, MAE=np.nan, iRMSE=np.nan, iMAE=np.nan, REL=np.nan), (0,)*6

    p = pred[valid]
    g = gt[valid]
    e = p - g

    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))

    invp = 1.0 / np.clip(p, 1e-6, None)
    invg = 1.0 / np.clip(g, 1e-6, None)
    ie   = invp - invg
    irmse = float(np.sqrt(np.mean(ie**2)))
    imae  = float(np.mean(np.abs(ie)))

    rel = float(np.mean(np.abs(e) / g))

    se_sum   = float(np.sum(e**2))
    ae_sum   = float(np.sum(np.abs(e)))
    sie_sum  = float(np.sum(ie**2))
    aie_sum  = float(np.sum(np.abs(ie)))
    rel_sum  = float(np.sum(np.abs(e) / g))
    n_valid  = int(g.size)

    return dict(RMSE=rmse, MAE=mae, iRMSE=irmse, iMAE=imae, REL=rel), (se_sum, ae_sum, sie_sum, aie_sum, rel_sum, n_valid)

class PixelAverager:
    def __init__(self):
        self.se = 0.0
        self.ae = 0.0
        self.sie = 0.0
        self.aie = 0.0
        self.rel = 0.0
        self.n = 0
    
    def add(self, se_sum, ae_sum, sie_sum, aie_sum, rel_sum, n):
        self.se += se_sum
        self.ae += ae_sum
        self.sie += sie_sum
        self.aie += aie_sum
        self.rel += rel_sum
        self.n += n
    
    def mean(self):
        if self.n == 0:
            return dict(RMSE=np.nan, MAE=np.nan, iRMSE=np.nan, iMAE=np.nan, REL=np.nan)
        return dict(
            RMSE=float(np.sqrt(self.se / self.n)),
            MAE=float(self.ae / self.n),
            iRMSE=float(np.sqrt(self.sie / self.n)),
            iMAE=float(self.aie / self.n),
            REL=float(self.rel / self.n),
        )