import torch

def scale_shift_align(x, y, mask):
    eps = 1e-6
    m = (mask > 0).float()
    n = torch.clamp(m.sum(dim=[1,2,3], keepdim=True), min=1.0)
    x_mean = (x * m).sum(dim=[1,2,3], keepdim=True) / n
    y_mean = (y * m).sum(dim=[1,2,3], keepdim=True) / n
    xv = (x - x_mean) * m
    yv = (y - y_mean) * m
    a = (xv * yv).sum(dim=[1,2,3], keepdim=True) / (xv.square().sum(dim=[1,2,3], keepdim=True) + eps)
    b = y_mean - a * x_mean
    x_aligned = a * x + b
    return x_aligned, a, b

def lgt_affinity_invariant(pred_depth, gt_depth, valid_mask):
    pred_aligned, _, _ = scale_shift_align(pred_depth, gt_depth, valid_mask)
    diff = (pred_aligned - gt_depth) * valid_mask
    mse = (diff.square().sum(dim=[1,2,3]) / torch.clamp(valid_mask.sum(dim=[1,2,3]), min=1.)).mean()
    return mse

def fusion_quantile_loss(omega, GS, a=0.02, Nw=4):
    B, _, H, W = GS.shape
    loss = 0.0
    for b in range(B):
        g = GS[b].flatten()
        o = omega[b].flatten()
        for n in range(1, Nw+1):
            ta = torch.quantile(g, a*n)
            t1a = torch.quantile(g, 1.0 - a*n)
            Ta = torch.quantile(o, a*n)
            T1a = torch.quantile(o, 1.0 - a*n)
            idx_low = (g < ta)
            if idx_low.any():
                loss += torch.clamp(o[idx_low] - Ta, min=0).mean()
            idx_high = (g > t1a)
            if idx_high.any():
                loss += torch.clamp(T1a - o[idx_high], min=0).mean()
    return loss / B