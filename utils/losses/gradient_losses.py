import torch
from .depth_losses import scale_shift_align

def kmeans_masks_from_edges(GS, top_ratio=0.05, k=4, iters=10):
    B, _, H, W = GS.shape
    flat = GS.view(B, -1)
    thresh = torch.quantile(flat, 1.0 - top_ratio, dim=1, keepdim=True)
    bin_mask = (flat >= thresh).float()
    ys, xs = torch.meshgrid(torch.arange(H, device=GS.device), torch.arange(W, device=GS.device), indexing='ij')
    coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1).float()
    coords = coords.unsqueeze(0).expand(B, -1, -1)

    masks = []
    for b in range(B):
        sel = bin_mask[b] > 0
        pts = coords[b][sel]
        if pts.numel() == 0:
            Ms = torch.zeros(k, H, W, device=GS.device)
            Ms[0] = 1.0
            masks.append(Ms)
            continue
        N = pts.shape[0]
        idx = torch.randperm(N, device=GS.device)[:k]
        centers = pts[idx].clone()
        for _ in range(iters):
            d2 = (pts[:,None,:] - centers[None,:,:]).pow(2).sum(-1)
            assign = d2.argmin(dim=1)
            new_centers = torch.stack([pts[assign==i].mean(dim=0) if (assign==i).any() else centers[i] for i in range(k)], dim=0)
            if torch.allclose(new_centers, centers, atol=1e-3): 
                break
            centers = new_centers
        d2_full = (coords[b][:,None,:] - centers[None,:,:]).pow(2).sum(-1)
        region = d2_full.argmin(dim=1)
        Ms = torch.stack([(region==i).float().view(H, W) for i in range(k)], dim=0)
        Ms = Ms * bin_mask[b].view(1, H, W)
        area = Ms.view(k,-1).sum(-1).clamp_min(1.0).view(k,1,1)
        Ms = Ms / area
        masks.append(Ms)
    masks = torch.stack(masks, dim=0)
    return masks

def edge_guided_gradient_loss(G0, GS, P_masks):
    B, _, H, W = G0.shape
    Ng = P_masks.shape[1]
    loss = 0.0
    for n in range(Ng):
        P = P_masks[:, n:n+1]
        G0a, _, _ = scale_shift_align(G0, GS, P)
        diff = (G0a - GS) * P
        l1 = diff.abs().sum(dim=[1,2,3]) / torch.clamp(P.sum(dim=[1,2,3]), min=1.0)
        loss = loss + l1.mean()
    return loss / Ng