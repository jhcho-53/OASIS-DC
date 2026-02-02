# sddr_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_shift_align(x, y, mask):
    """
    Solve: min_{a,b} || a*x + b - y ||^2 over mask
    Closed-form: a = cov(x,y)/var(x), b = mean(y) - a*mean(x)
    Returns aligned x' and (a,b)
    """
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
    """
    L_gt: affinity-invariant MSE (참고: 본문 L_gt 설명)  :contentReference[oaicite:6]{index=6}
    """
    pred_aligned, _, _ = scale_shift_align(pred_depth, gt_depth, valid_mask)
    diff = (pred_aligned - gt_depth) * valid_mask
    mse = (diff.square().sum(dim=[1,2,3]) / torch.clamp(valid_mask.sum(dim=[1,2,3]), min=1.)).mean()
    return mse


def kmeans_masks_from_edges(GS, top_ratio=0.05, k=4, iters=10):
    """
    Fig.10: 상위 5% 에지 픽셀을 이진화 후 k-means로 몇 개의 에지-밀집 영역 P_n을 만든다.  :contentReference[oaicite:7]{index=7}
    반환: [B, k, H, W] 마스크
    """
    B, _, H, W = GS.shape
    flat = GS.view(B, -1)
    thresh = torch.quantile(flat, 1.0 - top_ratio, dim=1, keepdim=True)
    # binary mask for top edges
    bin_mask = (flat >= thresh).float()  # B x (HW)
    # coords of selected points
    ys, xs = torch.meshgrid(torch.arange(H, device=GS.device), torch.arange(W, device=GS.device), indexing='ij')
    coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1).float()  # (HW,2)
    coords = coords.unsqueeze(0).expand(B, -1, -1)  # B x (HW) x 2

    masks = []
    for b in range(B):
        sel = bin_mask[b] > 0
        pts = coords[b][sel]               # N x 2
        if pts.numel() == 0:
            # fallback: whole map as one region
            Ms = torch.zeros(k, H, W, device=GS.device)
            Ms[0] = 1.0
            masks.append(Ms)
            continue
        # init centers by random choose
        N = pts.shape[0]
        idx = torch.randperm(N, device=GS.device)[:k]
        centers = pts[idx].clone()         # k x 2
        # k-means iterations
        for _ in range(iters):
            d2 = (pts[:,None,:] - centers[None,:,:]).pow(2).sum(-1)  # N x k
            assign = d2.argmin(dim=1)                                # N
            new_centers = torch.stack([pts[assign==i].mean(dim=0) if (assign==i).any() else centers[i] for i in range(k)], dim=0)
            if torch.allclose(new_centers, centers, atol=1e-3): break
            centers = new_centers
        # rasterize to masks
        d2_full = (coords[b][:,None,:] - centers[None,:,:]).pow(2).sum(-1)  # (HW) x k
        region = d2_full.argmin(dim=1)                                      # (HW)
        Ms = torch.stack([(region==i).float().view(H, W) for i in range(k)], dim=0)  # k x H x W
        # keep only regions intersecting edge mask
        Ms = Ms * bin_mask[b].view(1, H, W)
        # normalize by area to avoid tiny regions dominating
        area = Ms.view(k,-1).sum(-1).clamp_min(1.0).view(k,1,1)
        Ms = Ms / area
        masks.append(Ms)
    masks = torch.stack(masks, dim=0)  # B x k x H x W
    return masks


def edge_guided_gradient_loss(G0, GS, P_masks):
    """
    Eq.(7): L_grad = (1/N_g) * sum_n || (β1*G0[P_n] + β0) - GS[P_n] ||_1  (영역별 스케일·시프트 정렬)  :contentReference[oaicite:8]{index=8}
    """
    B, _, H, W = G0.shape
    Ng = P_masks.shape[1]
    loss = 0.0
    for n in range(Ng):
        P = P_masks[:, n:n+1]             # Bx1xHxW
        # align
        G0a, _, _ = scale_shift_align(G0, GS, P)
        diff = (G0a - GS) * P
        # L1 over valid region
        l1 = diff.abs().sum(dim=[1,2,3]) / torch.clamp(P.sum(dim=[1,2,3]), min=1.0)
        loss = loss + l1.mean()
    return loss / Ng


def fusion_quantile_loss(omega, GS, a=0.02, Nw=4):
    """
    Eq.(9): 분위 샘플링으로 GS와 Ω의 분포 정렬(힌지형 페널티)  :contentReference[oaicite:9]{index=9}
    """
    B, _, H, W = GS.shape
    loss = 0.0
    for b in range(B):
        g = GS[b].flatten()
        o = omega[b].flatten()
        # lower/upper quantiles of GS
        for n in range(1, Nw+1):
            ta = torch.quantile(g, a*n)            # GS lower quantile
            t1a = torch.quantile(g, 1.0 - a*n)     # GS upper quantile
            Ta = torch.quantile(o, a*n)            # Ω lower
            T1a = torch.quantile(o, 1.0 - a*n)     # Ω upper
            # flat areas: GS < ta → want Ω < Ta
            idx_low = (g < ta)
            if idx_low.any():
                loss += torch.clamp(o[idx_low] - Ta, min=0).mean()
            # edge areas: GS > t1a → want Ω > T1a
            idx_high = (g > t1a)
            if idx_high.any():
                loss += torch.clamp(T1a - o[idx_high], min=0).mean()
    return loss / B

# self_distill.py
import torch
import torch.nn.functional as F


def _crop_tensor(x, y0, y1, x0, x1):
    return x[..., y0:y1, x0:x1]


def _paste_tensor(dst, src, y0, y1, x0, x1):
    dst[..., y0:y1, x0:x1] = src
    return dst


@torch.no_grad()
def build_edge_representation_GS(model, batch, S=3, overlap=32):
    """
    Coarse-to-fine self-distillation으로 GS 생성 (Fig.3, Eq.3/5/6).  :contentReference[oaicite:11]{index=11}
    batch: dict with tensors {image,sparse,estimation,pseudo}  (BxCxHxW)
    returns: GS (Bx1xHxW), D0 (Bx1xHxW), G0 (Bx1xHxW), Omega (Bx1xHxW)
    """
    model.eval()
    image = batch['image']; sparse = batch['sparse']
    estimation = batch['estimation']; pseudo = batch['pseudo']
    H, W = image.shape[-2:]

    # s=0: initial
    out0 = model(image, sparse, estimation, pseudo)
    D0, G0, Om = out0['depth'], out0['edge'], out0['omega']
    GS = G0.clone()  # 초기 에지 표현

    # s=1..S: window refinement
    for s in range(1, S+1):
        grid = s + 1
        h = H // grid
        w = W // grid
        # 윈도우 겹침
        stride_y = max(1, h - overlap)
        stride_x = max(1, w - overlap)
        # 윈도우 커버
        for y0 in range(0, H - h + 1, stride_y):
            for x0 in range(0, W - w + 1, stride_x):
                y1 = min(y0 + h, H)
                x1 = min(x0 + w, W)
                # crop inputs
                b_crop = {
                    'image'     : _crop_tensor(image,     y0,y1,x0,x1),
                    'sparse'    : _crop_tensor(sparse,    y0,y1,x0,x1),
                    'estimation': _crop_tensor(estimation,y0,y1,x0,x1),
                    'pseudo'    : _crop_tensor(pseudo,    y0,y1,x0,x1)
                }
                # forward on window
                outw = model(b_crop['image'], b_crop['sparse'], b_crop['estimation'], b_crop['pseudo'])
                Dw, Gw = outw['depth'], outw['edge']
                # align ∇Dw to previous GS window (Eq.6)
                GS_prev_w = _crop_tensor(GS, y0,y1,x0,x1)
                Gw_aligned, _, _ = scale_shift_align(Gw, GS_prev_w, mask=torch.ones_like(GS_prev_w))
                # paste back
                GS = _paste_tensor(GS, Gw_aligned, y0,y1,x0,x1)
    model.train()
    return GS, D0, G0, Om
