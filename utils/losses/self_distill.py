import torch
from .depth_losses import scale_shift_align

def _crop_tensor(x, y0, y1, x0, x1):
    return x[..., y0:y1, x0:x1]

def _paste_tensor(dst, src, y0, y1, x0, x1):
    dst[..., y0:y1, x0:x1] = src
    return dst

@torch.no_grad()
def build_edge_representation_GS(model, batch, S=3, overlap=32):
    model.eval()
    image = batch['image']
    sparse = batch['sparse']
    estimation = batch['estimation']
    pseudo = batch['pseudo']
    H, W = image.shape[-2:]

    out0 = model(image, sparse, estimation, pseudo)
    D0, G0, Om = out0['depth'], out0['edge'], out0['omega']
    GS = G0.clone()

    for s in range(1, S+1):
        grid = s + 1
        h = H // grid
        w = W // grid
        stride_y = max(1, h - overlap)
        stride_x = max(1, w - overlap)
        for y0 in range(0, H - h + 1, stride_y):
            for x0 in range(0, W - w + 1, stride_x):
                y1 = min(y0 + h, H)
                x1 = min(x0 + w, W)
                b_crop = {
                    'image'     : _crop_tensor(image,     y0,y1,x0,x1),
                    'sparse'    : _crop_tensor(sparse,    y0,y1,x0,x1),
                    'estimation': _crop_tensor(estimation,y0,y1,x0,x1),
                    'pseudo'    : _crop_tensor(pseudo,    y0,y1,x0,x1)
                }
                outw = model(b_crop['image'], b_crop['sparse'], b_crop['estimation'], b_crop['pseudo'])
                Dw, Gw = outw['depth'], outw['edge']
                GS_prev_w = _crop_tensor(GS, y0,y1,x0,x1)
                Gw_aligned, _, _ = scale_shift_align(Gw, GS_prev_w, mask=torch.ones_like(GS_prev_w))
                GS = _paste_tensor(GS, Gw_aligned, y0,y1,x0,x1)
    model.train()
    return GS, D0, G0, Om