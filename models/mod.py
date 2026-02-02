import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet34
from models.module import poisson_gpu


# ---------- utils ----------
class SobelGrad(nn.Module):
    """Compute gradient magnitude |∇x| with fixed Sobel kernels."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class InputFusionStem(nn.Module):
    """
    Fuse multi-channel priors (image + depth priors + sparse mask) -> 3ch 'image-like' tensor.
    This lets us reuse a standard ImageNet encoder.
    """
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, k=3, s=1, p=1),
            ConvBNReLU(mid_ch, mid_ch, k=3, s=1, p=1),
            nn.Conv2d(mid_ch, 3, kernel_size=1, bias=False)
        )
    def forward(self, x):  # x: B×C×H×W
        return self.fuse(x)


# ---------- shared encoder (ResNet34) ----------
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet34(pretrained=pretrained)
        self.conv1 = backbone.conv1
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 1/4, ch=64
        self.layer2 = backbone.layer2  # 1/8, ch=128
        self.layer3 = backbone.layer3  # 1/16, ch=256
        self.layer4 = backbone.layer4  # 1/32, ch=512

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        c1 = self.layer1(self.maxpool(x))    # /4
        c2 = self.layer2(c1)                 # /8
        c3 = self.layer3(c2)                 # /16
        c4 = self.layer4(c3)                 # /32
        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}


# ---------- attention-based feature interaction (per scale) ----------
class AttnInteractor(nn.Module):
    """
    Approximate SDDR's attention-based feature interaction (p.16 Fig.12, Eq.(12) 유사).
    Concatenate tokens from L/H, apply MHSA, split back.
    """
    def __init__(self, in_ch, num_heads=4):
        super().__init__()
        self.proj_l = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.proj_h = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.ln     = nn.LayerNorm(in_ch)
        self.mha    = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        # small FFN
        self.ffn = nn.Sequential(
            nn.Linear(in_ch, in_ch*4),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch*4, in_ch)
        )

    def forward(self, f_l, f_h):
        B, C, H, W = f_l.size()
        xl = self.proj_l(f_l).flatten(2).transpose(1,2)  # B×(HW)×C
        xh = self.proj_h(f_h).flatten(2).transpose(1,2)  # B×(HW)×C
        x  = torch.cat([xl, xh], dim=1)                  # B×(2HW)×C
        x  = self.ln(x)
        x2, _ = self.mha(x, x, x)
        x  = x + x2
        x  = x + self.ffn(self.ln(x))
        xl2, xh2 = torch.split(x, [H*W, H*W], dim=1)
        xl2 = xl2.transpose(1,2).view(B, C, H, W)
        xh2 = xh2.transpose(1,2).view(B, C, H, W)
        return xl2, xh2


# ---------- light FFM (RefineNet/FPN style) ----------
class FFM(nn.Module):
    def __init__(self, in_ch_cur, in_ch_prev, out_ch):
        super().__init__()
        self.conv_cur  = ConvBNReLU(in_ch_cur, out_ch, k=3, s=1, p=1)
        self.conv_prev = ConvBNReLU(in_ch_prev, out_ch, k=3, s=1, p=1)
        self.out       = ConvBNReLU(out_ch*2, out_ch, k=3, s=1, p=1)
    def forward(self, f_cur, f_prev_up):
        f1 = self.conv_cur(f_cur)
        f2 = self.conv_prev(f_prev_up)
        x  = torch.cat([f1, f2], dim=1)
        return self.out(x)


# ---------- main refiner ----------
class SDDRNYURefiner(nn.Module):
    """
    Inputs:
      image(3), sparse(1), estimation(1),
      pseudo(1|None), e_norm(1|None in [0,1])  # e_norm이 있으면 내부 Poisson으로 pseudo 생성
    """
    def __init__(self, encoder_pretrained=True, base_ch=64, heads=4,
                 dmax: float = 10.0,
                 poisson_tol: float = 1e-5, poisson_maxiter: int = 1000,
                 poisson_init: str = "est", poisson_clip_to_max_gt: bool = False,
                 poisson_auto_flip: bool = True, poisson_est_affine: bool = True,
                 poisson_smooth_est: bool = True):
        super().__init__()
        # stems
        self.stem_L = InputFusionStem(in_ch=6, mid_ch=base_ch)
        self.stem_H = InputFusionStem(in_ch=6, mid_ch=base_ch)
        # encoder
        self.encoder = ResNetEncoder(pretrained=encoder_pretrained)

        ch = {'c1':64, 'c2':128, 'c3':256, 'c4':512}
        self.attn_c1 = AttnInteractor(ch['c1'], num_heads=heads)
        self.attn_c2 = AttnInteractor(ch['c2'], num_heads=heads)
        self.attn_c3 = AttnInteractor(ch['c3'], num_heads=heads)
        self.attn_c4 = AttnInteractor(ch['c4'], num_heads=heads)

        self.mask_c1 = nn.Conv2d(ch['c1'], 1, kernel_size=1)
        self.mask_c2 = nn.Conv2d(ch['c2'], 1, kernel_size=1)
        self.mask_c3 = nn.Conv2d(ch['c3'], 1, kernel_size=1)
        self.mask_c4 = nn.Conv2d(ch['c4'], 1, kernel_size=1)

        self.proj_c4 = ConvBNReLU(ch['c4'], 256, k=1, s=1, p=0)
        self.proj_c3 = ConvBNReLU(ch['c3'], 256, k=1, s=1, p=0)
        self.proj_c2 = ConvBNReLU(ch['c2'], 128, k=1, s=1, p=0)
        self.proj_c1 = ConvBNReLU(ch['c1'],  64, k=1, s=1, p=0)

        self.ffm_43 = FFM(in_ch_cur=256, in_ch_prev=256, out_ch=256)  # 1/16
        self.ffm_32 = FFM(in_ch_cur=128, in_ch_prev=256, out_ch=128)  # 1/8
        self.ffm_21 = FFM(in_ch_cur= 64, in_ch_prev=128, out_ch= 64)  # 1/4

        self.head_depth = nn.Sequential(
            ConvBNReLU(64, 64, k=3, s=1, p=1),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.sobel = SobelGrad()

        # ---- Poisson options ----
        self.dmax = float(dmax)
        self.poisson_tol = float(poisson_tol)
        self.poisson_maxiter = int(poisson_maxiter)
        self.poisson_init = str(poisson_init)
        self.poisson_clip_to_max_gt = bool(poisson_clip_to_max_gt)
        # cfg-like flags (네 스니펫에서 getattr(self.cfg, ...)로 읽던 항목)
        self.poisson_auto_flip = bool(poisson_auto_flip)
        self.poisson_est_affine = bool(poisson_est_affine)
        self.poisson_smooth_est = bool(poisson_smooth_est)

    # ---- helpers for Poisson ----
    @staticmethod
    def _smooth3_reflect_np(x: np.ndarray) -> np.ndarray:
        # 3x3 평균 필터 (reflect padding)
        pad = ((1,1),(1,1))
        xpad = np.pad(x, pad, mode='reflect')
        k = np.ones((3,3), np.float32) / 9.0
        y = F.conv2d(torch.from_numpy(xpad)[None,None], torch.from_numpy(k)[None,None], padding=0).numpy()[0,0]
        return y

    def _poisson_batch(self, DL: torch.Tensor, ML: torch.Tensor, E01: torch.Tensor):
        """
        DL: Bx1xHxW (metric sparse), ML: Bx1xHxW {0,1}, E01: Bx1xHxW in [0,1]
        returns: P (Bx1xHxW, metric), stats_list
        """
        B = DL.shape[0]; dmax = self.dmax; dev = DL.device
        dev_str = str(dev)
        P_list, stats_list = [], []
        for b in range(B):
            e  = E01[b,0].detach().cpu().numpy().astype(np.float32)         # [0,1]
            dl = (DL[b,0] / dmax).detach().cpu().numpy().astype(np.float32) # ~[0,1]
            m  = (ML[b,0].detach().cpu().numpy().astype(np.float32) > 0)

            # auto flip by sparse correlation
            if self.poisson_auto_flip and m.sum() >= 10:
                em = e[m].reshape(-1); dm = dl[m].reshape(-1)
                if em.size > 1 and dm.size > 1:
                    corr = np.corrcoef(em, dm)[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    if corr < 0.0: e = 1.0 - e

            # robust affine align e -> dl on sparse
            if self.poisson_est_affine and m.sum() >= 10:
                x = e[m].reshape(-1,1); y = dl[m].reshape(-1,1)
                A = np.concatenate([x, np.ones_like(x)], axis=1)
                w = np.ones((A.shape[0],1), dtype=np.float32)
                for _ in range(3):
                    Aw = A * w; yw = y * w
                    theta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
                    r = (A @ theta - y)
                    c = 1.345 * np.median(np.abs(r)) + 1e-6
                    w = (1.0 / np.maximum(1.0, np.abs(r)/c)).astype(np.float32)
                a, b0 = float(theta[0,0]), float(theta[1,0])
                e = np.clip(a*e + b0, 0.0, 1.0)

            # optional smooth
            if self.poisson_smooth_est:
                e = self._smooth3_reflect_np(e)

            est_m    = (e * dmax).astype(np.float32)
            sparse_m = (DL[b,0].detach().cpu().numpy().astype(np.float32) * m.astype(np.float32))

            # solve (prefer external poisson_gpu, else fallback)
            if callable(poisson_gpu):
                P_np, st = poisson_gpu(
                    sparse_m=sparse_m, est_m=est_m,
                    tol=self.poisson_tol, maxiter=self.poisson_maxiter,
                    device=dev_str, init=self.poisson_init,
                    clip_to_max_gt=self.poisson_clip_to_max_gt
                )
            else:
                # very simple harmonic fill fallback (Δv=0 with Dirichlet at known∪border)
                P_np, st = self._poisson_fallback_np(sparse_m, est_m, m)

            P_list.append(torch.from_numpy(P_np)[None, None])
            stats_list.append(st)
        P = torch.cat(P_list, dim=0).to(device=dev, dtype=torch.float32)
        return P, stats_list

    @staticmethod
    def _poisson_fallback_np(sparse_m: np.ndarray, est_m: np.ndarray, m_bool: np.ndarray,
                             iters: int = 1000, tol: float = 1e-5):
        H, W = est_m.shape
        v = est_m.copy()
        known = m_bool.copy()
        # add image border as Dirichlet (use est)
        known[0,:] = True; known[-1,:] = True; known[:,0] = True; known[:,-1] = True
        v_known = est_m.copy()
        known_sparse = m_bool & (sparse_m > 0)
        v_known[known_sparse] = sparse_m[known_sparse]

        # Jacobi iterations solving Δv=0 in unknown
        for it in range(iters):
            v_old = v
            v = 0.25*(np.roll(v,1,0)+np.roll(v,-1,0)+np.roll(v,1,1)+np.roll(v,-1,1))
            v[known] = v_known[known]
            if np.mean(np.abs(v - v_old)) < tol:
                break
        return v.astype(np.float32), {"solver":"jacobi", "iters": it+1}

    # ---------- forward ----------
    def _build_branch_inputs(self, image, sparse, estimation, pseudo):
        mask = (sparse > 0).float()
        xH = torch.cat([image, estimation, sparse, mask], dim=1)  # 3+1+1+1
        xL = torch.cat([image, pseudo,    sparse, mask], dim=1)  # 3+1+1+1
        return xL, xH

    def _attend_and_weighted_fuse(self, fL, fH, attn, mask_head):
        fL2, fH2 = attn(fL, fH)
        omega = torch.sigmoid(mask_head(fH2))
        Ffuse = (1.0 - omega) * fL2 + omega * fH2
        return Ffuse, omega

    def forward(self, image, sparse, estimation, pseudo=None, e_norm=None, return_pseudo: bool = False):
        """
        - pseudo가 None이면 e_norm(0~1) 또는 estimation/dmax로 내부 Poisson 생성
        - return_pseudo=True 이면 반환 dict에 'poisson' 포함
        """
        B, _, H, W = image.shape
        ML = (sparse > 0).float()

        # 내부 Poisson 생성
        pseudo_used = pseudo
        if pseudo_used is None:
            if e_norm is None:
                e01 = torch.clamp(estimation / self.dmax, 0.0, 1.0)
            else:
                e01 = e_norm
            with torch.no_grad():
                pseudo_used, _ = self._poisson_batch(DL=sparse, ML=ML, E01=e01)

        # stems -> encoder
        xL_in, xH_in = self._build_branch_inputs(image, sparse, estimation, pseudo_used)
        xL = self.stem_L(xL_in); xH = self.stem_H(xH_in)
        fL = self.encoder(xL);   fH = self.encoder(xH)

        # Attention + Ω-fusion
        c4_fuse, om4 = self._attend_and_weighted_fuse(fL['c4'], fH['c4'], self.attn_c4, self.mask_c4)
        c3_fuse, om3 = self._attend_and_weighted_fuse(fL['c3'], fH['c3'], self.attn_c3, self.mask_c3)
        c2_fuse, om2 = self._attend_and_weighted_fuse(fL['c2'], fH['c2'], self.attn_c2, self.mask_c2)
        c1_fuse, om1 = self._attend_and_weighted_fuse(fL['c1'], fH['c1'], self.attn_c1, self.mask_c1)

        # top-down decoder (타깃 사이즈로 보간)
        p4    = self.proj_c4(c4_fuse)
        p3_in = self.proj_c3(c3_fuse)
        p4_up = F.interpolate(p4, size=p3_in.shape[-2:], mode='bilinear', align_corners=False)
        p3    = self.ffm_43(p3_in, p4_up)

        p2_in = self.proj_c2(c2_fuse)
        p3_up = F.interpolate(p3, size=p2_in.shape[-2:], mode='bilinear', align_corners=False)
        p2    = self.ffm_32(p2_in, p3_up)

        p1_in = self.proj_c1(c1_fuse)
        p2_up = F.interpolate(p2, size=p1_in.shape[-2:], mode='bilinear', align_corners=False)
        p1    = self.ffm_21(p1_in, p2_up)

        # heads
        d0_4 = self.head_depth(p1)
        d0   = F.interpolate(d0_4, size=(H, W), mode='bilinear', align_corners=False)
        g0   = self.sobel(d0)
        om   = torch.sigmoid(om1)
        om_up= F.interpolate(om, size=(H, W), mode='bilinear', align_corners=False)

        out = {'depth': d0, 'edge': g0, 'omega': om_up}
        if return_pseudo:
            out['poisson'] = pseudo_used  # Bx1xHxW (metric)
        return out


class InputStem(nn.Module):
    """[RGB(3), P(1), DL(1), ML(1)] -> 3ch (이미지처럼 백본 재사용)"""
    def __init__(self,in_ch=6, mid=32):
        super().__init__()
        self.net=nn.Sequential(ConvBNReLU(in_ch,mid),ConvBNReLU(mid,mid),nn.Conv2d(mid,3,1,bias=False))
    def forward(self,x): return self.net(x)

class ResNet34Enc(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        b = resnet34(pretrained=pretrained)
        self.conv1=b.conv1; self.bn1=b.bn1; self.relu=b.relu; self.maxpool=b.maxpool
        self.l1=b.layer1; self.l2=b.layer2; self.l3=b.layer3; self.l4=b.layer4
    def forward(self,x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        c1=self.l1(x); c2=self.l2(c1); c3=self.l3(c2); c4=self.l4(c3)
        return c1,c2,c3,c4  # /4,/8,/16,/32
    
    
class PoissonRefineNet(nn.Module):
    """
    단일 스트림 정제기: RGB, DL, ML, (내부)Poisson P -> δ 예측 -> D = P + tanh(δ)*cap
    """
    def __init__(self, dmax=10.0, cap_ratio=0.15, encoder_pretrained=True,
                 poisson_tol=1e-5, poisson_maxiter=1000, poisson_init="est",
                 poisson_clip_to_max_gt=False, auto_flip=True, est_affine=True, smooth_est=True):
        super().__init__()
        self.dmax=float(dmax); self.cap_ratio=float(cap_ratio)
        # Poisson 옵션
        self.poisson_tol=float(poisson_tol); self.poisson_maxiter=int(poisson_maxiter)
        self.poisson_init=str(poisson_init); self.poisson_clip=bool(poisson_clip_to_max_gt)
        self.auto_flip=bool(auto_flip); self.est_affine=bool(est_affine); self.smooth_est=bool(smooth_est)

        # 단일 스트림
        self.stem = InputStem(in_ch=6, mid=64)
        self.enc  = ResNet34Enc(pretrained=encoder_pretrained)
        # 디코더(FPN 간단)
        self.p4 = ConvBNReLU(512,256,k=1,p=0); self.p3 = ConvBNReLU(256,256,k=1,p=0)
        self.p2 = ConvBNReLU(128,128,k=1,p=0); self.p1 = ConvBNReLU( 64, 64,k=1,p=0)
        self.ffm43=FFM(256,256,256); self.ffm32=FFM(128,256,128); self.ffm21=FFM(64,128,64)
        # 잔차 헤드(δ)
        self.head = nn.Sequential(ConvBNReLU(64,64), nn.Conv2d(64,1,1))
        self.sobel = SobelGrad()

    # ====== Poisson 유틸 ======
    @staticmethod
    def _smooth3_reflect_np(x: np.ndarray) -> np.ndarray:
        pad=((1,1),(1,1)); xpad=np.pad(x,pad,mode="reflect")
        k=np.ones((3,3),np.float32)/9.0
        y=F.conv2d(torch.from_numpy(xpad)[None,None], torch.from_numpy(k)[None,None], padding=0).numpy()[0,0]
        return y

    def _poisson_fallback_np(self, sparse_m: np.ndarray, est_m: np.ndarray, m_bool: np.ndarray,
                             iters: int = 400, tol: float = 1e-5):
        H,W=est_m.shape; v=est_m.copy(); known=m_bool.copy()
        known[0,:]=known[-1,:]=True; known[:,0]=known[:,-1]=True
        v_known=est_m.copy(); ks=m_bool&(sparse_m>0); v_known[ks]=sparse_m[ks]
        for it in range(iters):
            v_old=v
            v=0.25*(np.roll(v,1,0)+np.roll(v,-1,0)+np.roll(v,1,1)+np.roll(v,-1,1))
            v[known]=v_known[known]
            if np.mean(np.abs(v-v_old))<tol: break
        return v.astype(np.float32), {"solver":"jacobi","iters":it+1}

    def _poisson_batch(self, DL, ML, E01):
        B=DL.shape[0]; dev=DL.device; dmax=self.dmax; dev_str=str(dev)
        Ps=[]
        for b in range(B):
            e  = E01[b,0].detach().cpu().numpy().astype(np.float32)
            dl = (DL[b,0]/dmax).detach().cpu().numpy().astype(np.float32)
            m  = (ML[b,0].detach().cpu().numpy().astype(np.float32)>0)
            if self.auto_flip and m.sum()>=10:
                em=e[m].reshape(-1); dm=dl[m].reshape(-1)
                if em.size>1 and dm.size>1:
                    corr=float(np.corrcoef(em,dm)[0,1]); corr=0.0 if not np.isfinite(corr) else corr
                    if corr<0.0: e=1.0-e
            if self.est_affine and m.sum()>=10:
                x=e[m].reshape(-1,1); y=dl[m].reshape(-1,1)
                A=np.concatenate([x,np.ones_like(x)],1); w=np.ones((A.shape[0],1),np.float32)
                for _ in range(3):
                    Aw=A*w; yw=y*w
                    theta,*_=np.linalg.lstsq(Aw,yw,rcond=None)
                    r=A@theta-y; c=1.345*np.median(np.abs(r))+1e-6
                    w=(1.0/np.maximum(1.0,np.abs(r)/c)).astype(np.float32)
                a,b0=float(theta[0,0]),float(theta[1,0]); e=np.clip(a*e+b0,0.0,1.0)
            if self.smooth_est: e=self._smooth3_reflect_np(e)
            est_m=(e*dmax).astype(np.float32)
            sparse_m=(DL[b,0].detach().cpu().numpy().astype(np.float32)*m.astype(np.float32))
            if callable(poisson_gpu):
                P_np,_=poisson_gpu(sparse_m=sparse_m, est_m=est_m, tol=self.poisson_tol,
                                   maxiter=self.poisson_maxiter, device=dev_str,
                                   init=self.poisson_init, clip_to_max_gt=self.poisson_clip)
            else:
                P_np,_=self._poisson_fallback_np(sparse_m,est_m,m)
            Ps.append(torch.from_numpy(P_np)[None,None])
        P=torch.cat(Ps,0).to(dev,dtype=torch.float32)
        return P

    # ====== Forward ======
    def forward(self, image, DL, ML, e_norm=None, P_external=None, return_poisson=True):
        """
        - P_external이 있으면 그대로 사용, 없으면 e_norm+DL/ML로 내부 Poisson 생성
        """
        B,_,H,W = image.shape
        if P_external is None:
            assert e_norm is not None, "Need e_norm(0~1) to build Poisson internally"
            with torch.no_grad():
                P = self._poisson_batch(DL, ML, e_norm)   # metric
        else:
            P = P_external

        x_in = torch.cat([image, P, DL, ML], dim=1)      # 3+1+1+1=6
        x    = self.stem(x_in)
        c1,c2,c3,c4 = self.enc(x)

        p4    = self.p4(c4)
        p3_in = self.p3(c3); p4_up=F.interpolate(p4, size=p3_in.shape[-2:], mode='bilinear', align_corners=False)
        f3    = self.ffm43(p3_in, p4_up)

        p2_in = self.p2(c2); f3_up=F.interpolate(f3, size=p2_in.shape[-2:], mode='bilinear', align_corners=False)
        f2    = self.ffm32(p2_in, f3_up)

        p1_in = self.p1(c1); f2_up=F.interpolate(f2, size=p1_in.shape[-2:], mode='bilinear', align_corners=False)
        f1    = self.ffm21(p1_in, f2_up)

        delta_4 = self.head(f1)                               # /4
        delta   = F.interpolate(delta_4, size=(H,W), mode='bilinear', align_corners=False)

        # 잔차 제한 + 앵커(희소) 보정
        cap = self.cap_ratio * self.dmax
        D = torch.clamp(P + torch.tanh(delta)*cap, min=0.0, max=self.dmax)
        # 희소 앵커(정확히 고정하고 싶으면 아래 줄 주석 해제)
        # D = (1-ML)*D + ML*DL

        out = {"depth": D, "edge": self.sobel(D)}
        if return_poisson: out["poisson"] = P
        return out