import numpy as np
import matplotlib.pyplot as plt
from dataloader.dataloader_nyu import build_mcprop_dataloaders
def save_pipeline_figure(DL, ML, E01, dmax, out_path="pipeline_main_figure.png",
                         tol=1e-4, maxiter=300, gt=None):
    """한 장에 (a)~(h) 파이프라인 패널을 저장.
    DL: (H,W) or (1,H,W) sparse metric depth(0=미측정)
    ML: (H,W) or (1,H,W) valid mask {0,1}
    E01: (H,W) or (1,H,W) affine-invariant prior in [0,1]
    dmax: metric 정규화 상한(float)
    gt: (선택) 정답 depth의 0~1 정규화 버전(동일 해상도)
    """
    # --- to numpy 2D ---
    def to_np2(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 1: x = x[0]
        assert x.ndim == 2, f"Expected 2D or (1,H,W), got {x.shape}"
        return x.astype(np.float32, copy=False)

    DL, ML, E01 = to_np2(DL), to_np2(ML), to_np2(E01)
    H, W = DL.shape
    M = ML > 0.5
    DL_norm = np.where(M, DL / float(dmax), 0.0).astype(np.float32)
    E01_pre = E01.copy()

    # --- (b) Auto-flip by corr ---
    x_corr = E01_pre[M].reshape(-1)
    y_corr = DL_norm[M].reshape(-1)
    corr, flipped = np.nan, False
    if x_corr.size >= 2 and np.std(x_corr) > 1e-8 and np.std(y_corr) > 1e-8:
        corr = float(np.corrcoef(x_corr, y_corr)[0, 1])
        if corr < 0:
            E01 = 1.0 - E01
            flipped = True

    # --- (c) IRLS (Huber, 3회 재가중) ---
    x = E01[M].reshape(-1); y = DL_norm[M].reshape(-1)
    if x.size >= 2:
        w = np.ones_like(x, dtype=np.float32); a, b = 1.0, 0.0
        for _ in range(3):
            X = np.stack([x, np.ones_like(x)], axis=1)
            Xw, yw = X * w[:, None], y * w
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            a, b = float(beta[0]), float(beta[1])
            r = y - (a * x + b)
            med = np.median(r)
            mad = np.median(np.abs(r - med)) + 1e-8
            c = 1.345 * mad
            w = np.where(np.abs(r) <= c, 1.0, c / (np.abs(r) + 1e-12)).astype(np.float32)
    else:
        a, b = 1.0, 0.0
    E = np.clip(a * E01 + b, 0.0, 1.0).astype(np.float32)

    # --- (d) 3-탭 반사 스무딩 ---
    def smooth3(u):
        k = np.array([1., 2., 1.], dtype=np.float32); k /= k.sum()
        uh = np.pad(u, ((0, 0), (1, 1)), mode='reflect')
        uh = k[0]*uh[:, :-2] + k[1]*uh[:, 1:-1] + k[2]*uh[:, 2:]
        uv = np.pad(uh, ((1, 1), (0, 0)), mode='reflect')
        uv = k[0]*uv[:-2, :] + k[1]*uv[1:-1, :] + k[2]*uv[2:, :]
        return uv.astype(np.float32)
    E_s = smooth3(E)

    # --- (e) K/U 구성 ---
    border = np.zeros_like(M, bool)
    border[0, :]=border[-1, :]=border[:, 0]=border[:, -1]=True
    K = M | border
    Y = np.zeros_like(E_s, np.float32)
    Y[M] = DL_norm[M]
    Y[border & ~M] = E_s[border & ~M]

    # --- (f) 5-점 라플라시안 ---
    def laplace(z):
        z = z.astype(np.float32, copy=False)
        lap = np.zeros_like(z, np.float32)
        lap[1:-1,1:-1] = 4*z[1:-1,1:-1] - (z[2:,1:-1]+z[:-2,1:-1]+z[1:-1,2:]+z[1:-1,:-2])
        lap[0,1:-1]   = 4*z[0,1:-1]    - (z[1,1:-1]  + z[0,2:]   + z[0,:-2])
        lap[-1,1:-1]  = 4*z[-1,1:-1]   - (z[-2,1:-1] + z[-1,2:]  + z[-1,:-2])
        lap[1:-1,0]   = 4*z[1:-1,0]    - (z[2:,0]    + z[:-2,0]  + z[1:-1,1])
        lap[1:-1,-1]  = 4*z[1:-1,-1]   - (z[2:,-1]   + z[:-2,-1] + z[1:-1,-2])
        lap[0,0]      = 4*z[0,0]       - (z[1,0]     + z[0,1])
        lap[0,-1]     = 4*z[0,-1]      - (z[1,-1]    + z[0,-2])
        lap[-1,0]     = 4*z[-1,0]      - (z[-2,0]    + z[-1,1])
        lap[-1,-1]    = 4*z[-1,-1]     - (z[-2,-1]   + z[-1,-2])
        return lap

    # Δx = ΔE, x|K=Y  → x = x0 + u, Δu = ΔE − Δx0(in U), u|K=0
    x0 = np.zeros_like(E_s, np.float32); x0[K] = Y[K]
    rhs = laplace(E_s) - laplace(x0); rhs[K] = 0.0

    # --- (g) PCG + Jacobi(대각=4) ---
    def pcg(rhs, K, maxiter, tol, u0=None):
        if u0 is None: u = np.zeros_like(rhs, np.float32)
        else: u = u0.astype(np.float32).copy()
        u[K] = 0.0
        def A(z):
            z0 = z.copy(); z0[K] = 0.0
            return laplace(z0)
        r = rhs - A(u); r[K] = 0.0
        M_inv = np.zeros_like(rhs, np.float32); M_inv[~K] = 0.25
        z = M_inv * r; p = z.copy()
        rz_old = float(np.sum(r[~K]*z[~K]))
        res0 = np.sqrt(float(np.sum(r[~K]**2))); res_init = res0 if res0>0 else 1.0
        residuals, iters = [res0/res_init], 0
        for it in range(maxiter):
            Ap = A(p); denom = float(np.sum(p[~K]*Ap[~K]))
            if abs(denom) < 1e-20: break
            alpha = rz_old/denom
            u += alpha*p; r -= alpha*Ap; r[K]=0.0
            res = np.sqrt(float(np.sum(r[~K]**2)))
            residuals.append(res/res_init); iters = it+1
            if res/res_init < tol: break
            z = M_inv*r; rz_new = float(np.sum(r[~K]*z[~K]))
            beta = rz_new/(rz_old + 1e-20)
            p = z + beta*p; p[K]=0.0; rz_old = rz_new
        return u, residuals, iters

    u_est0 = (E_s - x0) * (~K)      # init="est"
    u_est, res_est, it_est = pcg(rhs, K, maxiter, tol, u0=u_est0)
    u_zero, res_zero, it_zero = pcg(rhs, K, maxiter, tol, u0=None)
    X = np.clip(x0 + u_est, 0.0, 1.0)

    err = None
    if gt is not None:
        gt = to_np2(gt)
        err = np.abs(X - gt)

    # --- Figure (2×5 패널) ---
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=180)

    ax = axes[0,0]; ax.imshow(DL_norm); ax.set_title("(a-1) DL (norm)"); ax.axis("off")
    ax = axes[0,1]; ax.imshow(M.astype(np.float32)); ax.set_title("(a-2) ML (mask)"); ax.axis("off")
    ax = axes[0,2]; ax.imshow(E01_pre); ax.set_title("(a-3) E01 (prior)"); ax.axis("off")

    ax = axes[0,3]
    if x_corr.size >= 2:
        ax.plot(x_corr, y_corr, '.', ms=2, alpha=0.5)
        ax.set_xlabel("E01[M]"); ax.set_ylabel("DL[M]/dmax"); ax.set_title("(b) Auto-flip")
        ax.text(0.02, 0.95, f"corr={corr:.3f}\nflipped={flipped}", transform=ax.transAxes,
                va="top", ha="left")
    else:
        ax.axis("off"); ax.set_title("(b) Auto-flip (insufficient M)")

    ax = axes[0,4]
    if x.size >= 2:
        ax.plot(x, y, '.', ms=2, alpha=0.5)
        xs = np.linspace(0, 1, 100); ax.plot(xs, a*xs + b, lw=1.5)
        ax.set_xlabel("E01'[M]"); ax.set_ylabel("DL[M]/dmax"); ax.set_title("(c) IRLS fit")
        ax.text(0.02, 0.95, f"a={a:.3f}, b={b:.3f}", transform=ax.transAxes, va="top", ha="left")
    else:
        ax.axis("off"); ax.set_title("(c) IRLS fit (insufficient M)")

    ax = axes[1,0]; ax.imshow(np.hstack([E, E_s])); ax.set_title("(d) E → E_s"); ax.axis("off")

    ax = axes[1,1]; ax.imshow(K.astype(np.float32)); ax.set_title("(e) Known set K"); ax.axis("off")

    ax = axes[1,2]
    ker = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
    ax.imshow(ker, interpolation='nearest')
    for (i,j), v in np.ndenumerate(ker): ax.text(j, i, f"{int(v)}", ha='center', va='center')
    ax.set_title("(f) 5-pt Laplacian"); ax.axis("off")

    ax = axes[1,3]
    if len(res_est):
        ax.semilogy(res_est, label="init=est"); ax.semilogy(res_zero, label="init=0")
        ax.set_xlabel("iter"); ax.set_ylabel("rel. residual")
        ax.set_title(f"(g) PCG+Jacobi (tol={tol}, maxit={maxiter})"); ax.legend(loc="upper right", fontsize=8)
    else:
        ax.axis("off"); ax.set_title("(g) PCG")

    ax = axes[1,4]
    if err is None:
        ax.imshow(X); ax.set_title("(h) Result X")
    else:
        ax.imshow(np.hstack([X, err])); ax.set_title("(h) X | |X−GT|")
    ys, xs = np.where(M); ax.plot(xs, ys, '.', ms=0.5, alpha=0.6)  # sparse 위치 오버레이
    ax.axis("off")

    plt.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return {"corr": corr, "flipped": flipped, "a": a, "b": b,
            "iters_est": len(res_est)-1, "iters_zero": len(res_zero)-1,
            "out_path": out_path}

# --------- 예시 실행(데모) ----------
if __name__ == "__main__":
    import argparse
    import numpy as np
    import torch
    import os

    # (선택) 앞서 제공했던 파이프라인 그림 함수가 별도 파일에 있다면 import
    # 없으면 아래 fallback으로 간단 요약 이미지만 저장합니다.
    try:
        from pipeline_figure import save_pipeline_figure  # 사용자 보유 함수
    except Exception:
        save_pipeline_figure = None

    # 간단 요약 저장(backup): DL/ML/E_norm/D_gt를 2x2로 저장
    def _save_quick_grid(DL, ML, E01, D_gt01, out_path="pipeline_quick.png"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=160)
        ax[0,0].imshow(DL);    ax[0,0].set_title("DL (norm by dmax)"); ax[0,0].axis("off")
        ax[0,1].imshow(ML);    ax[0,1].set_title("ML (mask)");         ax[0,1].axis("off")
        ax[1,0].imshow(E01);   ax[1,0].set_title("E_norm [0,1]");      ax[1,0].axis("off")
        ax[1,1].imshow(D_gt01);ax[1,1].set_title("GT (norm)");         ax[1,1].axis("off")
        plt.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
        return out_path

    parser = argparse.ArgumentParser(description="Run MCProp dataloader once and save figure.")
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--train-list", default=None, help="train list txt")
    parser.add_argument("--test-list", required=True, help="test list txt")
    parser.add_argument("--split", choices=["train","test"], default="test", help="which split to draw")
    parser.add_argument("--shots", type=int, default=None, help="few-shot k (train split만 해당)")
    parser.add_argument("--emode", default="precomputed", choices=["precomputed","foundation","zeros"])
    parser.add_argument("--dmax", type=float, default=None, help="metric max; 미지정시 GT 99.9%분위")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sparse", type=int, default=500)
    parser.add_argument("--out", default="pipeline_main_figure.png")
    parser.add_argument("--strict-precomputed", action="store_true")
    args = parser.parse_args()

    # DataLoader 구성
    train_loader, test_loader = build_mcprop_dataloaders(
        root=args.root,
        train_list=args.train_list if args.train_list else args.test_list,  # shots로 train만 따로 만들 때 필요
        test_list=args.test_list,
        seed=args.seed,
        shots=args.shots,
        n_sparse=args.n_sparse,
        emode=args.emode,
        foundation_infer=None,
        batch_size=1,
        num_workers=0,
        pre_mono_dir="mono_rel",
        strict_precomputed=args.strict_precomputed or True,  # 기본 True 유지
        normalize_on_valid_gt=True,
    )
    loader = train_loader if args.split == "train" else test_loader

    # 첫 배치만 추출
    batch = next(iter(loader))
    I      = batch["I"][0]        # (3,228,304) torch
    DL_t   = batch["DL"][0,0]     # (228,304)
    ML_t   = batch["ML"][0,0]     # (228,304) float mask
    E01_t  = batch["E_norm"][0,0] # (228,304)
    D_gt_t = batch["D_gt"][0,0]   # (228,304)

    # numpy 변환
    DL   = DL_t.detach().cpu().numpy().astype(np.float32)
    ML   = ML_t.detach().cpu().numpy().astype(np.float32)
    E01  = E01_t.detach().cpu().numpy().astype(np.float32)
    D_gt = D_gt_t.detach().cpu().numpy().astype(np.float32)

    # dmax 결정: 지정 없으면 GT의 유효값 99.9% 분위수(극단치 완화)
    valid = D_gt > 0
    if args.dmax is None:
        if np.any(valid):
            dmax = float(np.quantile(D_gt[valid], 0.999))
        else:
            dmax = 10.0  # 안전 기본값(실내)
    else:
        dmax = float(args.dmax)

    # 정규화 버전들
    DL_norm = np.where(valid, np.clip(DL / max(dmax, 1e-6), 0.0, 1.0), 0.0).astype(np.float32)

    # GT를 [0,1]로(유효 영역 기준)
    def _minmax01_local(arr, mask=None, eps=1e-6):
        if mask is not None and mask.any():
            v = arr[mask]
            a, b = float(v.min()), float(v.max())
        else:
            a, b = float(arr.min()), float(arr.max())
        if b - a < eps: return np.zeros_like(arr, np.float32)
        out = (arr - a) / (b - a + eps)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    D_gt01 = _minmax01_local(D_gt, valid)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) 파이프라인 전체 그림(가능 시)
    if save_pipeline_figure is not None:
        info = save_pipeline_figure(
            DL=DL, ML=ML, E01=E01, dmax=dmax,
            out_path=args.out, tol=1e-4, maxiter=300, gt=D_gt01
        )
        print("Saved pipeline:", info["out_path"])
    else:
        # 2) 빠른 요약 저장(대체)
        quick_path = _save_quick_grid(DL_norm, ML, E01, D_gt01,
                                      out_path=os.path.splitext(args.out)[0] + "_quick.png")
        print("Saved quick grid:", quick_path)

    # 샘플 메타 정보 출력(확인용)
    meta = batch["meta"]
    sid = meta["id"]
    crop_box = tuple(meta["crop_box"].tolist())
    orig_hw = tuple(meta["orig_size"].tolist())
    mono_path = meta["mono_path"]
    print(f"[First sample] id={sid}, orig={orig_hw}, crop={crop_box}, mono={mono_path}, dmax={dmax:.3f}")
