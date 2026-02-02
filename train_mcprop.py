23.114694595336914# -*- coding: utf-8 -*-
"""
두 가지 모드만 평가:
  (A) residual 끄기 (residual-off) -> 네트워크 출력 D_pred 평가
  (B) poisson-only (pseudo-depth P) -> Poisson으로 만든 P만 평가

필수 의존:
 - mcprop_dataloader.py  (NYUv2ForMCProp / build_mcprop_dataloaders)
 - MCPropNet (사용자가 제공한 클래스; 아래 import 경로만 환경에 맞게 수정)

사용 예시:
  python train_mcprop.py \
    --root /media/vip/T31/NYUv2_official \
    --test-list /home/vip/jaehyeon/OASIS/lists/nyu_test_full.txt \
    --shots 100 \
    --emode precomputed \
    --dmax 10.0 \
    --batch-size 1 \
    --num-workers 4 \
    --save-json results_nyu.json
"""

import os
import json
import argparse
from types import SimpleNamespace
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.dataloader_nyu import build_mcprop_dataloaders, seed_all
from models.model import MCPropNet
# 사용자 파일 import (경로/모듈명은 환경에 맞게 바꿔주세요)

# --------------------------- metrics
@torch.no_grad()
def rmse_mae(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor):
    """
    pred, gt: (B,1,H,W) [meters]
    valid_mask: (B,1,H,W) or (B,H,W) boolean
    """
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid = valid_mask > 0
    n = valid.sum().item()
    if n == 0:
        return float("nan"), float("nan")

    err = (pred - gt)[valid]
    rmse = torch.sqrt(torch.mean(err ** 2)).item()
    mae  = torch.mean(torch.abs(err)).item()
    return rmse, mae


def pretty_stats(stats_dict):
    return {k: round(float(v), 6) for k, v in stats_dict.items()}


# --------------------------- cfg builder (MCPropNet에 필요한 최소 설정)
def make_cfg(
    dmax=10.0,
    steps=8,
    geometry="hyper",             # "hyper" or "ellip"
    use_sparse=True,
    use_residual=False,           # (A) residual-off 에서 False 유지
    # ----- Poisson/anchor/affinity 등 디폴트
    poisson_only=False,
    use_poisson=True,
    poisson_tol=1e-5,
    poisson_maxiter=1000,
    poisson_init="est",
    poisson_clip_to_max_gt=False,
    poisson_auto_flip=True,
    poisson_est_affine=True,
    poisson_smooth_est=True,
    use_p_affinity=True,
    p_only_gate=False,
    kernels=(3,5),
    kappa_min=0.03,
    kappa_max=0.5,
    anchor_learnable=False,
    anchor_mode="map",         # "scalar" or "map"
    anchor_alpha=0.1,             # 초기 anchor 비율
    min_gate=0.0,
    min_alpha=0.0,
):
    return SimpleNamespace(
        dmax=float(dmax),
        steps=int(steps),
        geometry=str(geometry),
        use_sparse=bool(use_sparse),
        use_residual=bool(use_residual),

        poisson_only=bool(poisson_only),
        use_poisson=bool(use_poisson),
        poisson_tol=float(poisson_tol),
        poisson_maxiter=int(poisson_maxiter),
        poisson_init=str(poisson_init),
        poisson_clip_to_max_gt=bool(poisson_clip_to_max_gt),
        poisson_auto_flip=bool(poisson_auto_flip),
        poisson_est_affine=bool(poisson_est_affine),
        poisson_smooth_est=bool(poisson_smooth_est),

        use_p_affinity=bool(use_p_affinity),
        p_only_gate=bool(p_only_gate),

        kernels=tuple(kernels),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),

        # learnable anchor
        anchor_learnable=bool(anchor_learnable),
        anchor_mode=str(anchor_mode),
        anchor_alpha=float(anchor_alpha),

        min_gate=float(min_gate),
        min_alpha=float(min_alpha),
    )


# --------------------------- evaluation loops
@torch.no_grad()
def eval_residual_off(model: MCPropNet, loader: DataLoader, device: torch.device):
    """
    (A) residual 끄기: model.res = None 로 강제 비활성화
    - Poisson + (optional) affinity propagation만 적용
    - 최종 D_pred(Dt) 로 RMSE/MAE
    """
    # 안전하게 residual 비활성화
    if hasattr(model, "res"):
        model.res = None
    # steps는 사용자 cfg에 설정된 값 사용(기본 8). propagation을 완전히 끄고 싶으면 0으로 조절 가능.

    model.eval()
    dmax = float(model.cfg.dmax)

    agg = defaultdict(float)
    n_im = 0

    pbar = tqdm(loader, desc="[A] residual-off", ncols=100)
    for batch in pbar:
        I   = batch["I"].to(device)                 # (B,3,H,W)
        DL  = batch["DL"].to(device)                # (B,1,H,W)
        ML  = batch["ML"].to(device)                # (B,1,H,W)
        E01 = batch["E_norm"].to(device)            # (B,1,H,W) [0,1]
        GT  = batch["D_gt"].to(device)              # (B,1,H,W)
        VM  = batch["meta"]["valid_mask"].to(device)  # (B,H,W) bool

        # forward
        D_pred, aux = model(I, DL, ML, E01)         # D_pred: (B,1,H,W) [m]

        rmse, mae = rmse_mae(D_pred, GT, VM)
        agg["rmse"] += rmse
        agg["mae"]  += mae
        n_im += 1
        pbar.set_postfix(rmse=f"{rmse:.4f}", mae=f"{mae:.4f}")

    if n_im == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    return {"rmse": agg["rmse"]/n_im, "mae": agg["mae"]/n_im}


@torch.no_grad()
def eval_poisson_only(model: MCPropNet, loader: DataLoader, device: torch.device):
    """
    (B) Poisson으로 만든 pseudo-depth(P)만 평가:
      - 네트워크의 내부 Poisson 배치 함수만 호출하여 P 계산
      - affinity/refinement/residual은 전혀 사용하지 않음
      - P(meters) vs GT 로 RMSE/MAE
    """
    model.eval()
    agg = defaultdict(float)
    n_im = 0

    pbar = tqdm(loader, desc="[B] poisson-only (P)", ncols=100)
    for batch in pbar:
        DL  = batch["DL"].to(device)           # (B,1,H,W)
        ML  = batch["ML"].to(device)           # (B,1,H,W)
        E01 = batch["E_norm"].to(device)       # (B,1,H,W)
        GT  = batch["D_gt"].to(device)         # (B,1,H,W)
        VM  = batch["meta"]["valid_mask"].to(device)  # (B,H,W) bool

        # 내부 Poisson만 사용 (가장 빠르고 정확한 "pseudo-only" 평가)
        P, _ = model._poisson_batch(DL, ML, E01)   # (B,1,H,W) [m]

        rmse, mae = rmse_mae(P, GT, VM)
        agg["rmse"] += rmse
        agg["mae"]  += mae
        n_im += 1
        pbar.set_postfix(rmse=f"{rmse:.4f}", mae=f"{mae:.4f}")

    if n_im == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    return {"rmse": agg["rmse"]/n_im, "mae": agg["mae"]/n_im}


# --------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="NYUv2 root")
    parser.add_argument("--test-list", type=str, required=True, help="test list txt")
    parser.add_argument("--train-list", type=str, default=None, help="(옵션) train list txt")
    parser.add_argument("--shots", type=int, default=None, help="few-shot 학습 split 생성 시 샷 수 (평가만 할 땐 무시됨)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sparse", type=int, default=500)
    parser.add_argument("--emode", type=str, default="precomputed", choices=["precomputed","foundation","zeros"],
                        help="E_norm 생성 모드 (precomputed 권장)")
    parser.add_argument("--midas-short", type=int, default=256, help="foundation 모드에서 짧은 변 길이")
    parser.add_argument("--pre-mono-dir", type=str, default="mono_rel")
    parser.add_argument("--dmax", type=float, default=10.0, help="데이터셋 최대 깊이 (NYU=10, KITTI=80 등)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-json", type=str, default=None, help="평가 결과 저장 경로(JSON)")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # dataloader (train_loader는 여기선 사용하지 않아도 build 함수 요구사항으로 생성)
    train_list = args.train_list if args.train_list is not None else args.test_list  # dummy
    train_loader, test_loader = build_mcprop_dataloaders(
        root=args.root,
        train_list=train_list,
        test_list=args.test_list,
        seed=args.seed,
        shots=args.shots,
        n_sparse=args.n_sparse,
        emode=args.emode,
        foundation_infer=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        midas_short=args.midas_short,
        pre_mono_dir=args.pre_mono_dir,
        strict_precomputed=True,
        normalize_on_valid_gt=True,
    )

    # 모델 구성: residual-off가 기본이므로 use_residual=False
    cfg = make_cfg(
        dmax=args.dmax,
        use_residual=False,   # (A) 모드에서 residual 끈 상태로 forward
        steps=1,              # propagation step; 필요시 0으로 줄여 속도↑
        geometry="hyper",
        use_sparse=True,
    )
    model = MCPropNet(cfg).to(device)
    model.eval()

    # (A) residual-off 평가 (D_pred)
    res_residual_off = eval_residual_off(model, test_loader, device)

    # (B) poisson-only 평가 (P)
    res_poisson_only = eval_poisson_only(model, test_loader, device)

    # 결과 출력
    print("\n=== Evaluation (NYUv2) ===")
    print("[A] residual-off -> D_pred")
    print(f"  RMSE: {res_residual_off['rmse']:.4f} m | MAE: {res_residual_off['mae']:.4f} m")
    print("[B] Poisson-only -> P (pseudo-depth)")
    print(f"  RMSE: {res_poisson_only['rmse']:.4f} m | MAE: {res_poisson_only['mae']:.4f} m")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(
                {
                    "residual_off": pretty_stats(res_residual_off),
                    "poisson_only": pretty_stats(res_poisson_only),
                    "config": {
                        "dmax": cfg.dmax,
                        "steps": cfg.steps,
                        "n_sparse": args.n_sparse,
                        "emode": args.emode,
                        "pre_mono_dir": args.pre_mono_dir,
                        "seed": args.seed,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()
