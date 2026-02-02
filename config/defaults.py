# config/defaults.py
from typing import Dict, Any

DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        "rgb":      "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "sparse":   "/home/vip/jaehyeon/RPN/data/8line/train/firstseq_one/sparse/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "pseudo":   "/home/vip/jaehyeon/RPN/data/8line/train/firstseq_one/poisson/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "estim":    "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw_da/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "gt":       "/home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png"
    },
    "train": {
        "epochs": 300,
        "lr": 1e-3,
        "save_dir": "runs_oneshot",
        "tag": "mcprop_1shot"
    },
    "model": {
        "dmax": 80.0,
        "steps": 6,
        "use_residual": True,
        "use_sparse": False,

        # --- anchor(Dirichlet) ---
        "anchor_alpha": 0.7,         # 초기값 (learnable이 False면 고정)
        "anchor_learnable": True,   # True면 학습
        "anchor_mode": "map",     # "scalar" or "map"

        # curvature / geometry
        "kappa_min": 1e-3,
        "kappa_max": 1.0,
        "kernels": [3,5,7],
        "geometry": "hyperbolic"
    },
    "loss": {
        "mu_scaleinv": 0.1,
        "w_lidar": 0.3,
        "w_anchor_reg": 0.0          # >0이면 LiDAR 위치에서 α→1로 유도
    },
    "experiments": [
        {"geometry": "hyperbolic", "tag": "mcprop_1shot_hyp"},
        {"geometry": "elliptic",   "tag": "mcprop_1shot_ellip"}
    ],

    "kshot": {
        # (A) 경로 기반 지정: 각 modality의 루트 디렉터리
        #   - 디렉터리 구조가 동일하면 relpath 매칭(권장)
        #   - 구조가 다르면 파일명(stem) 매칭 사용
        "train_dirs": {
            "rgb_dir":    "",   # 예: ".../kitti_raw/train/....../image_02"
            "sparse_dir": "",   # 예: ".../data_depth_velodyne/train/....../velodyne_raw/image_02"
            "pseudo_dir": "",   # 예: ".../pseudo_depth_map/train/....../velodyne_raw/image_02"
            "estim_dir":  "",   # 예: ".../kitti_raw_da/train/....../image_02"
            "gt_dir":     "",   # (선택) 없으면 비지도
            "pattern":    "*.png",
            "match_mode": "relpath"   # "relpath" | "stem"
        },
        "val_dirs": {           # 비우면 train pool에서 홀드아웃
            "rgb_dir":    "",
            "sparse_dir": "",
            "pseudo_dir": "",
            "estim_dir":  "",
            "gt_dir":     "",
            "pattern":    "*.png",
            "match_mode": "relpath"
        },

        # (B) 샘플 수/추출
        "K_train": 1,          # 10-shot: 학습 10장
        "K_val":   1,           # 검증 5장 (디렉터리 없으면 train pool에서 홀드아웃)
        "seed":    1,
        "epochs":  50,
        "lr":      1e-3,
        "warmup_epochs": 3,
        "min_lr_ratio": 0.05,
        "steps_start": 8,
        "steps_final": 16,
        "batch_size": 1,
        "num_workers": 4,
        "shuffle": True,
        "save_dir": "runs_1shot_8lines",
        "tag": "mcprop_kshot",
        "preview_every": 5
    }
}
