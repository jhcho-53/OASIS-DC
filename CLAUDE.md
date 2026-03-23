# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OASIS-DC is a depth completion system that uses monocular foundation models with sparse range measurements. The codebase implements a refinement network for metric depth prediction from sparse anchors, particularly effective for few-shot learning regimes.

## Key Commands

### Unified Training and Evaluation (Refactored)
```bash
# Train NYU n-shot model with flexible shot count
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 10 --seed 0 --save-dir runs/nyu_10shot

# Train KITTI n-shot model
python train.py --dataset kitti --config configs/kitti_nshot.yaml --shots 100 --seed 0 --save-dir runs/kitti_100shot

# Resume from checkpoint
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 1 --resume runs/nyu_1shot/checkpoint_best.pth

# Evaluate all modes (full, residual_off, poisson_only)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode all --output results.json

# Evaluate n-shot model with specific configuration
python eval.py --dataset nyu --config configs/nyu_eval.yaml --shots 10 --mode full --checkpoint runs/nyu_10shot/checkpoint_best.pth

# Poisson-only evaluation (no checkpoint needed)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode poisson_only
```

### Using Scripts
```bash
# N-shot training scripts with flexible parameters
./scripts/train_nyu_nshot.sh 10 0 runs/nyu_10shot     # Train NYU 10-shot, seed 0
./scripts/train_kitti_nshot.sh 100 1 runs/kitti_100shot_s1  # Train KITTI 100-shot, seed 1

# N-shot evaluation script (supports both datasets)
./scripts/eval_nshot.sh nyu 10 runs/nyu_10shot/checkpoint_best.pth all results/nyu_10shot.json
./scripts/eval_nshot.sh kitti 1 runs/kitti_1shot/checkpoint_best.pth poisson_only results/kitti_1shot.json
```

### Data Preparation
```bash
# Generate dataset lists manually (if needed)
python make_list.py --dataset nyuv2 --root /path/to/NYUv2 --out nyu_val.txt
python make_list.py --dataset kitti --root /path/to/kitti --out kitti_val.txt

# Note: Training lists are automatically generated when missing
# The training script will create n-shot lists by sampling from full dataset
# with the specified seed for reproducibility
```

## Architecture Overview

### Refactored Structure

The codebase has been refactored into a unified training/evaluation pipeline:

1. **Unified Entry Points**:
   - `train.py`: Unified training script for both NYU and KITTI
   - `eval.py`: Unified evaluation script with multiple modes

2. **Modular Trainers** (`src/trainers/`):
   - `base_trainer.py`: Common training/evaluation logic
   - `nyu_trainer.py`: NYU-specific implementation
   - `kitti_trainer.py`: KITTI-specific implementation

3. **Dataset Modules** (`src/datasets/`):
   - `nyu_dataset.py`: NYU dataset and dataloader
   - `kitti_dataset.py`: KITTI dataset and dataloader

4. **Refactored Utilities** (`utils/`):
   - `core/`: Basic utilities (metrics, tensor ops, I/O)
   - `training/`: Training utilities (checkpoints, config builders)
   - `datasets/`: Dataset handling with NYU/KITTI specific modules
   - `depth/`: Depth processing (Poisson solvers, sampling)
   - `visualization/`: Depth visualization and colormaps
   - `losses/`: Loss functions (depth, gradient, self-distillation)
   - `scripts/`: Data processing and preparation scripts

### Core Model Components (Refactored)

The model architecture has been refactored into a modular structure:

1. **Main Architecture** (`models/architectures/oasis.py`): Complete OASIS-DC implementation
   - Combines monocular foundation priors with sparse anchors
   - Uses Poisson equation solving for pseudo-depth generation
   - Includes affinity propagation and residual refinement

2. **Base Components** (`models/base/`):
   - `TinyFeat`: Lightweight feature backbone
   - `ResidualHead`: Final depth refinement network  
   - `CurvatureGen`: Curvature and FiLM parameter generation
   - `KernelGate`: Kernel gating mechanism
   - `AnchorHead`: Learnable anchor point system

3. **Affinity Modules** (`models/affinity/`):
   - `EllipticAffinity`: Elliptical geometry-aware propagation
   - `HCLApproxAffinity`: Hyperbolic approximation affinity
   - Utility functions for affinity normalization

4. **Poisson Solvers** (`models/poisson/`):
   - `poisson_gpu`: GPU-accelerated CG solver
   - `ScreenedPoissonLayer`: Differentiable Poisson layer
   - Finite difference operations

5. **Utilities** (`models/utils/`):
   - Shape manipulation utilities
   - Weight initialization functions  
   - Common tensor operations

## Configuration System

The refactored codebase uses YAML-based configuration files:

- `configs/nyu_nshot.yaml`: NYU n-shot flexible configuration with adaptive parameters
- `configs/kitti_nshot.yaml`: KITTI n-shot flexible configuration with adaptive parameters
- `configs/nyu_eval.yaml`: NYU evaluation configuration
- `configs/kitti_eval.yaml`: KITTI evaluation configuration

### Key Configuration Sections

- `data`: Dataset paths, preprocessing settings, and shot configurations (`shots`, `seed`)
- `model`: Model architecture, Poisson solver parameters, affinity settings
- `train`: Training hyperparameters, loss weights, optimization settings
- `eval`: Evaluation modes and batch settings
- `shot_adaptations`: Automatic parameter adjustments based on shot count (1, 10, 100, sequence)

### Important Model Parameters

- `dmax`: Maximum depth (10.0 for NYU, 80.0 for KITTI)
- `steps`: Number of affinity propagation steps (6-8)
- `geometry`: "hyper" (hyperbolic) or "ellip" (elliptic) affinity computation
- `anchor_alpha`: Dirichlet boundary condition strength (0.1-0.7)
- `kernels`: Kernel sizes for affinity computation [3,5] or [3,5,7]
- `use_residual`: Enable/disable residual refinement network
- `poisson_*`: Poisson solver tolerances, iterations, initialization

## Dataset Structure

### NYU Depth V2 Expected Structure
```
/path/to/NYUv2/
├── rgb_da/              # RGB images
├── depth_inpainted_mm/  # Ground truth depth (16-bit PNG in mm)
├── mono_rel/            # Precomputed relative depth estimates
└── lists/               # Sample lists for different shot settings
```

### KITTI Depth Completion Expected Structure
```  
/path/to/kitti/
├── image/               # RGB images
├── groundtruth_depth/   # Ground truth depth (16-bit PNG in mm)
├── velodyne_raw/        # Sparse LiDAR depth (16-bit PNG in mm)
└── est/                 # [Optional] Precomputed depth estimates
```

## Evaluation Modes

The codebase supports multiple evaluation modes:
1. **full**: Complete model with all components
2. **residual_off**: Network output without residual refinement
3. **poisson_only**: Pure Poisson-solved pseudo-depth (no checkpoint needed)
4. **all**: Evaluate all modes above

## Migration Notes

The refactored version maintains compatibility with the original model architecture:

- Original training scripts → Use `train.py --dataset [nyu|kitti]`
- Original evaluation scripts → Use `eval.py --dataset [dataset] --mode [mode]`
- Configuration files use YAML format instead of command-line arguments

## Development Notes

- The model configuration is defined in `config/schema.py` using dataclasses
- Training uses Adam optimizer with configurable learning rate and weight decay
- Checkpoints include model state, optimizer state, epoch, and metrics
- Reproducibility is ensured through consistent random seeding
- The system supports both CPU and CUDA execution

## Automatic List Generation

The training script automatically generates n-shot training lists when they don't exist:

1. **Full Dataset Lists**: If `lists/nyu_train_full.txt` or `lists/kitti_train_full.txt` don't exist, they are created using `make_list.py`
2. **N-Shot Sampling**: Custom shot counts (e.g., 5-shot, 25-shot) are generated by randomly sampling from the full training set
3. **Reproducible Sampling**: The `seed` parameter ensures consistent sample selection across runs
4. **Automatic Fallback**: If requested shots exceed available samples, all available samples are used

### Example Usage:
```bash
# This will automatically create lists/nyu_train_5shot_seed0.txt if it doesn't exist
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 5 --seed 0
```