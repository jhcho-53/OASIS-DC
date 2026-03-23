# OASIS-DC Refactored

This is the refactored version of OASIS-DC with unified training and evaluation pipeline supporting both NYU Depth V2 and KITTI datasets.

## Project Structure

```
├── src/
│   ├── trainers/
│   │   ├── base_trainer.py      # Common training/evaluation logic
│   │   ├── nyu_trainer.py       # NYU-specific trainer
│   │   └── kitti_trainer.py     # KITTI-specific trainer
│   ├── datasets/
│   │   ├── nyu_dataset.py       # NYU dataset and dataloader
│   │   └── kitti_dataset.py     # KITTI dataset and dataloader  
│   └── utils/
│       ├── metrics.py           # Common evaluation metrics
│       ├── io_utils.py          # I/O utilities for images/depth
│       └── poisson_utils.py     # Poisson solver utilities
├── configs/                     # Configuration files
├── scripts/                     # Execution scripts
├── train.py                     # Unified training entry point
└── eval.py                      # Unified evaluation entry point
```

## Usage

### Training

```bash
# NYU 1-shot training
python train.py --dataset nyu --config configs/nyu_1shot.yaml --save-dir runs/nyu_1shot

# NYU 100-shot training  
python train.py --dataset nyu --config configs/nyu_100shot.yaml --save-dir runs/nyu_100shot

# Resume from checkpoint
python train.py --dataset nyu --config configs/nyu_1shot.yaml --resume runs/nyu_1shot/checkpoint_best.pth
```

### Evaluation

```bash
# Evaluate all modes (full, residual_off, poisson_only)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode all --output results.json

# Evaluate specific mode
python eval.py --dataset nyu --config configs/nyu_1shot.yaml --mode poisson_only --checkpoint runs/nyu_1shot/checkpoint_best.pth

# Poisson-only evaluation (no checkpoint needed)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode poisson_only
```

### Using Scripts

```bash
# Training scripts
./scripts/train_nyu_1shot.sh

# Evaluation scripts  
./scripts/eval_kitti.sh
```

## Configuration

Configuration files use YAML format with dataset-specific settings:

- `configs/nyu_1shot.yaml` - NYU 1-shot configuration
- `configs/nyu_100shot.yaml` - NYU 100-shot configuration
- `configs/kitti_eval.yaml` - KITTI evaluation configuration

Key configuration sections:

- `data`: Dataset paths and preprocessing settings
- `model`: Model architecture and Poisson solver parameters
- `train`: Training hyperparameters and loss weights
- `eval`: Evaluation settings

## Dataset Setup

### NYU Depth V2

Expected structure:
```
/path/to/NYUv2/
├── rgb_da/              # RGB images
├── depth_inpainted_mm/  # Ground truth depth (16-bit PNG in mm)
├── mono_rel/            # Precomputed relative depth estimates
└── lists/               # Sample lists for different shot settings
```

### KITTI Depth Completion

Expected structure:
```  
/path/to/kitti/
├── image/               # RGB images
├── groundtruth_depth/   # Ground truth depth (16-bit PNG in mm)
├── velodyne_raw/        # Sparse LiDAR depth (16-bit PNG in mm)
└── est/                 # [Optional] Precomputed depth estimates
```

## Key Features

1. **Unified Interface**: Single entry points for both datasets
2. **Modular Design**: Dataset-specific implementations with shared base classes
3. **Flexible Evaluation**: Support for multiple evaluation modes
4. **Configuration-Driven**: YAML-based configuration system
5. **Reproducible**: Consistent random seeding across components
6. **Extensible**: Easy to add new datasets or evaluation modes

## Migration from Original Code

The refactored version maintains compatibility with the original model architecture while providing a cleaner interface:

- Original `train_mcprop.py` → Use `train.py --dataset nyu`
- Original `eval_kitti_root_poisson_only.py` → Use `eval.py --dataset kitti --mode poisson_only`
- Original dataset-specific scripts → Use unified scripts with appropriate configs

## Dependencies

Same as original OASIS-DC:
- PyTorch
- NumPy  
- PIL/Pillow
- tqdm
- PyYAML (new dependency for configuration)

## Notes

- The refactored code maintains the same model architecture and core algorithms
- All evaluation metrics and Poisson solving remain identical to the original
- Configuration files can be easily modified for different experimental settings
- The modular structure allows for easy extension to new datasets