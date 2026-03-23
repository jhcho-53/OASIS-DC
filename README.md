<h2 align="center">OASIS-DC: Generalizable Depth Completion via Output-level Alignment of Sparse-Integrated Monocular Pseudo Depth</h2>

<p align="center">
  <strong>Jaehyeon Cho</strong> ·
  <strong>Jhonhyun An</strong>
  <br>
  <strong>ICRA 2026</strong><br>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2602.01268">
    <strong><code>📄 Paper</code></strong>
  </a>
  <a href="#">
    <strong><code>💻 Source Code</code></strong>
  </a>
</p>

---

## 🔥 News
- **[2026]** OASIS-DC has been **accepted to ICRA 2026**.
- **Code and pretrained models will be released soon.** (This repository will be updated.)

---

## Overview
Recent monocular foundation models excel at zero-shot depth estimation, yet their outputs are inherently relative rather than metric, limiting direct use in robotics and autonomous driving. We leverage the fact that relative depth preserves global layout and boundaries: by calibrating it with sparse range measurements, we transform it into a pseudo metric depth prior. Building on this prior, we design a refinement network that follows the prior where reliable and deviates where necessary, enabling accurate metric predictions from very few labeled samples. The resulting system is particularly effective when curated validation data are unavailable, sustaining stable scale and sharp edges across few-shot regimes. These findings suggest that coupling foundation priors with sparse anchors is a practical route to robust, deployment-ready depth completion under real-world label scarcity.

> **Note:** This repository is under preparation.  
> Training code, evaluation scripts, and checkpoints will be uploaded soon.

---

## What this repository contains
- [x] Official implementation of **OASIS-DC**
- [x] Training & evaluation code for **NYU Depth V2**
- [x] Training & evaluation code for **KITTI Depth Completion**
- [ ] Pretrained models & logs
- [x] Reproducible scripts (dataset preparation, evaluation, visualization)

---

## Dataset Setup

### NYU Depth V2 Dataset Structure
```
NYUv2_root/
├── rgb_da/                    # RGB images
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── depth_inpainted_mm/        # Ground truth depth (millimeters)
│   ├── 00001.png  
│   ├── 00002.png
│   └── ...
└── mono_rel/                  # Precomputed monocular estimates (optional)
    ├── 00001.png
    ├── 00002.png
    └── ...
```

**NYU Dataset Requirements:**
- **RGB Images**: `rgb_da/` directory with PNG files (480×640 → resized to 240×320)
- **Ground Truth**: `depth_inpainted_mm/` directory with PNG files (millimeter units, max 10m)
- **Monocular Estimates**: `mono_rel/` or custom directory (optional, normalized to [0,1])
- **Sparse Points**: Sampled from ground truth at runtime (default: 500 points)
- **File Format**: 5-digit zero-padded naming (00001.png, 00002.png, ...)

### KITTI Depth Completion Dataset Structure
```
KITTI_root/
├── data_depth_selection/
│   └── depth_selection/
│       └── val_selection_cropped/
│           ├── image/                  # RGB images
│           │   ├── 2011_09_26_drive_0001_sync_image_0000000000_02.png
│           │   ├── 2011_09_26_drive_0001_sync_image_0000000001_02.png
│           │   └── ...
│           ├── velodyne_raw/           # Sparse LiDAR depth  
│           │   ├── 2011_09_26_drive_0001_sync_velodyne_raw_0000000000_02.png
│           │   ├── 2011_09_26_drive_0001_sync_velodyne_raw_0000000001_02.png
│           │   └── ...
│           └── groundtruth_depth/      # Ground truth depth
│               ├── 2011_09_26_drive_0001_sync_groundtruth_depth_0000000000_02.png
│               ├── 2011_09_26_drive_0001_sync_groundtruth_depth_0000000001_02.png
│               └── ...
└── [est_dir]/                         # Precomputed depth estimates (optional)
    ├── 2011_09_26_drive_0001_sync_image_0000000000_02.png
    └── ...
```

**KITTI Dataset Requirements:**
- **RGB Images**: `image/` directory with PNG files (original size ~375×1242)
- **Sparse Depth**: `velodyne_raw/` directory with LiDAR measurements (PNG, 16-bit)
- **Ground Truth**: `groundtruth_depth/` directory with PNG files (max 80m)
- **Depth Estimates**: Custom directory (optional, various formats supported)
- **File Format**: Long descriptive naming with date_drive_sync_type_frame_camera format

### Creating Dataset Lists

```bash
# NYU dataset list (automatic detection)
python make_list.py --dataset nyuv2 --root /path/to/NYUv2 --out nyu_val.txt

# KITTI dataset list (automatic detection)  
python make_list.py --dataset kitti --root /path/to/KITTI --out kitti_val.txt

# Manual directory specification (NYU)
python make_list.py --dataset nyuv2 --root /path/to/NYUv2 \
    --nyu-est-dir /path/to/NYUv2/rgb_da \
    --nyu-gt-dir /path/to/NYUv2/depth_inpainted_mm \
    --out nyu_custom.txt
```

**Note:** Training lists are automatically generated with n-shot sampling when missing. File names must match across all directories using stem matching (filename without extension).

## Training

### Basic Training Commands

```bash
# Train NYU n-shot model
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 10 --seed 0 --save-dir runs/nyu_10shot

# Train KITTI n-shot model
python train.py --dataset kitti --config configs/kitti_nshot.yaml --shots 100 --seed 0 --save-dir runs/kitti_100shot

# Resume from checkpoint
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 1 \
    --resume runs/nyu_1shot/checkpoint_best.pth --save-dir runs/nyu_1shot_resume

# Override configuration parameters
python train.py --dataset nyu --config configs/nyu_nshot.yaml --shots 10 --seed 42 \
    --save-dir runs/nyu_10shot_seed42
```

### Using Training Scripts

```bash
# N-shot training scripts with flexible parameters
./scripts/train_nyu_nshot.sh 10 0 runs/nyu_10shot     # Train NYU 10-shot, seed 0
./scripts/train_kitti_nshot.sh 100 1 runs/kitti_100shot_s1  # Train KITTI 100-shot, seed 1
```

### Training Features

- **Automatic N-shot Sampling**: Training lists are automatically generated by sampling from the full dataset
- **Reproducible Seeds**: Use different seeds for different n-shot samples
- **Flexible Shot Counts**: Support for 1-shot, 10-shot, 100-shot, and custom counts
- **Checkpoint Management**: Automatic saving of best models and periodic checkpoints
- **Multi-mode Validation**: Evaluation during training with different model modes

## Evaluation

### Basic Evaluation Commands

```bash
# Evaluate all modes (full, residual_off, poisson_only)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode all --output results.json

# Evaluate specific mode with checkpoint
python eval.py --dataset nyu --config configs/nyu_eval.yaml --shots 10 --mode full \
    --checkpoint runs/nyu_10shot/checkpoint_best.pth --output nyu_results.json

# Poisson-only evaluation (no checkpoint needed)
python eval.py --dataset kitti --config configs/kitti_eval.yaml --mode poisson_only \
    --output poisson_results.json

# Evaluate n-shot configuration
python eval.py --dataset nyu --config configs/nyu_eval.yaml --shots 1 \
    --checkpoint runs/nyu_1shot/checkpoint_best.pth --mode all
```

### Using Evaluation Scripts

```bash
# N-shot evaluation script (supports both datasets)
./scripts/eval_nshot.sh nyu 10 runs/nyu_10shot/checkpoint_best.pth all results/nyu_10shot.json
./scripts/eval_nshot.sh kitti 1 runs/kitti_1shot/checkpoint_best.pth poisson_only results/kitti_1shot.json
```

### Evaluation Modes

1. **`full`**: Complete model pipeline including all components
2. **`residual_off`**: Network output without residual refinement
3. **`poisson_only`**: Pure Poisson-solved pseudo-depth (no neural network)
4. **`all`**: Run all above modes and compare results

### Evaluation Features

- **Multiple Metrics**: RMSE, MAE, δ1, δ2, δ3 accuracy metrics
- **Timing Analysis**: Solver timing statistics for Poisson-only mode
- **JSON Output**: Structured results saving for analysis
- **Progress Tracking**: Real-time progress bars with metric updates

---

## Experiment

### KITTI Depth Completion Benchmark

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left;">Method</th>
      <th colspan="2" style="text-align:center;">1-shot</th>
      <th colspan="2" style="text-align:center;">10-shot</th>
      <th colspan="2" style="text-align:center;">100-shot</th>
      <th colspan="2" style="text-align:center;">1-Sequence</th>
    </tr>
    <tr>
      <th style="text-align:center;">RMSE (m)</th>
      <th style="text-align:center;">MAE (m)</th>
      <th style="text-align:center;">RMSE (m)</th>
      <th style="text-align:center;">MAE (m)</th>
      <th style="text-align:center;">RMSE (m)</th>
      <th style="text-align:center;">MAE (m)</th>
      <th style="text-align:center;">RMSE (m)</th>
      <th style="text-align:center;">MAE (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left;">CSPN</td>
      <td style="text-align:center;">9.2748</td><td style="text-align:center;">3.5921</td>
      <td style="text-align:center;">2.0222</td><td style="text-align:center;">0.7825</td>
      <td style="text-align:center;">1.4510</td><td style="text-align:center;">0.5184</td>
      <td style="text-align:center;">2.6289</td><td style="text-align:center;">0.8355</td>
    </tr>
    <tr>
      <td style="text-align:left;">S2D</td>
      <td style="text-align:center;">8.8479</td><td style="text-align:center;">5.6022</td>
      <td style="text-align:center;">5.0500</td><td style="text-align:center;">3.1469</td>
      <td style="text-align:center;">4.2799</td><td style="text-align:center;">2.6633</td>
      <td style="text-align:center;">4.7950</td><td style="text-align:center;">2.5610</td>
    </tr>
    <tr>
      <td style="text-align:left;">NLSPN</td>
      <td style="text-align:center;">7.2899</td><td style="text-align:center;">4.7422</td>
      <td style="text-align:center;">4.0070</td><td style="text-align:center;">2.2588</td>
      <td style="text-align:center;">2.4979</td><td style="text-align:center;">1.1710</td>
      <td style="text-align:center;">4.0290</td><td style="text-align:center;">1.7881</td>
    </tr>
    <tr>
      <td style="text-align:left;">DySPN</td>
      <td style="text-align:center;">2.6350</td><td style="text-align:center;">0.8870</td>
      <td style="text-align:center;">2.2701</td><td style="text-align:center;">0.9150</td>
      <td style="text-align:center;">1.8777</td><td style="text-align:center;">0.6188</td>
      <td style="text-align:center;">2.8530</td><td style="text-align:center;">0.7980</td>
    </tr>
    <tr>
      <td style="text-align:left;">CompletionFormer</td>
      <td style="text-align:center;">4.7212</td><td style="text-align:center;">2.3789</td>
      <td style="text-align:center;">3.1601</td><td style="text-align:center;">1.4740</td>
      <td style="text-align:center;">2.6122</td><td style="text-align:center;">1.3299</td>
      <td style="text-align:center;">4.5588</td><td style="text-align:center;">1.9603</td>
    </tr>
    <tr>
      <td style="text-align:left;">BPNet</td>
      <td style="text-align:center;">5.4000</td><td style="text-align:center;">1.0740</td>
      <td style="text-align:center;">1.8799</td><td style="text-align:center;">0.5559</td>
      <td style="text-align:center;">1.3001</td><td style="text-align:center;">0.3910</td>
      <td style="text-align:center;">2.1322</td><td style="text-align:center;">0.6420</td>
    </tr>
    <tr>
      <td style="text-align:left;">DepthPrompting</td>
      <td style="text-align:center;">2.9840</td><td style="text-align:center;">1.1430</td>
      <td style="text-align:center;">2.3988</td><td style="text-align:center;">1.1290</td>
      <td style="text-align:center;">1.8249</td><td style="text-align:center;">0.6240</td>
      <td style="text-align:center;">2.9468</td><td style="text-align:center;">0.9869</td>
    </tr>
    <tr>
      <td style="text-align:left;"><b>OASIS-DC (Ours)</b></td>
      <td style="text-align:center;"><b>1.4190</b></td><td style="text-align:center;"><b>0.5073</b></td>
      <td style="text-align:center;"><b>1.2830</b></td><td style="text-align:center;"><b>0.4001</b></td>
      <td style="text-align:center;"><b>1.2455</b></td><td style="text-align:center;"><b>0.3548</b></td>
      <td style="text-align:center;"><b>1.5782</b></td><td style="text-align:center;"><b>0.5540</b></td>
    </tr>
  </tbody>
</table>

---

## Contact
- **Jaehyeon Cho** — `jjh000503@gachon.ac.kr`
- **Jhonhyun An** — `jhonghyun@gachon.ac.kr`  

> If you have questions, please open an issue or contact us via email.

---

## Related Works
We are deeply grateful for the following outstanding opensource work; without them, our work would not have been possible.
- [UniDC](https://github.com/AIR-THU/UniV2X)
- [DepthPrompting](https://github.com/JinhwiPark/DepthPrompting)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [CSPN](https://github.com/XinJCheng/CSPN)