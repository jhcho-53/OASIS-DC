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

## What this repository will contain (Coming Soon)
- [ ] Official implementation of **OASIS-DC**
- [ ] Training & evaluation code for **NYU Depth V2**
- [ ] Training & evaluation code for **KITTI Depth Completion**
- [ ] Pretrained models & logs
- [ ] Reproducible scripts (dataset preparation, evaluation, visualization)

---

## Experiment

### KITTI Depth Completion Benchmark
> Protocol: 1/10/100-shot are sampled from the training split only; evaluation is on the official 1,000-frame validation split.

| Method | 1-shot RMSE ↓ | 1-shot MAE ↓ | 10-shot RMSE ↓ | 10-shot MAE ↓ | 100-shot RMSE ↓ | 100-shot MAE ↓ | 1-Sequence RMSE ↓ | 1-Sequence MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 9.2748 | 3.5921 | 2.0222 | 0.7825 | 1.4510 | 0.5184 | 2.6289 | 0.8355 |
| S2D | 8.8479 | 5.6022 | 5.0500 | 3.1469 | 4.2799 | 2.6633 | 4.7950 | 2.5610 |
| NLSPN | 7.2899 | 4.7422 | 4.0070 | 2.2588 | 2.4979 | 1.1710 | 4.0290 | 1.7881 |
| DySPN | 2.6350 | 0.8870 | 2.2701 | 0.9150 | 1.8777 | 0.6188 | 2.8530 | 0.7980 |
| CompletionFormer | 4.7212 | 2.3789 | 3.1601 | 1.4740 | 2.6122 | 1.3299 | 4.5588 | 1.9603 |
| BPNet | 5.4000 | 1.0740 | 1.8799 | 0.5559 | 1.3001 | 0.3910 | 2.1322 | 0.6420 |
| DepthPrompting | 2.9840 | 1.1430 | 2.3988 | 1.1290 | 1.8249 | 0.6240 | 2.9468 | 0.9869 |
| **OASIS-DC (Ours)** | **1.4190** | **0.5073** | **1.2830** | **0.4001** | **1.2455** | **0.3548** | **1.5782** | **0.5540** |

---

### NYUv2 Depth Completion Benchmark
| Method | 1-shot RMSE ↓ | 1-shot MAE ↓ | 10-shot RMSE ↓ | 10-shot MAE ↓ | 100-shot RMSE ↓ | 100-shot MAE ↓ | 1-Sequence RMSE ↓ | 1-Sequence MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 1.4827 | 1.2058 | 0.3166 | 0.1961 | 0.2854 | 0.1307 | 0.3166 | 0.1961 |
| NLSPN | 1.9358 | 1.6132 | 1.5995 | 0.8261 | 0.5501 | 0.4150 | 0.8881 | 0.6421 |
| DySPN | 1.5474 | 1.2851 | 0.4102 | 0.2817 | 0.3079 | 0.1706 | 0.2584 | 0.1320 |
| CompletionFormer | 1.8218 | 1.5539 | 1.1583 | 1.0162 | 0.9914 | 0.8164 | 0.6779 | 0.5356 |
| CostDCNet | 1.2298 | 0.9754 | 0.2363 | 0.1288 | 0.1770 | 0.0836 | 0.2066 | 0.0954 |
| BPNet | 0.3573 | 0.2077 | 0.2392 | 0.1120 | 0.1757 | 0.0793 | 0.2220 | 0.1040 |
| DepthPrompting | 0.3583 | 0.2067 | 0.2195 | 0.1006 | 0.2101 | 0.1008 | 0.2335 | 0.1191 |
| UniDC | **0.2099** | **0.1075** | **0.1657** | **0.0794** | **0.1473** | **0.0669** | **0.1632** | **0.0745** |
| **OASIS-DC (Ours)** | 0.2105 | 0.1105 | 0.1670 | 0.0838 | 0.1484 | 0.0706 | 0.1644 | 0.0787 |

---

## Contact
- **Jaehyeon Cho** — `jjh000503@gachon.ac.kr`
- **Jhonhyun An** — `jhonghyun@gachon.ac.kr`  

> If you have questions, please open an issue or contact us via email.
