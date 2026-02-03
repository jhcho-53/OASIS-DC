<h2 align="center">OASIS-DC: Generalizable Depth Completion via Output-level Alignment of Sparse-Integrated Monocular Pseudo Depth</h2>

<p align="center">
  <strong>Jaehyeon Cho</strong> ·
  <strong>Jhonhyun An</strong>
  <br>
  <strong>ICRA 2026</strong><br>
</p>

<p align="center">
  <a href="ARXIV_OR_PAPER_URL"><strong><code>📄 Paper</code></strong></a>
  <a href="ARXIV_OR_PAPER_URL"><img src="https://img.shields.io/badge/Arxiv-Paper-2b9348.svg?logo=arXiv" alt="arXiv Paper" /></a>
  <a href="SOURCE_CODE_URL"><strong><code>💻 Source Code</code></strong></a>
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

## Contact
- **Jaehyeon Cho** — `jjh000503@gachon.ac.kr`
- **Jhonhyun An** — `jhonghyun@gachon.ac.kr`  

> If you have questions, please open an issue or contact us via email.
