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
  <a href="https://github.com/jhcho-53/OASIS-DC">
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

## Qualitative Demo (10-shot Training)

[▶ Demo video](https://github.com/user-attachments/assets/65fe38f0-6758-4c65-9e79-d05f117583fc)


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
