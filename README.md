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

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left;">Method</th>
      <th colspan="2">1-shot</th>
      <th colspan="2">10-shot</th>
      <th colspan="2">100-shot</th>
    </tr>
    <tr>
      <th>RMSE (m)</th>
      <th>MAE (m)</th>
      <th>RMSE (m)</th>
      <th>MAE (m)</th>
      <th>RMSE (m)</th>
      <th>MAE (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left;">CSPN</td>
      <td align="right">1.4827</td><td align="right">1.2058</td>
      <td align="right">0.3166</td><td align="right">0.1961</td>
      <td align="right">0.2854</td><td align="right">0.1307</td>
    </tr>
    <tr>
      <td style="text-align:left;">NLSPN</td>
      <td align="right">1.9358</td><td align="right">1.6132</td>
      <td align="right">1.5995</td><td align="right">0.8261</td>
      <td align="right">0.5501</td><td align="right">0.4150</td>
    </tr>
    <tr>
      <td style="text-align:left;">DySPN</td>
      <td align="right">1.5474</td><td align="right">1.2851</td>
      <td align="right">0.4102</td><td align="right">0.2817</td>
      <td align="right">0.3079</td><td align="right">0.1706</td>
    </tr>
    <tr>
      <td style="text-align:left;">CompletionFormer</td>
      <td align="right">1.8218</td><td align="right">1.5539</td>
      <td align="right">1.1583</td><td align="right">1.0162</td>
      <td align="right">0.9914</td><td align="right">0.8164</td>
    </tr>
    <tr>
      <td style="text-align:left;">CostDCNet</td>
      <td align="right">1.2298</td><td align="right">0.9754</td>
      <td align="right">0.2363</td><td align="right">0.1288</td>
      <td align="right">0.1770</td><td align="right">0.0836</td>
    </tr>
    <tr>
      <td style="text-align:left;">BPNet</td>
      <td align="right">0.3573</td><td align="right">0.2077</td>
      <td align="right">0.2392</td><td align="right">0.1120</td>
      <td align="right">0.1757</td><td align="right">0.0793</td>
    </tr>
    <tr>
      <td style="text-align:left;">DepthPrompting</td>
      <td align="right">0.3583</td><td align="right">0.2067</td>
      <td align="right">0.2195</td><td align="right">0.1006</td>
      <td align="right">0.2101</td><td align="right">0.1008</td>
    </tr>
    <tr>
      <td style="text-align:left;"><b>UniDC</b></td>
      <td align="right"><b>0.2099</b></td><td align="right"><b>0.1075</b></td>
      <td align="right"><b>0.1657</b></td><td align="right"><b>0.0794</b></td>
      <td align="right"><b>0.1473</b></td><td align="right"><b>0.0669</b></td>
    </tr>
    <tr>
      <td style="text-align:left;"><b>OASIS-DC (Ours)</b></td>
      <td align="right">0.2105</td><td align="right">0.1105</td>
      <td align="right">0.1670</td><td align="right">0.0838</td>
      <td align="right">0.1484</td><td align="right">0.0706</td>
    </tr>
  </tbody>
</table>

---

## Contact
- **Jaehyeon Cho** — `jjh000503@gachon.ac.kr`
- **Jhonhyun An** — `jhonghyun@gachon.ac.kr`  

> If you have questions, please open an issue or contact us via email.
