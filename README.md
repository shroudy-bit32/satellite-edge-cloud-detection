# satellite-edge-cloud-detection

# Satellite On-Board Cloud Detection & Bandwidth Optimization Framework

This repository contains the software framework, trained ONNX models, and technical dockets for on-board satellite cloud detection and downlink bandwidth optimization. The project targets 6-band Sentinel-2 imagery and is benchmarked against **ESA $\Phi$-sat-1 (CloudScout)** operational parameters.

---

## Project Architecture & Technical Specifications

### 1. Model Pipeline & Topologies
- **Primary Binary Classifier:** MobileNetV2 backbone (6 input channels adapted via `timm`), single logit output.
- **Standalone U-Net Segmentation (Recommended):** MobileNetV2 (`0.5x` width) encoder paired with a lightweight Depthwise Separable Convolutional decoder.
  - *Parameters:* 540,249
  - *Input Tensor:* `1 x 6 x 64 x 64` (6-band Sentinel-2 L1C reflectances: B02, B03, B04, B08, B10, B11)
  - *Output Tensor:* `1 x 1 x 64 x 64` (Binary cloud mask logits)

### 2. Quantization Scheme (Post-Training Quantization)
- **Engine:** ONNX Runtime QDQ Static Quantization
- **Weight Precision:** `QInt8` with **Per-Channel** scaling (mandatory for MobileNetV2 inverted residual stability)
- **Activation Precision:** `QUInt8` (2.4x speedup on x86/ARM over signed INT8)
- **Calibration Method:** Percentile-based calibration over 100 representative validation scenes

---

## Benchmark & Memory Profiling

Measurements reflect ONNX Runtime CPU execution threads.

| Metric / Parameter | Classifier (FP32) | Classifier (INT8) | U-Net (FP32) | **U-Net v3.2 (INT8 - Proposed)** |
| :--- | :--- | :--- | :--- | :--- |
| **ONNX File Size** | 8.66 MB | 2.59 MB | 2.35 MB | **0.96 MB** |
| **Tile Resolution** | $256 \times 256$ | $256 \times 256$ | $256 \times 256$ | **$64 \times 64$** |
| **Accuracy / IoU** | 0.9563 Acc | 0.9552 Acc | 0.8892 IoU | **0.8807 IoU (0.9366 Dice)** |
| **Precision / Recall** | 0.9751 / 0.9462 | 0.9679 / 0.9517 | 0.9704 / 0.9140 | **0.9771 / 0.8993** |
| **Latency (per Scene)**| 9.76 ms | 5.76 ms | 129 ms | **175 ms** (0.685 ms/tile) |
| **Peak Activation Memory**| ~15.8 MB | ~15.8 MB | 15.81 MB | **4.08 MB** |
| **Total Process Footprint**| 22.12 MB | 22.12 MB | 26.72 MB | **14.87 MB** (includes ~10.5 MB ORT runtime base) |

---

##  Release Matrix & Trade-Off Analysis

Every significant iteration was frozen alongside its measured metrics to trace progress and trade-offs.

### Version Comparison Matrix

| Version | Patch / Tile Size | Encoder | IoU | Memory Footprint | Latency (ms/scene) | Precision | Data Loss (30% Cloud Threshold) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | $256 \times 256$ | `mnv2_100` | 0.8663 | 27.27 MB | 129 ms | — | — |
| **v2** | $256 \times 256$ | `mnv2_050` | 0.8855 | 26.69 MB | 129 ms | 0.9264 | 16.17% |
| **v3** | $256 \times 256$ | `mnv2_050` | 0.8844 | 26.47 MB | 129 ms | 0.9531 | 16.17% |
| **v3.1**| $128 \times 128$ | `mnv2_050` | 0.8824 | 17.82 MB | 138 ms | 0.9442 | 13.78% |
| **v3.2**| **$64 \times 64$** | **`mnv2_050`** | **0.8807** | **14.87 MB** | **175 ms** | **0.9771** | **8.13%** |

### Metric Pareto Extremes

- **Highest Accuracy (IoU):** `v2` (0.8855) — *Note: Difference with v3.2 (-0.0037) is within training variance.*
- **Lowest Latency:** `v1` / `v2` / `v3` (129 ms/scene)
- **Lowest Memory Footprint:** **`v3.2` (14.87 MB)** — *44% reduction compared to v3.*
- **Highest Precision:** **`v3.2` (0.9771)**
- **Lowest Scientific Data Loss:** **`v3.2` (8.13% clean area lost)** — *Halved compared to v3.*
- **Best Balanced Compromise:** `v3.1` (128x128) — *2.4x less RAM, only 7% latency overhead.*

> **Deployment Decision (v3.2):** **v3.2** was selected as the primary production release. Because the latency budget was ample (175 ms vs ESA $\Phi$-sat-1's 325 ms budget), the slight processing time trade-off was exchanged for significant RAM savings (44%) and halving scientific data loss.

---

## Operating Point & Downlink Analysis

Decision threshold tuned under **High-Precision Constraint ($\ge 0.995$)** on ambiguous cloud scenes ($2\% < \text{cloud} < 98\%$):

- **Threshold Value:** `0.7947`
- **Net Downlink Data Reduction:** **48.65%** (capturing 87% of theoretical maximum 56.15%)
- **Usable Scientific Data Loss:** **0.475%**
- **External Sensor Validation (Landsat-8 / SPARCS):** Maintained **0.9749 ROC-AUC** despite systematic uncorrected solar elevation reflectance shifts.

---

## File & Model Artifact Mapping

```text
├── releases/
│   └── v3.2/
│       ├── unet_int8.onnx         # Primary deployable INT8 model (0.96 MB, 64x64)
│       ├── classifier_int8.onnx   # Backup INT8 classifier (2.59 MB, 256x256)
│       └── operating_points.json  # Measured operating thresholds and sensitivity trade-offs
├── docs/
│   ├── TEKNIK_RAPOR.pdf          # Full technical docket, quantization failure bisect analyses
│   └── STAJ_RAPORU.pdf           # Final academic report
└── README.md
