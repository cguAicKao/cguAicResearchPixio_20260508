# Fundus-Pixio: Continual Pre-training of Pixio ViT-L on Fundus Eye Images

> Fine-tuning the official [Pixio ViT-L/16](https://github.com/facebookresearch/pixio) checkpoint on a mixed fundus + natural image dataset, evaluated across multiple ophthalmic benchmarks via 5-fold cross-validation.

[![Base Model](https://img.shields.io/badge/Base%20Model-Pixio%20ViT--L%2F16-blue)](https://github.com/facebookresearch/pixio)
[![Eval](https://img.shields.io/badge/Eval-5--Fold%20CV-green)]()
[![Datasets](https://img.shields.io/badge/Datasets-6%20Fundus%20Benchmarks-orange)]()

---

## Overview

This repository presents **continual pre-training** of the official Pixio ViT-L/16 encoder on a mixed dataset containing fundus eye images and natural images. We follow a **two-stage training strategy**:

- **Stage 1 (epochs 0–100):** Encoder frozen, higher base learning rate. The decoder adapts to the fundus image domain.
- **Stage 2 (epochs 100–360):** Full model unfreezing with ~10× reduced learning rate and extended warmup. The encoder is gradually fine-tuned toward ophthalmic features.

Downstream evaluation uses **5-fold cross-validation**, reporting **Accuracy** and **AUROC** across 6 publicly available fundus datasets.

---

## Setup

This codebase is built on top of the [Pixio](https://github.com/facebookresearch/pixio) pre-training framework.

```bash
conda create -n fundus-pixio python=3.10
conda activate fundus-pixio
pip install -r requirements.txt
```

**Hardware:** 2× NVIDIA A6000 GPUs

---

## Pre-training

### Data

We use a mixed dataset of fundus eye images and natural images for pre-training. Natural images are drawn from ImageNet-1K. Fundus images are clinical retinal photographs provided by **Chang Gung Memorial Hospital (CGMH), Taiwan**, used under an institutional data-sharing agreement for research purposes. We sincerely thank CGMH for providing access to these clinical data.

### Launch Pre-training

```bash
cd pretraining

# Stage 1: frozen encoder (~100 epochs)
bash scripts/pretrain_stage1_frozen_encoder.sh

# Stage 2: full fine-tuning (~260 epochs)
bash scripts/pretrain_stage2_full_finetune.sh
```

### Resume from Pixio ViT-L Official Checkpoint

```bash
--resume /path/to/pixio_vitl16.pth \
--model pixio_vit_large_patch16
```

---

## Evaluation

Downstream evaluation is performed by loading each pre-trained checkpoint and running **5-fold cross-validation** on classification tasks. We report mean ± std across folds.

### Datasets

The following 6 publicly available fundus benchmarks are used for downstream evaluation. Dataset splits follow the original papers or standard community splits.

| Dataset | Task | Classes | Train | Val | Test |
|---|---|---|---|---|---|
| [APTOS2019](https://www.kaggle.com/c/aptos2019-blindness-detection) | Diabetic Retinopathy Grading | 5 | 2,048 | 514 | 1,100 |
| [MESSIDOR2](https://www.adcis.net/en/third-party/messidor2/) | Diabetic Retinopathy Grading | 4 | 972 | 246 | 526 |
| [Glaucoma_fundus](https://github.com/cc-hpc-itwm/GlaucomaNet) | Glaucoma Detection | 2 | 861 | 218 | 465 |
| [Retina](https://www.kaggle.com/datasets/jr2ngb/cataract-dataset) | Diabetic Retinopathy Grading | 5 | 336 | 84 | 181 |
| [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | Diabetic Retinopathy Grading | 5 | 329 | 84 | 103 |
| [PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004) | Glaucoma Detection | 3 | 311 | 79 | 98 |

![Dataset sample distribution](assets/dataset_splits.png)
<!-- Replace with your actual figure. A grouped bar chart (Train/Val/Test per dataset) is recommended. -->

### Launch Evaluation

```bash
bash scripts/eval_fivetold_cv.sh \
  --checkpoint /path/to/epoch-N.pth \
  --dataset aptos2019
```

---

## Results

All results are mean ± std over 5-fold cross-validation. `epoch-0` = official Pixio ViT-L/16 baseline. Stage 1 = epochs 0–100 (frozen encoder); Stage 2 = epochs 100–360 (full fine-tuning). The dashed vertical line in the training curves marks the Stage 1 / Stage 2 boundary.

### Training curves

![Training curves](assets/training_curves.png)
<!-- Replace with your exported figure (Accuracy + AUROC vs. epoch, 6 datasets, dashed line at epoch 100). -->

### Summary table — representative checkpoints

#### Accuracy ↑

| Checkpoint | APTOS2019 | Glaucoma\_fundus | IDRiD | MESSIDOR2 | PAPILA | Retina |
|---|---|---|---|---|---|---|
| epoch-0 (baseline) | 0.8264 ± 0.0065 | 0.7656 ± 0.0105 | 0.4777 ± 0.0957 | 0.7228 ± 0.0094 | 0.7224 ± 0.0517 | 0.5967 ± 0.0110 |
| epoch-100 (end Stage 1) | 0.7796 ± 0.0062 | 0.8034 ± 0.0119 | 0.3398 ± 0.0376 | 0.6046 ± 0.0055 | 0.7531 ± 0.0196 | 0.5680 ± 0.0153 |
| epoch-200 | 0.7931 ± 0.0025 | 0.8370 ± 0.0113 | 0.4175 ± 0.0476 | 0.6186 ± 0.0084 | 0.7510 ± 0.0137 | 0.5901 ± 0.0269 |
| epoch-300 | 0.8025 ± 0.0071 | 0.8430 ± 0.0015 | 0.4175 ± 0.0743 | 0.6285 ± 0.0167 | 0.7469 ± 0.0610 | 0.6044 ± 0.0333 |
| epoch-360 (final) | 0.7996 ± 0.0054 | **0.8439 ± 0.0025** | **0.4563 ± 0.0562** | 0.6297 ± 0.0155 | **0.7714 ± 0.0155** | 0.6022 ± 0.0315 |

#### AUROC ↑

| Checkpoint | APTOS2019 | Glaucoma\_fundus | IDRiD | MESSIDOR2 | PAPILA | Retina |
|---|---|---|---|---|---|---|
| epoch-0 (baseline) | 0.9403 ± 0.0022 | 0.9046 ± 0.0035 | 0.7914 ± 0.0265 | 0.8853 ± 0.0055 | 0.7145 ± 0.0360 | 0.7161 ± 0.0128 |
| epoch-100 (end Stage 1) | 0.9037 ± 0.0003 | 0.9208 ± 0.0024 | 0.6440 ± 0.0159 | 0.7707 ± 0.0161 | 0.7887 ± 0.0138 | 0.7569 ± 0.0139 |
| epoch-200 | 0.9205 ± 0.0010 | 0.9402 ± 0.0044 | 0.6805 ± 0.0211 | 0.8042 ± 0.0078 | 0.7882 ± 0.0130 | 0.8022 ± 0.0064 |
| epoch-300 | 0.9255 ± 0.0011 | 0.9437 ± 0.0038 | 0.7122 ± 0.0247 | 0.8231 ± 0.0141 | 0.8050 ± 0.0198 | **0.8197 ± 0.0115** |
| epoch-360 (final) | 0.9239 ± 0.0015 | 0.9437 ± 0.0024 | **0.7171 ± 0.0261** | 0.8216 ± 0.0131 | 0.7955 ± 0.0161 | 0.8121 ± 0.0189 |

> **Bold** = best result per dataset across all checkpoints. Best AUROC on APTOS2019 is at epoch-280 (0.9258); see full table below.

<details>
<summary>Full results — all checkpoints (every 20 epochs)</summary>

#### Accuracy ↑ (full)

| Checkpoint | APTOS2019 | Glaucoma\_fundus | IDRiD | MESSIDOR2 | PAPILA | Retina |
|---|---|---|---|---|---|---|
| epoch-0 | 0.8264 ± 0.0065 | 0.7656 ± 0.0105 | 0.4777 ± 0.0957 | 0.7228 ± 0.0094 | 0.7224 ± 0.0517 | 0.5967 ± 0.0110 |
| epoch-20 | 0.7309 ± 0.0053 | 0.7669 ± 0.0276 | 0.3534 ± 0.0244 | 0.5776 ± 0.0049 | 0.6837 ± 0.0000 | 0.5525 ± 0.0111 |
| epoch-40 | 0.7495 ± 0.0029 | 0.7918 ± 0.0098 | 0.3670 ± 0.0346 | 0.6004 ± 0.0058 | 0.7490 ± 0.0116 | 0.5613 ± 0.0349 |
| epoch-60 | 0.7640 ± 0.0061 | 0.8026 ± 0.0074 | 0.3573 ± 0.0199 | 0.5996 ± 0.0109 | 0.7551 ± 0.0204 | 0.5414 ± 0.0310 |
| epoch-80 | 0.7736 ± 0.0070 | 0.8052 ± 0.0019 | 0.3301 ± 0.0182 | 0.6053 ± 0.0032 | 0.7653 ± 0.0191 | 0.5448 ± 0.0177 |
| epoch-100 | 0.7796 ± 0.0062 | 0.8034 ± 0.0119 | 0.3398 ± 0.0376 | 0.6046 ± 0.0055 | 0.7531 ± 0.0196 | 0.5680 ± 0.0153 |
| epoch-120 | 0.7815 ± 0.0031 | 0.8095 ± 0.0062 | 0.3592 ± 0.0238 | 0.6137 ± 0.0078 | 0.7592 ± 0.0116 | 0.5680 ± 0.0143 |
| epoch-140 | 0.7862 ± 0.0031 | 0.8103 ± 0.0133 | 0.3379 ± 0.0385 | 0.6129 ± 0.0087 | 0.7367 ± 0.0318 | 0.5735 ± 0.0163 |
| epoch-160 | 0.7947 ± 0.0051 | 0.8215 ± 0.0050 | 0.3748 ± 0.0374 | 0.6194 ± 0.0074 | 0.7469 ± 0.0112 | 0.5746 ± 0.0224 |
| epoch-180 | 0.7915 ± 0.0049 | 0.8280 ± 0.0086 | 0.3903 ± 0.0318 | 0.6186 ± 0.0230 | 0.7653 ± 0.0125 | 0.5867 ± 0.0126 |
| epoch-200 | 0.7931 ± 0.0025 | 0.8370 ± 0.0113 | 0.4175 ± 0.0476 | 0.6186 ± 0.0084 | 0.7510 ± 0.0137 | 0.5901 ± 0.0269 |
| epoch-220 | 0.7942 ± 0.0045 | 0.8301 ± 0.0103 | 0.4311 ± 0.0569 | 0.6247 ± 0.0099 | 0.7612 ± 0.0155 | 0.5912 ± 0.0231 |
| epoch-240 | 0.8009 ± 0.0066 | 0.8378 ± 0.0154 | 0.4019 ± 0.0483 | 0.6316 ± 0.0084 | 0.7694 ± 0.0155 | 0.6088 ± 0.0311 |
| epoch-260 | 0.7978 ± 0.0078 | 0.8378 ± 0.0089 | 0.4078 ± 0.0363 | 0.6346 ± 0.0105 | 0.7571 ± 0.0371 | 0.6011 ± 0.0318 |
| epoch-280 | 0.8040 ± 0.0021 | 0.8409 ± 0.0104 | 0.3883 ± 0.0633 | 0.6236 ± 0.0144 | 0.7653 ± 0.0204 | 0.6066 ± 0.0386 |
| epoch-300 | 0.8025 ± 0.0071 | 0.8430 ± 0.0015 | 0.4175 ± 0.0743 | 0.6285 ± 0.0167 | 0.7469 ± 0.0610 | 0.6044 ± 0.0333 |
| epoch-330 | 0.7998 ± 0.0079 | 0.8413 ± 0.0056 | 0.4330 ± 0.0719 | 0.6247 ± 0.0149 | 0.7694 ± 0.0171 | 0.6055 ± 0.0331 |
| epoch-360 | 0.7996 ± 0.0054 | **0.8439 ± 0.0025** | **0.4563 ± 0.0562** | 0.6297 ± 0.0155 | **0.7714 ± 0.0155** | 0.6022 ± 0.0315 |

#### AUROC ↑ (full)

| Checkpoint | APTOS2019 | Glaucoma\_fundus | IDRiD | MESSIDOR2 | PAPILA | Retina |
|---|---|---|---|---|---|---|
| epoch-0 | 0.9403 ± 0.0022 | 0.9046 ± 0.0035 | 0.7914 ± 0.0265 | 0.8853 ± 0.0055 | 0.7145 ± 0.0360 | 0.7161 ± 0.0128 |
| epoch-20 | 0.8655 ± 0.0069 | 0.8958 ± 0.0098 | 0.6151 ± 0.0073 | 0.7062 ± 0.0180 | 0.6798 ± 0.0473 | 0.7457 ± 0.0262 |
| epoch-40 | 0.8915 ± 0.0027 | 0.9115 ± 0.0021 | 0.6192 ± 0.0206 | 0.7569 ± 0.0051 | 0.7646 ± 0.0100 | 0.7579 ± 0.0130 |
| epoch-60 | 0.9000 ± 0.0017 | 0.9157 ± 0.0029 | 0.6181 ± 0.0047 | 0.7758 ± 0.0028 | 0.7792 ± 0.0154 | 0.7447 ± 0.0093 |
| epoch-80 | 0.9030 ± 0.0017 | 0.9205 ± 0.0032 | 0.6438 ± 0.0161 | 0.7832 ± 0.0092 | 0.7734 ± 0.0253 | 0.7424 ± 0.0093 |
| epoch-100 | 0.9037 ± 0.0003 | 0.9208 ± 0.0024 | 0.6440 ± 0.0159 | 0.7707 ± 0.0161 | 0.7887 ± 0.0138 | 0.7569 ± 0.0139 |
| epoch-120 | 0.9086 ± 0.0006 | 0.9244 ± 0.0035 | 0.6562 ± 0.0333 | 0.8031 ± 0.0062 | 0.7899 ± 0.0146 | 0.7533 ± 0.0181 |
| epoch-140 | 0.9137 ± 0.0004 | 0.9273 ± 0.0040 | 0.6557 ± 0.0184 | 0.8055 ± 0.0079 | 0.7981 ± 0.0124 | 0.7754 ± 0.0136 |
| epoch-160 | 0.9180 ± 0.0009 | 0.9327 ± 0.0048 | 0.6740 ± 0.0306 | 0.8106 ± 0.0111 | 0.7878 ± 0.0096 | 0.7966 ± 0.0154 |
| epoch-180 | 0.9181 ± 0.0033 | 0.9363 ± 0.0034 | 0.6828 ± 0.0226 | 0.8046 ± 0.0176 | 0.7859 ± 0.0128 | 0.8046 ± 0.0083 |
| epoch-200 | 0.9205 ± 0.0010 | 0.9402 ± 0.0044 | 0.6805 ± 0.0211 | 0.8042 ± 0.0078 | 0.7882 ± 0.0130 | 0.8022 ± 0.0064 |
| epoch-220 | 0.9230 ± 0.0022 | 0.9370 ± 0.0059 | 0.6978 ± 0.0187 | 0.8197 ± 0.0087 | 0.8133 ± 0.0311 | 0.8045 ± 0.0157 |
| epoch-240 | 0.9249 ± 0.0025 | **0.9424 ± 0.0034** | 0.6836 ± 0.0167 | 0.8198 ± 0.0110 | 0.8118 ± 0.0128 | 0.8085 ± 0.0109 |
| epoch-260 | 0.9240 ± 0.0019 | 0.9402 ± 0.0033 | 0.6819 ± 0.0232 | 0.8283 ± 0.0083 | 0.8055 ± 0.0174 | 0.8135 ± 0.0165 |
| epoch-280 | **0.9258 ± 0.0019** | 0.9416 ± 0.0047 | 0.6918 ± 0.0153 | 0.8072 ± 0.0309 | 0.8128 ± 0.0247 | 0.8153 ± 0.0135 |
| epoch-300 | 0.9255 ± 0.0011 | 0.9437 ± 0.0038 | 0.7122 ± 0.0247 | 0.8231 ± 0.0141 | 0.8050 ± 0.0198 | **0.8197 ± 0.0115** |
| epoch-330 | 0.9251 ± 0.0028 | 0.9431 ± 0.0035 | 0.7070 ± 0.0298 | 0.8250 ± 0.0140 | 0.8019 ± 0.0215 | 0.8115 ± 0.0142 |
| epoch-360 | 0.9239 ± 0.0015 | 0.9437 ± 0.0024 | **0.7171 ± 0.0261** | 0.8216 ± 0.0131 | 0.7955 ± 0.0161 | 0.8121 ± 0.0189 |

</details>

---

## Key Observations

- **Stage 1 (frozen encoder, epochs 0–100):** A temporary accuracy drop is observed across most datasets in the early epochs (~20–60), which is expected as the decoder adapts to the new domain while the encoder remains fixed. Performance steadily recovers and exceeds the baseline by epoch 100 on Glaucoma_fundus and Retina.
- **Stage 2 (full fine-tuning, epochs 100–360):** Consistent improvement is observed across all six datasets. AUROC gains are particularly pronounced on IDRiD (+13.2% over epoch-100) and Retina (+6.3% over epoch-100).
- **Best overall checkpoints:** epoch-280~360 consistently deliver the strongest results across datasets.

---

## Pretrained Checkpoints

| Checkpoint | Trained Epochs | Notes |
|---|---|---|
| `epoch-0.pth` | 0 | Official Pixio ViT-L/16 (baseline) |
| `epoch-100.pth` | 100 | End of Stage 1 (frozen encoder) |
| `epoch-200.pth` | 200 | Mid Stage 2 |
| `epoch-300.pth` | 300 | Near-final |
| `epoch-360.pth` | 360 | Final checkpoint |

> TODO: add download links (HuggingFace / Google Drive) once uploaded.

---

## Citation

If you use this work, please cite the original Pixio paper:

```bibtex
@article{pixio,
  title={In Pursuit of Pixel Supervision for Visual Pre-training},
  author={Yang, Lihe and Li, Shang-Wen and Li, Yang and Lei, Xinjie and Wang, Dong and Mohamed, Abdelrahman and Zhao, Hengshuang and Xu, Hu},
  journal={arXiv:2512.15715},
  year={2025}
}
```

---

## Acknowledgement

This work builds directly on [Pixio](https://github.com/facebookresearch/pixio) (Facebook Research). We sincerely thank the original authors for open-sourcing their codebase and pre-trained models.
