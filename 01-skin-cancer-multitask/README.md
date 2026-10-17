# 01 — Skin Cancer Classification & Segmentation

Multi-task learning on the **HAM10000** dataset (10,015 dermatoscopic images,
7 lesion classes) — classification with a fine-tuned ResNet-18, lesion
segmentation with a U-Net built from scratch, and a shared-encoder multi-task
model that does both from a single forward pass.

Full write-up with per-class metrics, curves, and confusion matrices:
[report.pdf](report.pdf)

## What's inside

`skin_cancer_multitask.ipynb` walks the full pipeline:

- **Data** — `HAM10000Dataset` (torchvision `Dataset`) serving both tasks from
  one dataframe; lazy image loading, 256×256 resize, stratified 70/15/15
  train/val/test split to handle the severe class imbalance (NV alone is ~67%
  of the data).
- **Classification** — `ResNet-18` (ImageNet weights) with a replaced 7-way FC
  head, early stopping on validation loss.
- **Segmentation** — `UNet` implemented from scratch in PyTorch (encoder
  blocks → bottleneck → ConvTranspose decoder with skip connections), BCE loss,
  IoU and Dice tracked per epoch.
- **Multi-task model** — shared ResNet-18 encoder with a branched decoder:
  an avg-pool → FC classification head, and a ConvTranspose segmentation head
  (no U-Net skip connections — see results below for what that costs).

## Results (test set)

| Model | Classification Acc | Segmentation IoU | Dice |
|---|---|---|---|
| ResNet-18 (cls only) | **84.3%** | — | — |
| U-Net (seg only) | — | **0.862** | **0.917** |
| Multi-task (shared encoder) | **84.5%** | 0.473 | 0.622 |

Per-class F1 (single-task classifier): NV 0.92, VASC 0.79, DF 0.74, BKL 0.64,
BCC 0.81, MEL 0.61, AKIEC 0.51 — minority classes suffer from the imbalance.

**Interesting trade-off:** the multi-task model *matches* single-task
classification accuracy (84.5% vs 84.3%) while also segmenting — but its
segmentation is much weaker than the dedicated U-Net (IoU 0.47 vs 0.86),
because the branched decoder lacks U-Net skip connections that carry
high-resolution spatial detail.

## Dataset & running

Dataset: [HAM10000 Segmentation and Classification](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
on Kaggle (`GroundTruth.csv`, `images/`, `masks/`).

```bash
pip install -r ../../requirements.txt
export DATA_DIR=/path/to/ham1000-segmentation-and-classification
jupyter notebook skin_cancer_multitask.ipynb
```

## Credits

Team project (Pattern Recognition, Alexandria University, March 2025):
Judy Nabil, Amr Abdelaziz, Ahmed Hany, Mahmoud Haytham.
