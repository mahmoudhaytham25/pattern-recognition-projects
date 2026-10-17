# Pattern Recognition Projects

Deep-learning projects on medical imaging and neural style transfer —
multi-task PyTorch models, from-scratch architectures, and classic papers
reimplemented in both PyTorch and TensorFlow.

| # | Project | Techniques | |
|---|---|---|---|
| 01 | [Skin Cancer Classification & Segmentation](01-skin-cancer-multitask/) | ResNet-18 transfer learning, from-scratch U-Net, shared-encoder multi-task model, IoU/Dice, HAM10000 | [notebook](01-skin-cancer-multitask/skin_cancer_multitask.ipynb) · [report](01-skin-cancer-multitask/report.pdf) |
| 02 | [Neural Style Transfer](02-neural-style-transfer/) | Feed-forward AdaIN style network (VGG-19 + learned decoder), controllable stylization strength, Gram-matrix perceptual losses, ViT variant (timm), per-image Gatys optimization in TensorFlow | [script](02-neural-style-transfer/style_transfer.py) · [notebooks](02-neural-style-transfer/notebooks/) · [report](02-neural-style-transfer/report.pdf) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets are downloaded from Kaggle and pointed at via env vars (`DATA_DIR`,
`CONTENT_DIR`, `STYLE_DIR`) — see each project's README for details. Notebooks
keep their executed outputs from the original training runs.

## Credits

Project 01 was a team effort — see its README for the full list.
