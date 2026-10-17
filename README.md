# Pattern Recognition Projects

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Python 3](https://img.shields.io/badge/python-3-3776ab?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-notebooks-f37626?logo=jupyter&logoColor=white)

![PyTorch](https://img.shields.io/badge/pytorch-2-ee4c2c?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/tensorflow-2-ff6f00?logo=tensorflow&logoColor=white)
![timm](https://img.shields.io/badge/vit-timm-9cf)
![Kaggle](https://img.shields.io/badge/kaggle-datasets-20beff?logo=kaggle&logoColor=white)

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

Both projects were a team effort at Alexandria University (March–May 2025):

- **Judy Nabil**
- **Amr Abdelaziz**
- **Ahmed Hany**
- **Mahmoud Haytham**
