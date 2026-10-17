# 02 — Neural Style Transfer

Two approaches to neural style transfer, implemented in both PyTorch and
TensorFlow:

1. **Feed-forward transfer network** (`style_transfer.py`,
   `notebooks/style_transfer_pytorch.ipynb`) — a *learned* stylization model:
   frozen VGG-19 encoder → **AdaIN** (adaptive instance normalization) →
   trained convolutional decoder. A scalar threshold `t` blends the AdaIN'ed
   features with the raw content features, so a single trained model produces
   a continuous stylization dial `t ∈ [0, 1]` at inference time — no
   per-image optimization needed.
2. **Per-image optimization** (`notebooks/style_transfer_tf.ipynb`) — the
   classic Gatys et al. approach: optimize the pixels of a generated image to
   match VGG-19 content features + Gram-matrix style statistics, implemented
   in TensorFlow/Keras.

Full write-up: [report.pdf](report.pdf)

## Feed-forward model (`StyleNet`)

```
content ──► VGG-19 enc ──► cf ─┐
                               ├─► AdaIN(cf, sf) ─► t·AdaIN + (1−t)·cf ─► decoder ─► stylized image
style ────► VGG-19 enc ──► sf ─┘
```

- **Training data** — 7k content images (Kaggle `image-classification`
  dataset) + 3k style images (WikiArt), stratified-sampled and split
  80/10/10.
- **Loss** — perceptual losses on VGG-19 features: content loss = MSE on
  feature maps; style loss = MSE on Gram matrices.
- **Threshold conditioning** — `t` is sampled randomly per training pair
  (`threshold_mode='random'`), teaching the decoder the full interpolation
  curve; a post-training sweep over `t ∈ {0.0, …, 1.0}` picks the best
  trade-off on the test set.
- **Early stopping + checkpointing** — best weights saved by val loss;
  `torch.jit` traced export for deployment.
- **Serving sketch** — `create_web_app()` in `style_transfer.py` wraps the
  model in a Flask `/transfer` endpoint (content + style upload → base64
  stylized image).

## ViT variant

`notebooks/style_transfer_vit.ipynb` (and `VitStyleNet` in the script) swap
the VGG encoder for `vit_base_patch16_224` (timm): the CLS token is projected
through an adapter back to a 512×4×4 feature map, then the same AdaIN +
decoder pipeline applies.

## Data & running

Datasets (Kaggle, not committed):

- Content: `image-classification` dataset (`images/images/<class>/…`)
- Style: `wikiart` (`<movement>/<artist>_<painting>.jpg`)

```bash
pip install -r ../../requirements.txt
export CONTENT_DIR=/path/to/image-classification/images/images
export STYLE_DIR=/path/to/wikiart
python style_transfer.py          # trains both CNN and ViT variants
```

Or open the notebooks:

- `notebooks/style_transfer_pytorch.ipynb` — feed-forward StyleNet (VGG-19)
- `notebooks/style_transfer_vit.ipynb` — ViT-encoder variant
- `notebooks/style_transfer_tf.ipynb` — per-image optimization (TensorFlow);
  point `CONTENT_IMG` / `STYLE_IMG` env vars (or the cell) at any two images

## Notes

- The feed-forward model trades a bit of fidelity for speed — one forward
  pass vs. hundreds of optimization steps per image — and `t` gives
  continuous control over stylization strength.
- ViT encoder path exists as an architecture experiment (bonus); the CNN
  variant was the primary submission.
