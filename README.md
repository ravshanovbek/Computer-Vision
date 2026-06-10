# Image Colorization

> A convolutional encoder–decoder network that restores color to greyscale images.
> **Computer Vision Project.**

---

## Overview

Image colorization is the task of predicting plausible RGB values for a greyscale input. It requires the model to *recognise* what's in the image, *understand* its context, and *infer* colors that look natural to a human observer.

This project implements a compact, U-Net-inspired CNN trained end-to-end with pixel-wise MSE loss. It demonstrates the full pipeline — data preprocessing, model design, training, and evaluation — on a modest compute budget.

### Why it matters

| Domain | Application |
|---|---|
| **Historical restoration** | Bringing old photographs, archival imagery, and family memories back to life. |
| **Film & media** | Restoring classic black-and-white films, television footage, and documentaries. |
| **Scientific imaging** | Enhancing greyscale captures (microscopy, satellite imagery, medical scans) to surface details and aid interpretation. |

---

## Dataset

**Source:** [Google Universal Image Embeddings — 128×128 (Kaggle)](https://www.kaggle.com/datasets/rhtsingh/google-universal-image-embeddings-128x128?select=128x128)

### Categories retained for training

| Category | Original source |
|---|---|
| Apparel | Deep Fashion Dataset |
| Artwork | Google-scraped |
| Cars | Stanford Cars Dataset |
| Dishes | Google-scraped |
| Furniture | Google-scraped |
| Illustrations | Google-scraped |
| Landmark | Google Landmark Dataset |
| Meme | Google-scraped |

The original dataset also contains *Packaged*, *Storefronts*, and *Toys* categories. These were dropped to reduce training time and keep the task tractable on limited compute.

### Preprocessing pipeline

1. **Color images (128×128)** — source RGB samples from the Kaggle dataset.
2. **Convert to greyscale** — the greyscale image becomes the network input.
3. **Pair (grey → color)** — the original RGB image is the supervision target.
4. **Split 80 / 20** — training and test splits, same fraction sampled.

> `DATA_FRACTION = 0.03` — only **3 %** of the data is sampled per run to enable rapid iteration on limited GPU resources. The same fraction is applied to both train and test splits to keep the evaluation comparable.

---

## Architecture

A symmetric encoder–decoder built from `Conv2d + ReLU` layers. The encoder compresses spatial resolution while expanding channels; the decoder upsamples back to RGB.

```
Input  3 × 128 × 128
   │
   │  ENCODER  (Conv2d + ReLU, stride 2 then 1)
   ▼  64 → 128 → 256 → 512 → 512 → 256
   │
   │  DECODER  (Conv2d + ReLU + Upsample)
   ▼  128 → 64 → 32 → 16 → 3   + Sigmoid
   │
Output  3 × 128 × 128
```

### Model definition

```python
import torch.nn as nn

class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   64,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),           nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,  32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32,  16,  3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  3,   3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

### Why ReLU

`ReLU(x) = max(0, x)` is applied after every `Conv2d` layer. It introduces non-linearity, keeps gradients alive for positive activations, and zeroes-out negatives — making training fast and stable without saturating gradients.

### Why Sigmoid at the output

The final layer uses `Sigmoid` so the predicted RGB values stay in the `[0, 1]` range — matching the normalized targets and avoiding the need to clip outputs at inference time.

---

## Training

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `EPOCHS` | 50 | Full training schedule |
| `BATCH_SIZE` | 16 | Per training step |
| `LR` | `1e-3` | Adam optimizer, constant |
| `CHECKPOINT_EVERY` | 10 | Save model + evaluate |
| `SAMPLE_EVERY` | 5 | Save preview images |
| Benchmark cadence | every 5 epochs | Posted to track progress |
| `NUM_WORKERS` | 0 | DataLoader workers |
| `DATA_FRACTION` | 0.03 | 3 % of total data |

### Loss function — MSE

The network is trained to minimize **Mean Squared Error** between its predicted RGB output and the original color image:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

where $y_i$ is the true pixel value, $\hat{y}_i$ is the predicted pixel, and $N$ is the number of pixels. The loss averages squared per-pixel error across all three RGB channels — penalizing large color deviations more strongly than small ones.

---

## Evaluation

Two metrics are tracked across training:

### MSE (Mean Squared Error)
- **What** — the training objective itself.
- **Why** — direct signal of optimization. If train and test curves stay close, the model is generalizing rather than memorizing.

### PSNR (Peak Signal-to-Noise Ratio, dB)
- **What** — standard image-quality metric. $\text{PSNR} = 10 \cdot \log_{10}(\text{MAX}^2 / \text{MSE})$. Higher is better.
- **Why** — log-scale, perceptual view of reconstruction quality. PSNR is the de-facto benchmark in image-restoration literature, making results comparable across studies.

### Observed convergence (50 epochs)

| Metric | Start | End |
|---|---|---|
| **MSE loss (test)** | ≈ 0.016 | ≈ 0.008 |
| **PSNR (test)** | ≈ 18.1 dB | ≈ 21.2 dB |

Train and test MSE curves track each other closely throughout training, suggesting minimal overfitting at this data fraction.

---

## Project structure

```
.
├── README.md
├── data/                   # downloaded Kaggle subset (gitignored)
├── checkpoints/            # saved model weights (every 10 epochs)
├── samples/                # preview images (every 5 epochs)
├── model.py                # ColorizationModel definition
├── dataset.py              # greyscale-pair Dataset + DataLoader
├── train.py                # training loop + checkpointing
└── evaluate.py             # MSE / PSNR computation on test split
```

---

## Getting started

```bash
# 1. Install dependencies
pip install torch torchvision pillow numpy matplotlib tqdm

# 2. Download the Kaggle dataset
#    https://www.kaggle.com/datasets/rhtsingh/google-universal-image-embeddings-128x128
#    Place the 128x128 folder under ./data/

# 3. Train
python train.py

# 4. Evaluate
python evaluate.py --checkpoint checkpoints/epoch_50.pt
```

---

## Limitations & future work

- **No skip connections.** The architecture is U-Net *inspired* but does not include the cross-encoder skip connections that give the original U-Net its precise spatial localization. Adding them would likely improve fine detail.
- **3 % data fraction.** Trained on a subset for compute reasons. Scaling to the full dataset is a natural next step.
- **Per-pixel MSE.** MSE produces smooth, *safe* colors and tends to desaturate uncertain regions. Perceptual or adversarial losses (e.g., GAN-based) typically yield more vivid outputs at the cost of training complexity.
- **RGB target space.** Many colorization works predict in Lab space (only the *a* and *b* channels), since the L-channel can be reused directly from the greyscale input. This is a documented improvement worth exploring.

---

## References

1. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. Medical Image Computing and Computer-Assisted Intervention (MICCAI), LNCS vol. 9351, pp. 234–241. Springer.
   <https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28>
   *Personal note — studied during my undergraduate coursework; this volume is where I was first introduced to the U-Net architecture.*

2. **Hu, Z., Shkurat, O., & Kasner, M.** (2024). *Grayscale Image Colorization Method Based on U-Net Network*. International Journal of Image, Graphics and Signal Processing (IJIGSP), 16(2), 70–82. DOI: [10.5815/ijigsp.2024.02.06](https://doi.org/10.5815/ijigsp.2024.02.06)
   <https://www.mecs-press.org/ijigsp/ijigsp-v16-n2/IJIGSP-V16-N2-6.pdf>

---

## License

Released under the MIT License unless stated otherwise. Dataset is subject to its original Kaggle terms.
