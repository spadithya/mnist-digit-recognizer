"""
Phase 1 — Load + visualize MNIST.

Goal: get MNIST onto disk, look at it, and sanity-check the shapes and
pixel-value ranges *before* we feed anything to a model. Each later phase
(logistic regression, NumPy MLP, PyTorch MLP, CNN) assumes the data looks
the way this script says it does.

Run from the project root:
    python notebooks/01_get_data.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — render to file only, never a window
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# Anchor paths to the project root so the script works no matter where
# it's invoked from.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "notebooks" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
# torchvision.datasets.MNIST returns PIL images by default. We deliberately
# skip transforms here — for visualization we want the raw 0..255 grayscale,
# not normalized tensors.
print("Downloading MNIST (cached after the first run)...")
train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=True)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True)

print(f"\nTrain set: {len(train_ds):>6} images")
print(f"Test set:  {len(test_ds):>6} images")


# ---------------------------------------------------------------------------
# 2. Inspect one sample
# ---------------------------------------------------------------------------
# Every later script treats images as fixed-size 28x28 grayscale tensors.
# Before we trust that, look at one and confirm.
img, label = train_ds[0]
arr = np.asarray(img)

print("\nFirst training sample")
print(f"  PIL size:  {img.size}              (W, H)")
print(f"  PIL mode:  {img.mode!r}                  (L = 8-bit grayscale)")
print(f"  array:     shape={arr.shape}  dtype={arr.dtype}")
print(f"  pixels:    min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}")
print(f"  label:     {label}")


# ---------------------------------------------------------------------------
# 3. Class distribution
# ---------------------------------------------------------------------------
# A classifier learns nothing useful if one class is missing from a split.
# MNIST is famously well-balanced; this is just the sanity check that it is.
train_labels = np.asarray(train_ds.targets)
test_labels  = np.asarray(test_ds.targets)

print("\nClass distribution")
print("  digit | train  | test")
for d in range(10):
    print(f"    {d}   | {(train_labels == d).sum():>5d}  | {(test_labels == d).sum():>4d}")


# ---------------------------------------------------------------------------
# 4. Visualize a 5x5 grid of random training images
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=0)  # reproducible sample
sample_idx = rng.choice(len(train_ds), size=25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, i in zip(axes.flat, sample_idx):
    img, label = train_ds[int(i)]
    ax.imshow(img, cmap="gray")
    ax.set_title(str(label), fontsize=10)
    ax.axis("off")
fig.suptitle("MNIST — 25 random training images", fontsize=12)
fig.tight_layout()

out_path = PLOTS_DIR / "01_sample_grid.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved sample grid → {out_path.relative_to(ROOT)}")
