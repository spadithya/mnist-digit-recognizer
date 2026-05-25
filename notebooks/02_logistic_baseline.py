"""
Phase 2 — Logistic regression baseline.

Goal: a credible floor. Every later model (NumPy MLP, PyTorch MLP, CNN) has
to beat this number. README target: ~92% test accuracy.

What's new vs the housing project:
- Classification, not regression. Output is one of 10 classes, not a number.
- Multi-class via *softmax* logistic regression (sklearn calls this the
  "multinomial" form; lbfgs solves it directly). One model, 10 outputs —
  not 10 separate one-vs-rest classifiers.
- Image data → flat feature vector. We reshape each 28x28 image to 784
  numbers, deliberately throwing away the 2D spatial structure. Phase 5's
  CNN reclaims it; here we want to see what a purely-linear model can do
  with raw pixels as independent features.
- Pixel normalization (÷ 255). Features now live in [0, 1], which makes
  the loss surface much friendlier and lbfgs converges faster.
- Classification metrics: accuracy, per-class precision/recall, confusion
  matrix. MAE/RMSE from the housing project do not apply.

Run from the project root:
    python notebooks/02_logistic_baseline.py
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")  # headless backend — file output only
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "notebooks" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load (cached by Phase 1, no re-download)
# ---------------------------------------------------------------------------
train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=False)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=False)


# ---------------------------------------------------------------------------
# 2. Reshape and normalize: (N, 28, 28) uint8 → (N, 784) float32 in [0, 1]
# ---------------------------------------------------------------------------
# train_ds.data is a torch.uint8 tensor of the raw image bytes — bypassing
# the PIL/transform pipeline gives us the underlying array directly.
X_train = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = train_ds.targets.numpy()
X_test  = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test  = test_ds.targets.numpy()

print(f"X_train: {X_train.shape}  {X_train.dtype}   range [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"y_train: {y_train.shape}  {y_train.dtype}   classes={np.unique(y_train).tolist()}")
print(f"X_test:  {X_test.shape}   {X_test.dtype}")
print(f"y_test:  {y_test.shape}   {y_test.dtype}")


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
# solver='lbfgs' — quasi-Newton optimizer; handles multinomial softmax natively.
# max_iter=1000 — lbfgs's default of 100 is too few for 60k×784; you'll get a
#   ConvergenceWarning. 1000 is comfortable.
# C=1.0      — inverse regularization strength (default). Smaller = stronger
#   L2 penalty. Fine to leave at default for a baseline.
clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)

print("\nTraining logistic regression (CPU: expect ~1–3 minutes)...")
t0 = time.perf_counter()
clf.fit(X_train, y_train)
print(f"  trained in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}  ({acc * 100:.2f}%)")

# precision = of the times the model said "this is a 7", how often was it right?
# recall    = of the actual 7s in the test set, how many did the model catch?
# f1        = harmonic mean of the two.
print("\nPer-class report:")
print(classification_report(y_test, y_pred, digits=3))


# ---------------------------------------------------------------------------
# 5. Plot — confusion matrix
# ---------------------------------------------------------------------------
# Rows = true label, columns = predicted label. The diagonal is correct
# classifications; off-diagonal cells are the systematic confusions
# (the 4↔9 and 3↔5 cells are usually the worst).
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
    ax=ax, cmap="Blues", colorbar=True, values_format="d"
)
ax.set_title(f"Logistic regression — confusion matrix (acc = {acc:.3f})")
fig.tight_layout()
cm_path = PLOTS_DIR / "02_confusion_matrix.png"
fig.savefig(cm_path, dpi=120, bbox_inches="tight")
print(f"\nSaved confusion matrix       → {cm_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 6. Plot — a grid of misclassified test images
# ---------------------------------------------------------------------------
wrong_idx = np.where(y_pred != y_test)[0]
rng = np.random.default_rng(seed=0)
sample = rng.choice(wrong_idx, size=min(25, len(wrong_idx)), replace=False)

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, i in zip(axes.flat, sample):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"true={y_test[i]}  pred={y_pred[i]}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"25 misclassified test images (of {len(wrong_idx)} total errors)")
fig.tight_layout()
miss_path = PLOTS_DIR / "02_misclassified.png"
fig.savefig(miss_path, dpi=120, bbox_inches="tight")
print(f"Saved misclassified grid     → {miss_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 7. Plot — what the model learned (per-class weight templates)
# ---------------------------------------------------------------------------
# clf.coef_ has shape (10, 784) — one weight per class per input pixel.
# Reshape each class's weights back to 28x28 and you can SEE what the
# model thinks each digit looks like. Red = pixel pushes toward this class,
# blue = pushes away. This visualization only works for linear models;
# enjoy it now because the MLP weights in Phase 3 won't be readable.
weights = clf.coef_.reshape(10, 28, 28)
vmax = np.abs(weights).max()  # symmetric color scale so 0 = neutral

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for d, ax in enumerate(axes.flat):
    ax.imshow(weights[d], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"class {d}", fontsize=10)
    ax.axis("off")
fig.suptitle("Learned weights per class (red = pro, blue = con)")
fig.tight_layout()
w_path = PLOTS_DIR / "02_weight_templates.png"
fig.savefig(w_path, dpi=120, bbox_inches="tight")
print(f"Saved weight templates       → {w_path.relative_to(ROOT)}")
