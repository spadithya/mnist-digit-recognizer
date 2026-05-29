"""
Phase 4 — The same 2-layer MLP as Phase 3, written in PyTorch.

The math is identical to Phase 3. The point is to swap your hand-written
forward/backward/SGD for the PyTorch idioms that show up in every modern
project. After this script, `nn.Module`, `DataLoader`, and the five-line
training step should feel routine.

The mapping (left = Phase 3, right = PyTorch primitive):

    W1 @ x + b1                      →  nn.Linear(784, 128)
    np.maximum(0, z1)                →  nn.ReLU()
    softmax(z2) + cross_entropy(...) →  nn.CrossEntropyLoss   (fused for stability!)
    backward()  (whole function)     →  loss.backward()       (autograd)
    W -= lr * dW   (×4 lines)        →  optimizer.step()
    perm[b*BS:(b+1)*BS] slicing      →  DataLoader(shuffle=True)

Upgrades over Phase 3 (small ones, targeting the slight overfit we saw):
- Dropout (p=0.2): zero 20% of hidden activations during training. Forces
  the network not to rely on any single hidden unit.
- Weight decay (1e-4): L2 penalty on every parameter, applied by the optimizer.
- SGD with momentum (0.9): smooths the gradient with an exponentially-weighted
  average of recent gradients. Converges faster than plain SGD.

Expected: ~97-98% test accuracy. The architecture is unchanged from Phase 3
so the gains come purely from the three regularization/optimization upgrades.

Run from the project root:
    python notebooks/04_pytorch_neural_net.py
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "notebooks" / "plots"
MODELS_DIR = ROOT / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility — seed the RNGs PyTorch uses internally for weight init,
# shuffling, and dropout. Without this, two runs would differ slightly.
torch.manual_seed(0)

# Canonical PyTorch idiom: write `device` once, then move every tensor with
# `.to(device)`. On your SSH VM this resolves to "cpu"; on a GPU box it
# would resolve to "cuda" with zero code changes elsewhere.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ===========================================================================
# 1. Data — DataLoader replaces our manual permute-and-slice
# ===========================================================================
# transforms.ToTensor() does TWO things in one step:
#   - PIL image (H, W) uint8 → torch tensor (1, H, W) float32
#   - Divides by 255 so values land in [0, 1]
# That's the normalization we did by hand in Phase 3.
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=False, transform=transform)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)

BATCH_SIZE = 64
# shuffle=True for train (replaces our rng.permutation each epoch)
# shuffle=False for test (order doesn't matter and stability helps debugging)
# num_workers=0 keeps loading single-process; fine for MNIST, and dodges
# the multiprocessing overhead that hurts on small datasets / VMs.
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}   Test batches: {len(test_loader)}")


# ===========================================================================
# 2. Model — the SAME 2-layer MLP as Phase 3
# ===========================================================================
# Architecture:  Flatten → Linear(784, 128) → ReLU → Dropout → Linear(128, 10)
#
# CRITICAL: NO softmax at the end. nn.CrossEntropyLoss does log-softmax + NLL
# internally, fused for numerical stability. If you also put a Softmax layer
# at the end of the model, you'd be applying it twice and silently break
# training. This is the single most common PyTorch beginner bug.
class MLP(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                  # (B, 1, 28, 28) → (B, 784)
            nn.Linear(784, hidden),        # = W1 @ x + b1
            nn.ReLU(),                     # = np.maximum(0, z1)
            nn.Dropout(dropout),           # zero 20% of hidden units (train only)
            nn.Linear(hidden, 10),         # = W2 @ a1 + b2   (NO softmax here)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(hidden=128, dropout=0.2).to(device)
print(f"\nModel:\n{model}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# 3. Loss and optimizer
# ===========================================================================
loss_fn = nn.CrossEntropyLoss()

# SGD with momentum=0.9 and weight_decay=1e-4 (L2 penalty).
# Plain Phase-3 SGD would be `momentum=0.0, weight_decay=0.0`.
# THE EQUATIONS
# v = 0.9*v + grad -> Blend new grad into running velocity
# W -= 0.1*(v + (1e-4*W) ->take a step agains the velocity,
#				scaled by lr, and apply the L2
#				pull towards zero (weight_decay)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)


# ===========================================================================
# 4. Training loop
# ===========================================================================
@torch.no_grad()
def evaluate(loader):
    """Run the model in eval mode and return (predictions, labels, accuracy).

    Two things matter here:
    - `model.eval()` switches Dropout into "pass-through" mode. Forgetting
      this is another classic beginner bug — it makes inference random.
    - `@torch.no_grad()` disables gradient tracking. We don't need it for
      eval, and turning it off makes the forward pass faster and uses less
      memory.
    """
    model.eval()
    all_pred, all_true = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return y_pred, y_true, accuracy_score(y_true, y_pred)


EPOCHS = 10
train_losses, test_accs = [], []

print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
      f"lr=0.1, momentum=0.9, weight_decay=1e-4, dropout=0.2\n")

t_start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()                          # enables dropout
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # ---- The five-line PyTorch training step ----
        logits = model(X_batch)            # forward (replaces our manual forward())
        loss = loss_fn(logits, y_batch)    # softmax + cross-entropy (fused)

        optimizer.zero_grad()              # clear stale gradients (else they accumulate)
        loss.backward()                    # autograd computes ALL gradients
        optimizer.step()                   # SGD update — replaces W -= lr*dW ×4

        epoch_loss += loss.item()

    _, _, test_acc = evaluate(test_loader)
    train_losses.append(epoch_loss / len(train_loader))
    test_accs.append(test_acc)
    print(f"epoch {epoch:2d}/{EPOCHS}   train_loss={train_losses[-1]:.4f}   test_acc={test_acc:.4f}")

print(f"\nTotal training time: {time.perf_counter() - t_start:.1f}s")


# ===========================================================================
# 5. Final evaluation
# ===========================================================================
y_pred, y_true, acc = evaluate(test_loader)
print(f"\nFinal test accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
print("\nPer-class report:")
print(classification_report(y_true, y_pred, digits=3))


# ===========================================================================
# 6. Save the trained weights
# ===========================================================================
# The PyTorch idiom is to save the `state_dict` (a plain dict of tensors),
# not the model class itself. Re-loading later means: instantiate the same
# MLP class, then `model.load_state_dict(torch.load(path))`.
weights_path = MODELS_DIR / "04_pytorch_mlp.pt"
torch.save(model.state_dict(), weights_path)
print(f"\nSaved model weights → {weights_path.relative_to(ROOT)}")


# ===========================================================================
# 7. Plots
# ===========================================================================
# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs_axis = np.arange(1, EPOCHS + 1)
ax1.plot(epochs_axis, train_losses, marker="o")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
ax1.set_title("Training loss"); ax1.grid(True, alpha=0.3)
ax2.plot(epochs_axis, test_accs, marker="o", color="C2")
ax2.set_xlabel("epoch"); ax2.set_ylabel("test accuracy")
ax2.set_title("Test accuracy"); ax2.grid(True, alpha=0.3)
fig.suptitle(f"PyTorch MLP — final test accuracy {acc:.3f}")
fig.tight_layout()
curves_path = PLOTS_DIR / "04_training_curves.png"
fig.savefig(curves_path, dpi=120, bbox_inches="tight")
print(f"Saved training curves    → {curves_path.relative_to(ROOT)}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
    ax=ax, cmap="Blues", colorbar=True, values_format="d"
)
ax.set_title(f"PyTorch MLP — confusion matrix (acc = {acc:.3f})")
fig.tight_layout()
cm_path = PLOTS_DIR / "04_confusion_matrix.png"
fig.savefig(cm_path, dpi=120, bbox_inches="tight")
print(f"Saved confusion matrix   → {cm_path.relative_to(ROOT)}")

# Misclassified
wrong_idx = np.where(y_pred != y_true)[0]
rng = np.random.default_rng(seed=0)
sample = rng.choice(wrong_idx, size=min(25, len(wrong_idx)), replace=False)

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, i in zip(axes.flat, sample):
    img, _ = test_ds[int(i)]                # img: tensor (1, 28, 28) float in [0, 1]
    ax.imshow(img.squeeze().numpy(), cmap="gray")
    ax.set_title(f"true={y_true[i]}  pred={y_pred[i]}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"25 misclassified test images (of {len(wrong_idx)} total errors)")
fig.tight_layout()
miss_path = PLOTS_DIR / "04_misclassified.png"
fig.savefig(miss_path, dpi=120, bbox_inches="tight")
print(f"Saved misclassified grid → {miss_path.relative_to(ROOT)}")
