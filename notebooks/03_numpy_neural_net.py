"""
Phase 3 — Two-layer neural network, written from scratch in NumPy.

This is the most important script in the curriculum. Once you've written
the forward pass, the cross-entropy loss, and (especially) the backward
pass by hand, backpropagation stops being magic.

Architecture:

    Input (784)  --W1,b1-->  Hidden (128)  --ReLU-->  --W2,b2-->  Output (10)  --softmax-->  probabilities

    784 numbers          128 numbers                 10 numbers              10 numbers summing to 1
    (one per pixel)      (learned features)          (one score per class)   (predicted distribution)

Trained with:
- Cross-entropy loss
- Mini-batch SGD (batch_size=64)
- He weight initialization (the right starting scale for ReLU networks)
- 10 epochs over the 60,000 training examples

Expected: ~97% test accuracy — roughly cutting Phase 2's ~7.5% error in half.

Run from the project root:
    python notebooks/03_numpy_neural_net.py
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


# ===========================================================================
# 1. Data — same load + flatten + normalize as Phase 2
# ===========================================================================
train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=False)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=False)

X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_ds.targets.numpy().astype(np.int64)
X_test  = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test  = test_ds.targets.numpy().astype(np.int64)

print(f"X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}    y_test:  {y_test.shape}")


# ===========================================================================
# 2. Helpers — activation, output, loss, one-hot
# ===========================================================================
def relu(z):
    """ReLU activation: f(z) = max(0, z). Element-wise. The nonlinearity that
    lets the network learn things a linear model can't."""
    return np.maximum(0.0, z)


def relu_deriv(z):
    """Derivative of ReLU: 1 if z > 0, else 0. Used in the backward pass to
    pass the gradient through ReLU; pixels that were 'off' (z <= 0) block
    the gradient because they didn't contribute to the output."""
    return (z > 0).astype(z.dtype)


def softmax(z):
    """Softmax: turn 10 raw scores per row into a probability distribution
    summing to 1.

    Numerical trick: subtract the row-wise max before exp(). Mathematically
    identical (the normalization cancels it out) but prevents exp(big number)
    from overflowing to inf.

    Input:  z      shape (N, 10)
    Output: probs  shape (N, 10), each row sums to 1
    """
    z_shift = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_idx):
    """Negative log-likelihood loss. For each example, pick the predicted
    probability of the *correct* class and take -log of it. Average over
    the batch. Confident-and-right → low loss. Confident-and-wrong → huge loss.

    The +1e-12 prevents log(0) when the model gives a class probability of
    exactly zero (rare but possible early in training).

    Input:  probs  shape (N, 10),  y_idx shape (N,) of integers in [0, 9]
    Output: scalar loss
    """
    N = probs.shape[0]
    return -np.mean(np.log(probs[np.arange(N), y_idx] + 1e-12))


def one_hot(y_idx, num_classes=10):
    """Turn integer labels [3, 7, 0, ...] into row vectors:
        3 -> [0,0,0,1,0,0,0,0,0,0]
        7 -> [0,0,0,0,0,0,0,1,0,0]
    Needed only for the backward pass formula `probs - y_onehot`."""
    oh = np.zeros((y_idx.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y_idx.shape[0]), y_idx] = 1.0
    return oh


# ===========================================================================
# 3. Parameters — initialize weights and biases
# ===========================================================================
# Two weight matrices and two bias vectors:
#   W1: (784, 128)    b1: (128,)
#   W2: (128,  10)    b2: (10,)
#
# He initialization: for a ReLU layer with `fan_in` inputs, sample weights
# from N(0, sqrt(2 / fan_in)). This keeps the variance of activations
# roughly constant as data flows through the network — without it, signals
# either explode or vanish layer by layer and training gets stuck.
rng = np.random.default_rng(seed=0)
W1 = rng.standard_normal((784, 128)).astype(np.float32) * np.sqrt(2.0 / 784)
b1 = np.zeros(128, dtype=np.float32)
W2 = rng.standard_normal((128,  10)).astype(np.float32) * np.sqrt(2.0 / 128)
b2 = np.zeros(10, dtype=np.float32)


# ===========================================================================
# 4. Forward / backward / predict
# ===========================================================================
def forward(X):
    """Compute predictions AND save the intermediates we'll need for backprop.

    Returns probs and a `cache` dict so the backward pass can reuse z1, a1
    without recomputing.
    """
    z1 = X  @ W1 + b1     # (N, 128)  — pre-activation of hidden layer
    a1 = relu(z1)          # (N, 128)  — hidden activations
    z2 = a1 @ W2 + b2      # (N, 10)   — pre-activation of output (called "logits")
    probs = softmax(z2)    # (N, 10)   — class probabilities, each row sums to 1
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
    return probs, cache


def backward(cache, y_idx):
    """The whole point of this script. Compute gradients of the loss with
    respect to each parameter (W1, b1, W2, b2) using the chain rule.

    The math, layer by layer:

      Output layer (softmax + cross-entropy — paired for a reason):
        ∂L/∂z2 = (probs - y_onehot) / N         <-- this beautiful simplification
                                                    is why softmax and cross-entropy
                                                    are always used together
        ∂L/∂W2 = a1.T @ ∂L/∂z2
        ∂L/∂b2 = sum_over_batch(∂L/∂z2)

      Backprop into hidden layer:
        ∂L/∂a1 = ∂L/∂z2 @ W2.T
        ∂L/∂z1 = ∂L/∂a1 * relu'(z1)             <-- elementwise; the ReLU mask
                                                    blocks gradient where z1 <= 0
        ∂L/∂W1 = X.T @ ∂L/∂z1
        ∂L/∂b1 = sum_over_batch(∂L/∂z1)
    """
    X, z1, a1, probs = cache["X"], cache["z1"], cache["a1"], cache["probs"]
    N = X.shape[0]

    y_oh = one_hot(y_idx)                       # (N, 10)

    # Output layer
    dz2 = (probs - y_oh) / N                    # (N, 10)
    dW2 = a1.T @ dz2                            # (128, 10)
    db2 = dz2.sum(axis=0)                       # (10,)

    # Hidden layer
    da1 = dz2 @ W2.T                            # (N, 128)
    dz1 = da1 * relu_deriv(z1)                  # (N, 128) — gradient zeroed where ReLU was 0
    dW1 = X.T @ dz1                             # (784, 128)
    db1 = dz1.sum(axis=0)                       # (128,)

    return dW1, db1, dW2, db2


def predict(X):
    """Run forward and return the argmax class for each row. (No need for the
    backward cache here, but we still reuse forward() for code unity.)"""
    probs, _ = forward(X)
    return probs.argmax(axis=1)


# ===========================================================================
# 5. Training loop — mini-batch SGD
# ===========================================================================
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.1                # learning rate — how big a step to take in the
                        # gradient direction. Too big: loss explodes.
                        # Too small: training crawls.

n_train = X_train.shape[0]
n_batches = n_train // BATCH_SIZE

train_losses, test_accs = [], []

print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
print(f"  {n_batches} batches per epoch ({n_batches * BATCH_SIZE} examples covered)\n")

t_start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    # Shuffle the indices each epoch so batches differ between epochs.
    # Without this, the model sees the same 64-example chunks every time
    # and can find spurious patterns at chunk boundaries.
    perm = rng.permutation(n_train)

    epoch_loss = 0.0
    for b in range(n_batches):
        idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # --- Forward: compute predictions and the loss for this batch
        probs, cache = forward(X_batch)
        loss = cross_entropy(probs, y_batch)
        epoch_loss += loss

        # --- Backward: compute gradients
        dW1, db1, dW2, db2 = backward(cache, y_batch)

        # --- SGD update: nudge each parameter against its gradient.
        # This is one step of gradient descent. After ~9000 of these
        # (1 epoch), the loss has dropped substantially.
        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

    # End-of-epoch report. Evaluate on the FULL test set (not just a batch)
    # so the accuracy is honest.
    y_pred_test = predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_losses.append(epoch_loss / n_batches)
    test_accs.append(test_acc)
    print(f"epoch {epoch:2d}/{EPOCHS}   train_loss={train_losses[-1]:.4f}   test_acc={test_acc:.4f}")

print(f"\nTotal training time: {time.perf_counter() - t_start:.1f}s")


# ===========================================================================
# 6. Final evaluation
# ===========================================================================
y_pred = predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal test accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, digits=3))


# ===========================================================================
# 7. Plots
# ===========================================================================
# Loss + accuracy over epochs — the canonical training-curve plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs_axis = np.arange(1, EPOCHS + 1)
ax1.plot(epochs_axis, train_losses, marker="o")
ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss (cross-entropy)")
ax1.set_title("Training loss")
ax1.grid(True, alpha=0.3)
ax2.plot(epochs_axis, test_accs, marker="o", color="C2")
ax2.set_xlabel("epoch")
ax2.set_ylabel("test accuracy")
ax2.set_title("Test accuracy")
ax2.grid(True, alpha=0.3)
fig.suptitle(f"NumPy MLP — final test accuracy {acc:.3f}")
fig.tight_layout()
curves_path = PLOTS_DIR / "03_training_curves.png"
fig.savefig(curves_path, dpi=120, bbox_inches="tight")
print(f"\nSaved training curves    → {curves_path.relative_to(ROOT)}")

# Confusion matrix (compare with Phase 2's)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
    ax=ax, cmap="Blues", colorbar=True, values_format="d"
)
ax.set_title(f"NumPy MLP — confusion matrix (acc = {acc:.3f})")
fig.tight_layout()
cm_path = PLOTS_DIR / "03_confusion_matrix.png"
fig.savefig(cm_path, dpi=120, bbox_inches="tight")
print(f"Saved confusion matrix   → {cm_path.relative_to(ROOT)}")

# Misclassified samples
wrong_idx = np.where(y_pred != y_test)[0]
sample = rng.choice(wrong_idx, size=min(25, len(wrong_idx)), replace=False)
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, i in zip(axes.flat, sample):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"true={y_test[i]}  pred={y_pred[i]}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"25 misclassified test images (of {len(wrong_idx)} total errors)")
fig.tight_layout()
miss_path = PLOTS_DIR / "03_misclassified.png"
fig.savefig(miss_path, dpi=120, bbox_inches="tight")
print(f"Saved misclassified grid → {miss_path.relative_to(ROOT)}")
