# MNIST Digit Recognizer

A learning project: classify handwritten digits (0-9) from the MNIST dataset,
built three times with progressively more sophisticated tools.

> Second project in my [ML curriculum](../curriculum.txt).
> First was the [Maricopa Housing Predictor](../maricopa-housing-predictor) —
> tabular regression with sklearn. This one introduces neural networks,
> PyTorch, and image data.

## The pedagogical idea — build it three ways

The same network, three implementations, each one teaching a different layer
of the stack:

1. **Logistic regression (sklearn)** — establishes the baseline. ~92% accuracy.
   Same API patterns as the housing project.
2. **A 2-layer neural network from scratch in NumPy** — forward pass,
   backward pass, manual SGD, all written by hand. ~97% accuracy.
   *This is the phase where backpropagation stops being magic.*
3. **The same network in PyTorch** — re-implement using `nn.Linear`,
   `optim.SGD`, `DataLoader`. Same accuracy as the from-scratch version,
   but in 30 lines of code. Now `nn.Module` and the training-loop idiom are
   muscle memory for every subsequent project.
4. **A convolutional network in PyTorch** — replace dense layers with
   `Conv2d` + pooling. Watch accuracy jump from 97% to 99%+.
   See why convolutions are the right inductive bias for images.
5. **Deployment** — a Streamlit "draw a digit" app where users draw on a
   canvas and the model predicts in real time.

## Status

| Phase | Description                                  | Status      |
|-------|----------------------------------------------|-------------|
| 1     | Load + visualize MNIST                       | in progress |
| 2     | Logistic regression baseline                 | pending     |
| 3     | Neural net from scratch in NumPy             | pending     |
| 4     | Same network in PyTorch + upgrades           | pending     |
| 5     | Convolutional network                        | pending     |
| 6     | Streamlit "draw a digit" app                 | pending     |

## Folder layout

```
mnist-digit-recognizer/
├── notebooks/
│   ├── 01_get_data.py            # Pull MNIST, visualize, sanity-check
│   ├── 02_logistic_baseline.py   # sklearn LogisticRegression
│   ├── 03_numpy_neural_net.py    # 2-layer MLP, hand-coded backprop
│   ├── 04_pytorch_neural_net.py  # Same MLP, PyTorch flavor + upgrades
│   └── 05_pytorch_cnn.py         # CNN: Conv2d + pooling, push past 99%
├── app.py                        # Streamlit deployment (entry point)
├── data/
│   ├── raw/                      # torchvision downloads here (gitignored)
│   └── processed/                # Optional cached arrays (gitignored)
├── models/                       # Trained weights — final CNN committed for demo
├── src/                          # Reusable helpers
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

PyTorch installation note: the standard `pip install torch` pulls the CPU
build, which is fine for MNIST (you'll train each network in ~1-3 minutes on
a modern laptop). If you have an NVIDIA GPU and want CUDA acceleration, see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)
for the right install command.

## Run the pipeline

```bash
python notebooks/01_get_data.py        # Download MNIST + visualize samples
python notebooks/02_logistic_baseline.py
python notebooks/03_numpy_neural_net.py
python notebooks/04_pytorch_neural_net.py
python notebooks/05_pytorch_cnn.py
streamlit run app.py                   # Launch the draw-a-digit demo
```

## What I expect to learn

Concepts not covered in the housing project that this one introduces:

- Image data: pixels as features, normalization, batching
- Classification metrics: accuracy, per-class precision/recall, confusion matrix
- The **softmax + cross-entropy** combo for multi-class outputs
- **One-hot encoding the target** (not just features)
- Neural network primitives: layers, activations (ReLU, sigmoid), losses
- **Backpropagation written by hand** (the most important single exercise)
- **SGD and minibatches**: batch_size, epochs, learning rate
- PyTorch idioms: `nn.Module`, `optim`, `DataLoader`, `Dataset`, training loops
- Convolutional layers and pooling — *why* they work for images
- Regularization in deep learning: dropout, weight decay

## License

Code is MIT-licensed ([LICENSE](LICENSE)).
MNIST data is in the public domain (LeCun, Cortes, Burges 1998).
