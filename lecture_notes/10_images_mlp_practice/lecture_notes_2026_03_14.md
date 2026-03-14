# L10: Images — MLP Practice (CIFAR-10)

Practice image classification independently on CIFAR-10. Experiment with architectures, hyperparameters, and regularization. Push the MLP as far as it can go on images.

## Topics

**Apply pipeline from L9 to a new dataset** — CIFAR-10: 10 classes, 32×32 RGB, 60k images. Cleaner than the bird dataset, so the focus is on modeling, not data cleaning.

**Architecture experimentation:**
- Width: how many neurons per layer?
- Depth: how many layers?
- Activation functions: ReLU vs others
- "Try things, compare, build intuition"

**Hyperparameter tuning:**
- Learning rate: the single most important hyperparameter
- Batch size: tradeoff between noise and speed
- Epochs: when to stop training

**Regularization** — preventing overfitting:
- Dropout: randomly zero out neurons during training, forces redundancy
- Weight decay: penalize large weights, keep the model simple
- Batch normalization: normalize activations between layers, stabilizes training

**Reading loss curves** — the diagnostic skill:
- Training loss drops, validation loss drops: good, keep going
- Training loss drops, validation loss rises: overfitting
- Both losses plateau: model capacity reached or learning rate too low
- Both losses are high: underfitting, need more capacity or better features

**Compare multiple model variants** — build a table of experiments. Run experiments systematically, not just guess.

**Observe the MLP ceiling** — "We've pushed it as far as it goes on images. The architecture fundamentally can't capture spatial patterns. Time for something new."

## Terminology Introduced

learning rate scheduler, weight decay, regularization strategies

## Dataset

CIFAR-10 (10 classes, 32×32 RGB)

## Notebook

`08_mlp_image_classification_project.ipynb`
