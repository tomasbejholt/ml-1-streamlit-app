# L09: Images — The Complete Pipeline with MLP

Teach the full image classification pipeline end-to-end. Same MLP architecture from tabular, applied to images. Demonstrate everything that plug-and-play datasets like CIFAR-10 hide from you.

## Topics

**Images as data:**
- Pixels: a grid of numbers
- RGB channels: three grids stacked (red, green, blue)
- Flattening: reshape the 2D grid into a 1D vector to feed into an MLP

**Data cleaning** — verify images load, handle corrupted files. Real datasets are messy. Check before training, not during.

**Resizing strategies:**
- Squash: distort aspect ratio (fast, sometimes fine)
- Crop: lose information at edges
- Resize + crop: resize to slightly larger, then center crop
- Show the visual difference — seeing it matters more than reading about it

**transforms.Compose()** — the image preprocessing pipeline. Compare to pandas preprocessing chains: same idea, different domain.

**Per-channel normalization** — compute mean/std per RGB channel, like StandardScaler but for each color channel independently. Why? Pretrained models expect it, and it helps training converge.

**Data augmentation** — create training variety without collecting more data:
- Horizontal flip, rotation, color jitter, random crop
- Only apply to training data, never validation
- "Free data, but realistic — don't distort beyond what makes sense"

**Visual verification** — always check your pipeline before training. Display sample batches. Catch mistakes before wasting GPU hours.

**Multi-class classification:**
- Softmax: output probabilities across all classes (they sum to 1)
- CrossEntropyLoss: combines log-softmax and NLL in one step
- Trace with real numbers to make the math concrete

**MLP on images:**
- Flatten + linear layers
- Parameter explosion: 32×32×3 = 3,072 inputs → 16 hidden → already thousands of weights
- Bigger images = exponentially more parameters

**Training:**
- LR finder: sweep learning rates, plot loss, pick the steepest descent
- Training loop with `model.train()` / `model.eval()`

**Evaluation:**
- Confusion matrix: which classes get confused with each other?
- Per-class accuracy: some classes are harder
- Top losses: look at the images the model got most wrong — often reveals data issues

**MLP limitations** — the pixel shuffle experiment:
- Randomly permute pixel positions
- Retrain: accuracy barely changes
- "The MLP has zero spatial awareness — it treats every pixel independently"
- This is the motivation for CNNs

## Terminology Introduced

softmax, multi-class classification, data augmentation, per-channel normalization, transforms pipeline, top losses

## Dataset

Bird photos (5 classes: eagle, flamingo, owl, parrot, penguin) — web-scraped and messy, forces every pipeline step.

## Notebook

`08_v2_mlp_image_classification.ipynb`
