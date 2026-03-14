# L11: CNNs — Concept & Architecture

Why CNNs work for images. Understand convolutions, build and train a CNN from scratch in pure PyTorch. No fastai — every piece is visible.

## Topics

**Why MLPs struggle** (recap from L9):
- Parameter explosion: flatten a 224×224×3 image = 150,528 inputs
- No spatial awareness: pixel shuffle experiment proved it
- CNNs solve both problems

**Convolution operation:**
- Kernel: a small grid of learnable weights (e.g. 3×3)
- Slides across the image, computing dot product at each position
- Output is a feature map: highlights where the kernel pattern was found
- Hand-craft edge detection kernels first, then let the network learn its own

**Feature maps** — what kernels detect:
- Early layers: edges, gradients, textures
- Later layers: parts, shapes, objects
- Each kernel produces one feature map

**Channels** — how depth grows:
- Input: RGB = 3 channels
- Each conv layer outputs multiple feature maps = multiple channels
- A 3×3 kernel on a 3-channel input is actually a 3×3×3 volume
- Output channels: one kernel per output channel

**Stride 2** — shrinking spatial dimensions:
- Skip every other position → output is half the size
- Modern replacement for max pooling
- The architecture pattern: spatial shrinks, channels grow, funnel to prediction

**Build CNN from scratch:**
- `nn.Conv2d` with a custom `conv` helper for consistent Conv→BN→ReLU blocks
- Track spatial dimensions through the network
- Flatten at the end → linear layer → 10 classes

**Training stability:**
- Diagnosing with activation statistics (track mean/std per layer)
- Activation collapse: when values drift to zero and the network stops learning
- Batch normalization: normalize activations between layers, fixes the root cause
- 1cycle LR scheduling: ramp up then decay the learning rate

**Compare CNN vs MLP:**
- Fewer parameters, higher accuracy
- The CNN exploits spatial structure the MLP ignores

**Famous architectures** (brief, conceptual only):
- ResNet: skip connections solve the vanishing gradient in deep networks
- EfficientNet: balance width, depth, and resolution

## Terminology Introduced

kernel, feature map, channels, stride, padding, batch normalization, 1cycle learning rate

## Dataset

MNIST for the teaching notebook (simple, fast iteration). Compare directly to MLP results.

## Notebook

Pure PyTorch, teaching notebook style. `L11_cnns.ipynb`

## Resources

CNN intro:
https://www.youtube.com/watch?v=pj9-rr1wDhM

Alternative intro:
https://www.youtube.com/watch?v=HGwBXDKFk9I

Practical convolutions in Excel (from ~44 min):
https://www.youtube.com/watch?v=htiNBPxcXgo
