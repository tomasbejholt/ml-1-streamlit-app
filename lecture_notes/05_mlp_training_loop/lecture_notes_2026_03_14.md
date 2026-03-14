# L05: First MLP — The Training Loop

Understand what happens inside a neural network. The forward pass is the main focus — trace your data through every layer and see what happens to the numbers.

## Topics

**Revisit the MLP from L4** — "Let's understand what each line does."

**Forward pass** (main focus, trace with real numbers):
1. Input data as tensor
2. First linear layer: matrix multiplication + bias — what the numbers actually look like
3. ReLU activation: negative values become zero, positive values pass through
4. Second linear layer: another matmul + bias
5. Sigmoid: final probability output
6. "Trace your data through the network" — follow actual numbers through each operation

**Loss calculation** — comparing predictions to targets. Cross-entropy: the model gets penalized more for confident wrong answers than uncertain ones.

**Backpropagation** (simplified, not the full calculus):
- "We calculate partial derivatives"
- "This tells us how much each weight affected the loss"
- "PyTorch does this with `loss.backward()`"
- Gradients are stored on each parameter, ready to use
- Don't need to understand the chain rule to use it — but understanding that it exists matters

**Weight update** — `weight -= lr * gradient`. This is what learning is. Each weight nudges in the direction that reduces the loss.

**The full loop** — forward → loss → backward → update → repeat. This pattern never changes, whether you're training on 10 rows or 10 million images.

**Epochs and batches** — process data in chunks (batches), repeat the full dataset multiple times (epochs). Why batches? Memory. Why epochs? One pass isn't enough.

**PyTorch mechanics** woven in naturally:
- Tensors: multi-dimensional arrays on GPU
- `requires_grad`: tells PyTorch to track operations for backprop
- `nn.Module`: the base class for all models

**Optimizers** — "SGD is what we've been doing manually. Adam is the common default — it adapts the learning rate per parameter."

## Terminology Introduced

tensor, batch, epoch, backpropagation (simplified), optimizer, SGD, Adam

## Dataset

Titanic — continuing from L4 keeps the focus on the internals, not the data.

## Resources

Pytorch neural network tutorial:
https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
