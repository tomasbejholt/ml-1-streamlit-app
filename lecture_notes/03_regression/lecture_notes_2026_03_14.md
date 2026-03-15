# L03: Regression

Introduce gradient descent — the engine that powers all learning. Every model we train for the rest of the course uses this same loop.

## Topics

**The problem** — predict a continuous value. House prices, salary, temperature. The output is a number, not a category.

**Linear regression** — `y = wx + b`. Start with one feature, one weight, one bias. Draw the line through data points. This is the simplest possible model.

**Loss function: MSE** — "how wrong are we?" Mean Squared Error penalizes big mistakes more than small ones. Visualize the loss landscape as a bowl — we want to find the bottom.

**Gradient descent** — "follow the slope downhill":
- Compute the gradient (which direction is downhill?)
- Update the weight (take a step)
- Learning rate controls step size (too big = overshoot, too small = crawl)
- Repeat until the loss stops improving

Build the intuition visually first: loss landscape, the point moving toward the minimum. Then show the math: compute gradient, update weights.

**Multiple features** — same thing, just dot product. Each feature gets its own weight. The linear equation becomes `y = w₁x₁ + w₂x₂ + ... + b`.

**The training loop** — the pattern that never changes:
1. Forward pass: compute prediction
2. Loss: compare to target
3. Backward pass: compute gradients
4. Update: adjust weights
5. Repeat

Code from scratch first (pure Python/NumPy), then show the PyTorch version. PyTorch automates what was just built manually — the connection should be obvious.

**Bias and variance** — in L2 we saw overfitting and underfitting as patterns in loss curves. Now we can explain *why* they happen.

- **Bias** is how far off the model is *structurally*. A straight line fitting curved data will always be wrong in the same way no matter how much data you give it. That's high bias. The model's assumptions are too simple for the real pattern.
- **Variance** is how much the model changes when trained on different data. A model that fits every noise point perfectly will look completely different on a new sample. That's high variance. The model has too much flexibility and chases noise instead of signal.
- **The tradeoff**: simple models (high bias, low variance) are consistently wrong. Complex models (low bias, high variance) are unstable. Total error = bias + variance + irreducible noise. We want the sweet spot in the middle.

Practically: if both train and val loss are high, you have a bias problem (model too simple). If train loss is low but val loss is high, you have a variance problem (model too complex or too little data). This connects directly to what we saw in L2's overfit/underfit demo, but now we know the underlying reason.

## Terminology Introduced

parameters/weights, loss function, gradient, gradient descent, forward pass, learning rate, bias (model), variance

## Dataset

Simple regression — house prices, salary prediction, or similar. Something intuitive.

## Resources

Google's crash course on linear regression (good visual explanations):
https://developers.google.com/machine-learning/crash-course/linear-regression

StatQuest - Bias and Variance (clear explanation of the tradeoff, connects to overfitting/underfitting):
https://www.youtube.com/watch?v=EuBBz3bI-aA
