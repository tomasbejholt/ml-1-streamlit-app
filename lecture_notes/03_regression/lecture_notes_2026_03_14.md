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

## Terminology Introduced

parameters/weights, loss function, gradient, gradient descent, forward pass, learning rate

## Dataset

Simple regression — house prices, salary prediction, or similar. Something intuitive.

## Resources

Google's crash course on linear regression (good visual explanations):
https://developers.google.com/machine-learning/crash-course/linear-regression

StatQuest - Bias and Variance (clear explanation of the tradeoff, connects to overfitting/underfitting):
https://www.youtube.com/watch?v=EuBBz3bI-aA
