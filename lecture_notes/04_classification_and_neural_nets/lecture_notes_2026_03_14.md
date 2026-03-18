# L04: Classification + Neural Networks (Top-Down)

Build the argument for neural networks. Show they work before explaining how. By the end: a trained MLP that beats logistic regression, and curiosity about what's inside.

## Topics

**Problem shift** — predicting categories, not numbers. Survived/died, cat/dog, spam/not-spam. The output is a probability.

**Sigmoid** — squash any number to a probability between 0 and 1. This is the bridge from regression to classification.

**Logistic regression** — linear model + sigmoid:
- Same gradient descent loop from L3
- Cross-entropy loss: "penalizes confident wrong answers" (brief, intuitive)
- Train on Titanic, see accuracy

**The limitation** — "only linear decision boundaries." Visualize problems logistic regression can't solve (XOR, concentric circles). Some patterns need curves, not lines.

**Enter the MLP** — stack layers with non-linearities:
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
```
- Train on same Titanic data
- Compare: MLP beats logistic regression
- "Adding a hidden layer with ReLU lets it learn curved boundaries"

**Conceptual explanation** (keep it intuitive, details come in L5):
- Hidden layers: intermediate representations — the network builds its own features
- ReLU: non-linearity between layers — without it, stacking layers does nothing
- Why "deep"? More layers = more abstract features

**Preview:** "It works better. But what's actually happening inside? Next lesson we trace the data through every layer."

## Terminology Introduced

sigmoid, cross-entropy, logistic regression, activation function, hidden layer, ReLU, neurons, layers, architecture

## Dataset

Titanic — familiar data, so the focus stays on the model.

## Resources

Google's tutorial on logistic regression:
https://developers.google.com/machine-learning/crash-course/logistic-regression

Resources:

https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078