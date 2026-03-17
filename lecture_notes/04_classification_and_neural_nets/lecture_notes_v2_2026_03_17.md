# L04: Classification + Neural Networks (Top-Down)

In Lesson 3, we built linear regression from scratch - predicting house prices with gradient descent, MSE loss, and a training loop. You saw the full cycle: forward pass, compute loss, backward pass, update weights. That loop is the backbone of all ML training, and it doesn't go away. What changes in L4 is the question we're asking.

Instead of "how much?" we're asking "which one?" The output isn't a number like $347,000 - it's a category. Survived or died. Spam or not spam. Benign or malignant. The model's job is to output a probability: "I'm 87% confident this passenger survived."

That shift from numbers to categories is the difference between regression and classification. And it turns out we only need two new ingredients to make the jump: a function that squashes any number into a probability (sigmoid), and a loss function designed for yes/no answers (cross-entropy). Everything else - gradient descent, the training loop, weight updates - stays the same.

Once we have logistic regression working, we'll hit its limit: it can only draw straight decision boundaries. Some problems need curves. That's where neural networks enter. We'll build a Multi-Layer Perceptron (MLP) in PyTorch, train it on the same data, and watch it outperform logistic regression. The MLP learns its own features - curved, nonlinear boundaries that capture patterns logistic regression misses.

The goal isn't to fully understand what's happening inside the network yet. That's L5. The goal here is to see it work and build intuition for why adding layers and activation functions changes what a model can learn.

## Key Concepts

**Sigmoid** - takes any number and squashes it to a value between 0 and 1. Large negative numbers become close to 0, large positive numbers become close to 1, and zero maps to exactly 0.5. This is the bridge between regression (any output) and classification (probability output). In L3, our model output could be any real number. Wrapping it in sigmoid gives us something we can interpret as "chance of survival."

**Logistic regression** - same linear model from L3 (z = wx + b), but we pass the output through sigmoid before making a prediction. If sigmoid(z) > 0.5, predict "survived." Otherwise, predict "died." It's still using gradient descent to learn the weights - the only change is the activation at the end and the loss function.

**Cross-entropy loss** - replaces MSE from L3. The intuition: it heavily penalizes confident wrong predictions. If the model says "99% chance this person survived" and they actually died, the loss spikes. If it says "51% survived" and gets it wrong, the penalty is much smaller. This is exactly what we want - push the model to be honest about its uncertainty.

**Decision boundary** - the line (or curve) where the model switches from predicting one class to the other. For logistic regression, this is always a straight line. You can engineer polynomial features to get curves, but that requires manual feature design. Neural networks learn curves automatically.

**The MLP (Multi-Layer Perceptron)** - logistic regression with hidden layers in between. Instead of going straight from input to output, the data passes through intermediate layers that learn to create useful feature combinations. Each layer transforms the data, making the next layer's job easier. The key insight: the network builds its own features instead of you engineering them.

**ReLU (Rectified Linear Unit)** - the activation function between hidden layers. It keeps positive values and zeros out negatives. Simple, but critical: without it, stacking linear layers just gives you another linear layer (matrix math collapses). ReLU introduces the non-linearity that lets the network learn curves. Sigmoid squashes to 0-1 at the output, ReLU bends the space between layers.

**Hidden layers** - the intermediate layers between input and output. Each neuron in a hidden layer computes a weighted sum of its inputs, applies ReLU, and passes the result forward. Think of them as the network building its own intermediate representations - features you didn't design but that turn out to be useful for the task.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Classification** | Predicting a category, not a number | Survived/died, benign/malignant |
| **Sigmoid** | Squashes any number to 0-1 range | Output layer of logistic regression |
| **Logistic regression** | Linear model + sigmoid for classification | Our first classifier on Titanic |
| **Cross-entropy loss** | Loss function that penalizes confident wrong answers | Replaces MSE from L3 |
| **Decision boundary** | Line/curve where model switches predictions | Visualized in 2D feature space |
| **MLP (Multi-Layer Perceptron)** | Neural network with hidden layers | Our upgrade from logistic regression |
| **Hidden layer** | Intermediate layer that learns feature combinations | 16 neurons between input and output |
| **ReLU** | Activation: keep positives, zero out negatives | Between hidden layers |
| **Activation function** | Non-linear function applied after a layer | Sigmoid (output), ReLU (hidden) |
| **Neuron** | One unit in a layer, computes weighted sum + activation | 16 neurons in our hidden layer |
| **Architecture** | The structure of a network (layers, sizes, activations) | Linear-ReLU-Linear-Sigmoid |
| **Parameters/weights** | Numbers the network learns during training | 97 total in our small MLP |

## Connection to L3 and L5

**From L3:** Everything we built in the regression lesson carries over. The training loop (forward, loss, backward, update) is identical. Gradient descent still drives the learning. The difference is the output activation (sigmoid instead of raw output) and the loss function (cross-entropy instead of MSE). If L3 made sense, L4 is a small step - same engine, different task.

**To L5:** We've seen the MLP work and beat logistic regression, but we've treated it like a black box. L5 opens that box. We trace data through every layer - matrix multiplication, ReLU, another matrix multiplication, sigmoid. We see what hidden neurons actually compute, how loss.backward() calculates gradients, and how optimizers update weights. L4 gives you the "what" and the "why." L5 gives you the "how."

## Resources

### Course video

TBA

### Before the lesson

These videos build the intuition you'll need. Watch in this order if possible.
Gradient descent - but now a neural network: https://www.youtube.com/watch?v=IHZwWFHWa-w
- 3Blue1Brown - But What is a Neural Network? (19 min, the best visual explanation of neural networks available, walks through layers, neurons, and activations with animations): https://www.youtube.com/watch?v=aircAruvnKk
- StatQuest - Neural Networks Pt. 1: Inside the Black Box (20 min, builds from simple models to neural nets, clear and methodical, great complement to 3Blue1Brown): https://www.youtube.com/watch?v=CqOfi41LfDw
- StatQuest - Logistic Regression (15 min, clear walkthrough of sigmoid, log-odds, and maximum likelihood - the same model we build from scratch in the notebook): https://www.youtube.com/watch?v=yIYKR4sgzI8

### Quick videos

These are shorter and more focused. Good for reviewing specific concepts after the lesson.

- StatQuest - Neural Networks Pt. 3: ReLU in Action (10 min, shows exactly what ReLU does to a neural network's output, with step-by-step examples): https://www.youtube.com/watch?v=68BZ5f7P94E
- StatQuest - Neural Networks Pt. 6: Cross Entropy (11 min, builds the intuition for why cross-entropy works as a loss function for classification): https://www.youtube.com/watch?v=6ArSys5qHAU

### Documentation and tutorials

- Google ML Crash Course - Logistic Regression (interactive text, decent starting point for the math): https://developers.google.com/machine-learning/crash-course/logistic-regression
- Medium - Train a Neural Network in PyTorch (beginner walkthrough, covers the same ground as our notebook): https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
- PyTorch - Building Models tutorial (official docs, reference for nn.Module and nn.Sequential): https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

### Going deeper (optional)

- 3Blue1Brown - What is Backpropagation Really Doing? (13 min, preview of L5 content, watch if you're curious about what happens inside the network): https://www.youtube.com/watch?v=Ilg3gGewQ5U
