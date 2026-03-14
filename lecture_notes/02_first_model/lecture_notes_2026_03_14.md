# L02: Your First Model (Quick Win)

Train and fine-tune a vision model using fastai. See it work before understanding why. This is the only lesson that uses fastai — from L3 onwards everything is PyTorch to expose what's happening under the hood.

## Topics

**fastai DataBlock API** — loading image data with minimal code. No need to understand every argument yet, just see the pattern: tell the library what data you have and how it's organized.

**Transfer learning** — use a pretrained ResNet. Someone already trained this model on millions of images. We're borrowing that knowledge.

**Fine-tuning** — `learn.fine_tune(3)`. Three lines of code, a working image classifier. Watch the loss go down and accuracy go up in real time.

**Interpreting results** — what did the model learn? Look at:
- Confusion matrix: where does it get confused?
- Top losses: which images were hardest?

**Terminology as it appears in output** — don't front-load definitions, explain them when they show up:
- Epoch: "we're on epoch 2 of 3" — one pass through all training data
- Loss: "see this number going down?" — how wrong the model is
- Learning rate: "this controls step size" — how aggressively it learns
- Train/valid: "why two losses?" — checking for overfitting
- Accuracy: "how often it's right" — the metric we care about

**The hook:** "It works! But how? That's what we'll learn next."

## Terminology Introduced

epoch, loss, learning rate, train/valid split, accuracy

## Dataset

Pet breeds, food images, or similar — visual and fun. The dataset should spark curiosity.
