# L01: Lay of the Land

Show what ML can do, introduce core terminology. Goal: a mental map of the field and confidence that this is approachable.

## Notebook Structure

The notebook opens with "What is ML?" (traditional programming vs ML), then goes straight into three live demos before any taxonomy. The demos make the terminology concrete before defining it.

1. **What is ML?** — Traditional rules vs learned rules (spam filter example)
2. **Where ML Lives** — Real-world applications (recommendations, vision, language, fraud, science)
3. **Types of ML** — Supervised, unsupervised, self-supervised, reinforcement learning
4. **Classic vs Modern** — Feature engineering era vs deep learning, the practical split
5. **Demo 1: Titanic** — Tabular classification: explore → train → evaluate
6. **Demo 2: Images** — ResNet classifying pet photos, RGB channel visualization
7. **Demo 3: Text** — Sentiment analysis + text generation with HuggingFace
8. **The overwhelm talk** — Normalize the learning curve, "släpp sargen"
9. **Workflow** — The universal loop: Understand → Prepare → Train → Evaluate → Iterate → Deploy
10. **Vocabulary tables** — Reference definitions for features, labels, loss, etc.
11. **Course roadmap** — What's ahead

## Topics

**Tour of ML domains** — working demos across data types to show the breadth before zooming in:
- Tabular: Random Forest on Titanic (predict survival from passenger features)
- Images: torchvision ResNet classifying Oxford Pets photos
- Text: HuggingFace pipeline doing sentiment analysis + text generation

**Supervised learning** — the core pattern everything else builds on:
- Input → Model → Output
- The model learns a mapping from examples (features → labels)
- Training vs inference: learning vs using what you learned

**Core terminology** — introduce naturally through the demos:
- Features: the input data (columns, pixels, tokens)
- Labels: what we're predicting (survived/died, cat/dog, positive/negative)
- Model: the learned function
- Training: adjusting the model to fit the data
- Inference: using the trained model on new data

**Brief landscape** — one compact section after the demos, not three:
- Supervised vs unsupervised vs self-supervised
- Data types → best tools (tabular → trees, images → CNNs, text → transformers)
- Model families: linear, trees, neural networks

**The overwhelm talk** — normalize the learning curve early. ML has a brutal amount of moving parts. Repetition is how it clicks. "Släpp sargen."

## Terminology Introduced

features, labels, model, training, inference, supervised learning, classification, regression, overfitting, underfitting

## Lecture Notes

- The Titanic demo is the longest section and the best teaching moment — walk through it interactively, let students make predictions before the model does
- The image demo downloads Oxford Pets (~800MB) on first run — consider running this cell before the lecture starts
- The text demos download distilbert and distilgpt2 (~250MB) — same applies
- The "overwhelm talk" lands better mid-lesson (after the demos tire people out) than at the very start

## Getting Started

### Using Google Colab

1. Go to [colab.google.com](https://colab.google.com)
2. Click **File → Open notebook → GitHub**
3. Authorize GitHub if prompted
4. Paste the repo URL: `https://github.com/UA-classroom/pia25-ml_1_course-ua_ml_1`
5. Select the notebook to open
6. Work in Colab, then **File → Save a copy in GitHub** to save back to your fork

Colab gives you a free GPU and a pre-installed Python environment. Most packages (numpy, pandas, sklearn, torch, matplotlib) are already there. If something is missing:
```python
!pip install fastai transformers xgboost lightgbm
```

Colab sessions timeout after ~90 minutes of inactivity. Save your work frequently.

### Working Locally

For local setup, follow the instructions in the repo [README.md](../../README.md). The short version:

**With mamba (recommended):**
```bash
git clone <YOUR_FORK_URL>
cd ua_machine_learning_1
mamba create -n ml-venv python=3.12 -y
mamba activate ml-venv
mamba install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y
mamba install scikit-learn pandas numpy matplotlib seaborn jupyter -y
pip install fastai xgboost lightgbm transformers datasets kaggle
jupyter notebook
```

**With pip:**
```bash
git clone <YOUR_FORK_URL>
cd ua_machine_learning_1
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn jupyter
pip install fastai xgboost lightgbm transformers datasets kaggle
jupyter notebook
```

No GPU? Skip `pytorch-cuda=12.8` and the `-c nvidia` channel. macOS Apple Silicon? Replace the pytorch line with `mamba install pytorch torchvision -c pytorch -y`.

### Syncing Updates

When new lessons or fixes are pushed:
```bash
./sync.sh
```
Or manually:
```bash
git fetch upstream
git merge upstream/main
```

### Using Claude Code / Codex

Use AI coding tools to help you learn — ask them to explain code, create practice exercises, or debug your work. Some suggestions:
- Ask Claude Code to explain a cell you don't understand
- Have it create a variation of a demo with different data
- Use it to debug errors when running notebooks locally

## Resources

### Before the lecture (short, hook-style)
- Fireship — Machine Learning in 100 Seconds (~2 min, fast overview): https://www.youtube.com/watch?v=PeMlgBn-0cs
- Google — Introduction to ML (interactive text): https://developers.google.com/machine-learning/intro-to-ml

### After the lecture (deeper understanding)
- 3Blue1Brown — But what is a neural network? Ch. 1 (19 min, visual, intuitive): https://www.youtube.com/watch?v=aircAruvnKk
- 3Blue1Brown — Gradient descent, how neural networks learn Ch. 2 (previews L3): https://www.youtube.com/watch?v=IHZwWFHWa-w
- StatQuest — Machine Learning Fundamentals (short, clear): https://www.youtube.com/watch?v=Gv9_4yMHFhI
- StatQuest — ML playlist (can cherry-pick topics): https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
- Google ML Crash Course — Framing ML problems: https://developers.google.com/machine-learning/crash-course/framing/ml-terminology

### For the ambitious (optional, goes beyond L01)
- fast.ai Practical Deep Learning — Lesson 1 (top-down intro, similar philosophy): https://course.fast.ai/Lessons/lesson1.html
- Kaggle Learn — Intro to Machine Learning (hands-on, short exercises): https://www.kaggle.com/learn/intro-to-machine-learning
- Patrick Loeber — PyTorch Tutorial series (relevant from L3 onwards): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
