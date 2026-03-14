# ML Course — Utvecklarakademin

Machine learning course material. Notebooks, projects, and datasets for hands-on learning with Python, PyTorch, and scikit-learn.

## Repo structure

```
lessons/              Lesson notebooks and projects
  intro/              L01–L02: Course intro, first model
  regression/         L03: Regression and gradient descent
  classification_and_mlps/  L04–L07: Classification, MLPs, trees
  neural_networks_images/   L09–L11: Image classification, CNNs
  recommendation_engines/   Collaborative filtering
  other_resources/    Reference notebooks (math intuition, etc.)

data/                 Shared datasets used across lessons
backups/              Snapshot copies of all lesson notebooks
documentation/        Course plans and notes
images/               Shared images
```

Each lesson folder follows this structure:

```
lesson_name/
  original/    ← Starter notebook (don't edit this one)
  modified/    ← Your working copy (edit, run, experiment here)
```

The `original/` folder is your reset point. If you break something in `modified/`, you can always copy the original back.

## Using Google Colab

1. Go to [colab.google.com](https://colab.google.com)
2. Click **File → Open notebook → GitHub**
3. Authorize GitHub if prompted
4. Search for this repo or paste the repo URL
5. Select the notebook you want to open
6. Work in Colab, then **File → Save a copy in GitHub** to save back to your fork

Always work from the `modified/` versions. Keep the `original/` copies clean.

## Getting updates

When new lessons or fixes are pushed, sync your fork with the upstream repo:

```bash
# One-time setup (only do this once)
git remote add upstream <UPSTREAM_REPO_URL>

# Pull latest changes
git fetch upstream
git merge upstream/main
```

If you haven't edited files in `original/` folders, this will merge cleanly. If you get conflicts in `modified/` notebooks you've changed, git will ask you to resolve them — keep your version or accept the update.

There's also a helper script you can run instead:

```bash
./sync.sh
```

## Working locally

If your machine can handle it, you can work locally instead of Colab.

### Setup with mamba (recommended)

[Miniforge](https://github.com/conda-forge/miniforge) gives you `mamba` — a fast drop-in replacement for `conda`. Install Miniforge first, then:

```bash
git clone <YOUR_FORK_URL>
cd ua_machine_learning_1

# Create environment
mamba create -n ml-venv python=3.12 -y
mamba activate ml-venv

# Core ML stack
mamba install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y
mamba install scikit-learn pandas numpy matplotlib seaborn jupyter -y

# Additional packages used in lessons
pip install fastai xgboost lightgbm transformers datasets kaggle

jupyter notebook
```

> **No GPU?** Skip `pytorch-cuda=12.8` and the `-c nvidia` channel — PyTorch will install CPU-only.
>
> **macOS Apple Silicon?** Replace the pytorch line with:
> `mamba install pytorch torchvision -c pytorch -y`

### Setup with pip

```bash
git clone <YOUR_FORK_URL>
cd ua_machine_learning_1
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn jupyter
pip install fastai xgboost lightgbm transformers datasets kaggle
jupyter notebook
```

### Datasets

Small datasets are included in `data/`. Larger ones need to be downloaded — see [`data/DATASETS.md`](data/DATASETS.md) for instructions.

Use tools like Claude Code or Codex to help you learn — create your own notebooks, experiment, break things.

## License

© 2025 Utvecklarakademin UA Aktiebolag. All rights reserved. See [LICENSE](LICENSE) for details.

This material may not be reproduced, distributed, or shared without written permission.
