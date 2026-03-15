# ML Course — Utvecklarakademin

Machine learning course material. Notebooks, projects, and datasets for hands-on learning with Python, PyTorch, and scikit-learn.

## Repo structure

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

## Staying in sync with course updates

I will be pushing new lessons, fixes, and improvements to the course repo throughout the semester. You need to pull these updates into your fork regularly. Here's how:

### First time setup (do this once)

Your repo was created via GitHub Classroom from the template repo. Add the template as an "upstream" remote so you can pull updates from it:

```bash
cd ua_machine_learning_1
git remote add upstream https://github.com/UA-classroom/ua_ml_1.git
```

Verify it worked:

```bash
git remote -v
# origin    -> your GitHub Classroom repo (pia25-ml_1_course-ua_ml_1)
# upstream  -> the course template (ua_ml_1) where I push updates
```

### Pulling updates (do this before each lesson)

```bash
git fetch upstream
git merge upstream/main
```

Or just run the helper script:

```bash
./sync.sh
```

This will pull any new lessons, notebook updates, or bug fixes into your local copy.

**Important:** If you only edit files in `modified/` folders (which you should), this will merge cleanly every time. The `original/` folders are kept in sync with the course repo automatically. If you do get merge conflicts in notebooks you've changed, keep your version - your work matters more than the upstream update.

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
