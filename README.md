# ML Course — Utvecklarakademin

Machine learning course material. Notebooks, projects, and datasets for hands-on learning with Python, PyTorch, and scikit-learn.

## Repo structure

Each lesson folder follows this structure:

```
lesson_name/
  original/    ← Starter notebook (don't edit this one)
  modified/    ← Your working copy (edit, run, experiment here)
  example/     ← Sometimes exists: a completed example for reference
```

The `original/` folder is your reset point. If you break something in `modified/`, you can always copy the original back. Some homework folders include an `example/` folder with a completed version you can reference if you get stuck - try to solve it yourself first.

## Using Google Colab

[Google Colab](https://colab.google.com) is a free online notebook environment that runs in your browser. Think of it as a cloud computer with Python and most ML libraries pre-installed - you don't need to install anything on your own machine. It also gives you free access to GPUs, which makes training models much faster. The notebooks look and work just like Jupyter notebooks, but everything runs on Google's servers.

Here's how to connect it to your course repo.

### First time setup

1. Go to [colab.google.com](https://colab.google.com) and sign in with your Google account
2. Connect GitHub to Colab: go to [colab.google.com/github](https://colab.google.com/github) - this will prompt you to authorize Colab to access your GitHub account
3. Check the **"Include private repositories"** checkbox (important - your course repo is private)
4. If prompted, grant access to the **UA-classroom** organization so Colab can see your course repo

### Opening a notebook

1. In the GitHub tab, paste your repo URL (the one from your GitHub Classroom assignment, something like `https://github.com/UA-classroom/pia25-ml_1_course-ua_ml_1-YOURUSERNAME`)
2. **Be patient** - the UA-classroom organization has a lot of repos, so searching can take 30-60 seconds. Just wait for the loading to finish.
3. Colab will list all notebooks in the repo. You can browse the folder structure to find what you need.
4. Pick the one you want to work on - always open from the `modified/` folder

### How Colab works with GitHub

When you open a notebook from GitHub, Colab creates a **copy** in your Colab session. It does not modify the repo. Your changes only exist in that session and will disappear if it times out (~90 minutes of inactivity).

To save your work back to the repo:
- **File -> Save a copy in GitHub** - this commits the notebook back to your repo on GitHub

To save to Google Drive instead (as a backup):
- **File -> Save a copy in Drive**

### Important notes

- Always work from the `modified/` versions. Keep the `original/` copies clean.
- Colab sessions timeout after ~90 minutes of inactivity - save frequently.
- Most packages are pre-installed. If something is missing: `!pip install fastai transformers xgboost lightgbm`
- If your repo doesn't show up in Colab, go to [github.com/settings/applications](https://github.com/settings/applications), find Google Colaboratory, and grant it access to the UA-classroom organization.

## Staying in sync with course updates

I will be pushing new lessons, fixes, and improvements to the course repo throughout the semester. You need to pull these updates into your fork regularly. Here's how:

### First time setup (do this once)

Your repo was created via GitHub Classroom from the template repo. Add the template as an "upstream" remote so you can pull updates from it:

```bash
cd ua_machine_learning_1
git remote add upstream https://github.com/UA-classroom/pia25-ml_1_course-ua_ml_1.git
```

Verify it worked:

```bash
git remote -v
# origin    -> your own repo (pia25-ml_1_course-ua_ml_1-YOURUSERNAME)
# upstream  -> the course assignment repo (pia25-ml_1_course-ua_ml_1) where updates are pushed
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
