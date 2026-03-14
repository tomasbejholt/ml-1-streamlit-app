# L06: Full Tabular Pipeline

Complete end-to-end workflow for training on tabular data. Every step from raw data to evaluated model. This pipeline gets reused throughout the course.

## Topics

**Data exploration** — understand what you're working with before touching the model. Shape, distributions, correlations, missing values. Never skip this.

**Data preparation:**
- Missing values: checking strategies, when to drop vs impute
- Normalization: why neural nets need it (gradient scale), standard scaling
- Categorical encoding: one-hot vs label encoding, when to use which
- Log transform for skewed distributions (e.g. prices, incomes)
- Train/validation split: why you need held-out data

**The full pipeline** — the end-to-end pattern:
Preprocessing → DataLoader → Model → Training loop → Evaluation

Each step is a building block. Compare to web dev patterns: data prep is like input validation, DataLoader is like pagination, the training loop is like a processing pipeline.

**Evaluation:**
- Accuracy: simple but misleading with imbalanced data
- Precision: "of everything you predicted positive, how many were correct?"
- Recall: "of everything that was actually positive, how many did you catch?"
- Confusion matrix: the full picture at a glance
- Cross-validation: "more robust than a single split" (brief intro)

**Class imbalance** — "what if 95% of data is one class?" A model that always predicts the majority class gets 95% accuracy but is useless. Brief intro to the problem and common approaches.

**Transition:** "We've trained an MLP on tabular data. But is this the best tool for tabular?"

## Terminology Introduced

train/test split, confusion matrix, precision, recall, cross-validation, class imbalance

## Dataset

New tabular dataset — something different from Titanic to practice the full pipeline on fresh data.

## Resources

PyTorch Dataset & DataLoaders tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

DataLoaders video:
https://www.youtube.com/watch?v=PXOzkkB5eH0

PyTorch classification from scratch:
https://www.slingacademy.com/article/pytorch-classification-from-scratch-building-a-dense-neural-network/

PyTorch for tabular data:
https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

NYC taxi fares example:
https://francescopochetti.com/pytorch-for-tabular-data-predicting-nyc-taxi-fares/
