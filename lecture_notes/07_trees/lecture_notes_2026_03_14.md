# L07: Trees — Random Forests & Gradient Boosting

Alternative to neural nets for tabular data. Honesty: "trees often win here." Compare approaches and learn to pick the right tool.

## Topics

**Decision trees:**
- How they work: binary splits at each node, splitting on the feature that best separates the data
- Impurity (Gini): "how mixed is each group after splitting?"
- Visual: tree structure, decision boundaries — trees carve the feature space into rectangles
- Pros: interpretable (you can read the rules), no preprocessing needed
- Cons: overfits easily — a deep tree memorizes the training data

**Random Forests:**
- Ensemble of trees: train many trees on random subsets of data and features
- Bagging: each tree sees a different slice, so they make different mistakes
- Why averaging works: uncorrelated errors cancel out
- Feature importance: which features mattered most across all trees?

**Gradient Boosting** (conceptual, don't go deep into the sequential math):
- XGBoost, LightGBM, CatBoost — the tools you'll see in production
- "Trees that learn from mistakes" — each tree corrects the errors of the previous ones
- Often state-of-the-art on tabular data
- More prone to overfitting than random forests, needs more tuning

**Head-to-head comparison** on the same dataset:
- MLP vs Random Forest vs Gradient Boosting
- Compare accuracy, training time, interpretability
- Discuss results honestly — trees usually win on tabular

**When to use what:**
- Tabular data: try trees first, they're fast and often best
- Images/text: neural nets (trees can't handle spatial/sequential structure)
- Small data: trees, they need less data to learn
- Both: always worth trying both and comparing

**MLP repetition** — rebuild the MLP and compare directly. Reinforces neural net understanding while being honest about limitations.

## Terminology Introduced

decision tree, random forest, bagging, ensemble, gradient boosting, feature importance

## Dataset

Titanic (compare to L6 MLP results) + possibly a second dataset for regression (bulldozer prices).

## Notebooks

- modern_tabular_ml.ipynb
- how random forests really work

## Resources

Random forests:
https://www.youtube.com/watch?v=AdhG64NF76E&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7

Gradient boosted trees (work through decision trees and random forests first):
https://www.youtube.com/watch?v=PxgVFp5a0E4

Model interpretation using SHAP:
https://www.youtube.com/watch?v=MQ6fFDwjuco

Model interpretable vs explainable:
https://www.youtube.com/watch?v=VY7SCl_DFho
