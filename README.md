# Ranking Fertilizers

This project builds a ranking-based machine learning model to recommend the top 3 fertilizers for given soil and crop conditions. It was developed as part of a submission to the [Kaggle Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6) competition, which provides a synthetic dataset for predicting suitable fertilizers.

This project is a part of **Kaggle's Playground Series** - [Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6).

The model uses **CatBoost**, a gradient boosting library optimized for categorical features, and is fine-tuned using **Optuna**, a hyperparameter optimization framework. Evaluation is based on **Mean Average Precision at 3 (MAP@3)**, which rewards correct predictions appearing earlier in the ranked list.

---

## Data
This project uses the competition dataset from Kaggle, licensed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). All code is released under the MIT License (or your preferred license — feel free to update this).

---

## Notebooks & Scripts

| Notebook | Purpose |
|----------|---------|
| `FertilizerExploration.ipynb` | Preliminary data exploration and feature understanding using the original (non-synthetic) dataset. |
| `FertilizerTuning_Phase1.ipynb` | Broad hyperparameter tuning with Optuna to explore viable ranges. |
| `FertilizerTuning_Phase2.ipynb` | Focused tuning within narrowed ranges from Phase 1. |
| `FertilizerModelValidation.ipynb` | 4-fold cross-validation with early stopping to estimate final model performance and determine iteration count. |

- `FertilizerRankings.py`: Loads the full dataset, trains the final model with tuned parameters and calibrated iteration count, and generates ranked predictions for the test set.
- `utils.py`: Contains utility functions for MAP@3 scoring, ranking conversion, and formatted string output.

---

## Key Lessons
- Better hyperparameter tuning (Optuna)
- Difficuly in rankings
- Evaluation set
- 


## Requirements

The following packages need to be installed to run the project:
```bash
pip install numpy pandas matplotlib scikit-learn optuna catboost
```
These packages need to be installed in addition in order to run the Exploratory Data Analysis notebook:
```bash
pip install scipy seaborn xgboost
```
