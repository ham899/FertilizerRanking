# Ranking Fertilizers

This project was developed for [Kaggle's Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6), which closed on June 30, 2025.

The objective is to produce ranked fertilizer recommendations based on environmental features such as weather conditions, soil type, and crop type. The rankings aim to propose the most suitable fertilizers under a given set of growing conditions.

The evaluation metric used is **Mean Average Precision at 3 (MAP@3)**, which assigns higher scores to models that rank the correct fertilizer label higher in the top-3 predictions.

The model used is a **CatBoostClassifier** model, with its hyperparameters tuned using the **Optuna** package.

---

## Data

The competition data is synthetically generated from an original dataset that I have included in this repository. The competition dataset can be found on the [Kaggle Playground Series S5E6 Competition Page](https://www.kaggle.com/competitions/playground-series-s5e6/data). Both datasets are licensed under **CC0: Public Domain**.

To run the files without modification, ensure a folder named `/Data` and a folder named `/Output` exist in your working directory.

---

## Project Summary

I wanted to explore the `catboost` package and try out the **CatBoostClassifier**, especially since the dataset included two important categorical features. This project gave me a chance to aim for a strong MAP@3 score while also learning a new modelling library.

For hyperparameter tuning, I used the **Optuna** package. In my last project, I used `RandomSearchCV` and `GridSearchCV`; however, this time I wanted to experiment with `optuna`, as it employs a more efficient, optimization-based approach to finding the best parameters.

The final CatBoostClassifier model achieved an **MAP@3 score of 0.33073** in the competition, placing me 1,399th out of 2,648 participants.


### Notebooks & Scripts

The preliminary data exploration can be found in the notebook `FertilizerExploration.ipynb`, which works with the original dataset.

The rest of the pipeline is based on the competition dataset and is organized as follows:

| File | Purpose |
|----------|---------|
| `utils.py` | Contains utility functions for computing MAP@3 scores, generating prediction rankings from probabilities, and formatting final outputs.|
| `FertilizerTuning_Phase1.ipynb` | Broad hyperparameter tuning with Optuna to explore viable ranges. |
| `FertilizerTuning_Phase2.ipynb` | Focused tuning within narrowed ranges from Phase 1 results. |
| `FertilizerModelValidation.ipynb` | 4-fold cross-validation with early stopping to estimate final model performance and determine best iteration count. |
| `FertilizerRankings.py` | Trains the final model on the full dataset using the tuned parameters and best iteration count, and generates ranked predictions for the test set. |

---

## Key Lessons

- **Handling Challenging Classification Tasks**: Compared to academic datasets, this problem offered limited feature separability. The experience emphasized the importance of fine-tuning, training time, and data volume. In hindsight, I would have invested more time in feature engineering.
- **Learning Optuna**: This was my first project using `optuna`. Its optimization-based search is both efficient and customizable. I look forward to using it on more complex pipelines.
- **Evaluation Set Usage**: Including an evaluation set helped control overfitting and allowed the model to dynamically determine the best number of boosting iterations using early stopping.

---

## Requirements

The following packages need to be installed to run the project:
```bash
pip install numpy pandas matplotlib scikit-learn optuna catboost
```
These packages need to be installed in addition in order to run the Exploratory Data Analysis notebook:
```bash
pip install seaborn scipy xgboost
```
