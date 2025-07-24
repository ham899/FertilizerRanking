# File: utils.py
# Author: Hunter Moricz
# Last Modified: June 22, 2025
# Description: Contains utility functions for evaluating ranking-based classification models.
# Includes functions to compute MAP@3, convert predicted rankings to string format,
# and generate prediction rankings from model probabilities.
# ---------------------------------------------------------------------------------------

from typing import List
import numpy as np
import pandas as pd


def list_to_string(series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series of list elements into a Series of string elements with the list entries separated by spaces.

    Parameters
    ----------
    series : pd.Series
        A Series where each entry is a list of strings.

    Returns
    -------
    pd.Series
        A Series with each list converted to a single space-separated string.
    """
    return series.apply(lambda x: ' '.join(x))


def AP3(label: str, ranking: List[str]) -> float:
    """
    Computes the Average Precision at 3 (AP@3) for a single predicted ranking.

    Parameters
    ----------
    label : str
        The true class label.
    ranking : List[str]
        A list of the rankings for predicted class labels ordered in descending order by probability.

    Returns
    -------
    float
        Returns the AP@3 score for the input list ranking.
    """
    # Go through the ranking to find the position of the true label
    for i in range(min(3, len(ranking))):
        if ranking[i] == label:
            return 1 / (i+1)
    
    # Since the true label was not found, return 0
    return 0.0


def MAP3(labels: List[str], rankings: List[List[str]]) -> float:
    """
    Computes the Mean Average Precision at 3 for a set of predictions.

    Parameters
    ----------
    labels : List[str]
        A list of true class labels.

    rankings : List[List[str]]
        A list of predicted label rankings for each observation. Each sublist must contain the top-3 predicted class labels.
    
    Returns
    -------
    float
        The MAP@3 score of the input rankings.
    """
    n = len(labels)

    if n != len(rankings):
        raise ValueError('The number of rankings needs to be equal to the number of true labels.')

    return np.mean([AP3(label, ranking) for label, ranking in zip(labels, rankings)])


def get_ranking(probs: np.ndarray, mapping: List[str], n: int = 3) -> List[List[str]]:
    """
    Generates class label rankings for the top `n` predictions based on their predicted probabilities.

    Parameters
    ----------
        probs : np.ndarray
            A 2D numpy array of class prediction probabilities
        mapping : List[str]
            A list to transform the numeric labels back to string class label, mapping class indices to class labels.
        n : int, optional
            The number of top class label to select for the ranking (default is 3).

    Returns
    -------
        List[List[str]]
            A list of the label predictions with the highest probability of occurrence.
    """
    rankings = np.argsort(probs, axis=1)[:, ::-1][:, :n]

    return [[mapping[i] for i in row] for row in rankings]


def generate_model_rankings(model, X_test: pd.DataFrame, mapping: List[str], n: int = 3) -> List[List[str]]:
    """
    Generates prediction rankings using a trained classification model.

    Parameters
    ----------
        model: object
            A trained classification model with a `predict_proba()` method.
        X_test: pd.DataFrame
            The feature set for which to generate predictions.
        mapping: List[str]
            A list to transform the numeric labels back to string class label, mapping class indices to class labels.
        n : int, optional
            The number of top class label to select for the ranking (default is 3).

    Returns
    -------
        List[List[str]]
            A list of predicted class label rankings for each instance in `X_test`.
    """

    probs = model.predict_proba(X_test)

    return get_ranking(probs, mapping, n)