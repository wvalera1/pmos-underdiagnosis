# tests/test_evaluation.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.evaluation import tune_threshold, cv_recall_at_tuned_threshold


"""
generates fake data to test evaluation 
"""
def make_fake_classification_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    # make y somewhat predictable from f1 and imbalanced 
    y = (X["f1"] + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    y = pd.Series(y)
    return X, y


"""
tests if the threshold returned from [tune_threshold] outputs
a valid output (in the interval [0,1] and having one probability 
for each row in X_test)
"""
def test_tune_threshold_returns_valid_threshold():
    X, y = make_fake_classification_data()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=3)

    threshold, probs = tune_threshold(model, X_train, y_train, X_test, y_test, cv)

    assert 0 <= threshold <= 1
    assert len(probs) == len(X_test)


"""
ensures the recall_mean returned by [cv_recall_at_tuned_threshold] 
and that it returns valid recall and precision
numbers (in the interval [0, 1])
"""
def test_cv_recall_at_tuned_threshold_runs():
    X, y = make_fake_classification_data()
    model = LogisticRegression()
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    result = cv_recall_at_tuned_threshold(model, X, y, outer_cv, inner_cv)

    assert "recall_mean" in result
    assert 0 <= result["recall_mean"] <= 1
    assert 0 <= result["precision_mean"] <= 1