import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    recall_score,
    precision_score,
)


def tune_threshold(model, X_tr, y_tr, X_te, y_te, cv):
    """
    [tune_threshold] selects an F-beta-optimal classification threshold using out-of-fold
    predications on the training set and applies it once to the test set

    Returns [best_threashold] and [test_probabilites]
    """
    # trains on different folds
    train_probs = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    # finds best cutoff
    precision, recall, thresholds = precision_recall_curve(y_tr, train_probs)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f2)]

    test_probs = model.predict_proba(X_te)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(int)
    print(f"Threshold: {best_threshold:.3f}")
    print(classification_report(y_te, y_pred, target_names=["Low Risk","High Risk"]))
    return best_threshold, test_probs


def cv_recall_at_tuned_threshold(model, X, y, outer_cv, inner_cv):
    """
    [cv_recall_at_tuned_threshold] completes a nested cross-validation
    check then evaluates the recall and precision on the test outer fold

    prints thresholds and mean std recall/precision per fold 
    """
    recalls, precisions, thresholds_used = [], [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # INNER step: get honest out-of-fold probabilities on the training portion only
        inner_probs = cross_val_predict(model, X_tr, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
        prec, rec, thresh = precision_recall_curve(y_tr, inner_probs)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-10)
        best_thresh = thresh[np.argmax(f2)]
        thresholds_used.append(best_thresh)

        # fit on the FULL training fold, then evaluate on the held-out outer fold
        model.fit(X_tr, y_tr)
        probs_te = model.predict_proba(X_te)[:, 1]
        y_pred = (probs_te >= best_thresh).astype(int)

        recalls.append(recall_score(y_te, y_pred))
        precisions.append(precision_score(y_te, y_pred))

    print(f"Thresholds per fold: {[f'{t:.3f}' for t in thresholds_used]}")
    print(f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")

    return {
        "thresholds": thresholds_used,
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
    }


