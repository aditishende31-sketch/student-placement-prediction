"""
evaluate_model.py
------------------
Shared evaluation helpers used by both the training script and the GUI.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_predictions(y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall and F1-score for a set of
    predictions. zero_division=0 avoids crashes on edge cases where a
    model predicts only one class.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def get_confusion_matrix_values(y_true, y_pred):
    """
    Returns a dictionary of TP, TN, FP, FN for a binary classification
    problem where 1 = Placed, 0 = Not Placed.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
