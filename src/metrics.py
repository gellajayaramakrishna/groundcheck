"""Metrics for a rare positive class (hallucinated = 1). Plain accuracy is misleading here,
so we report precision/recall/F1 on the hallucinated class, balanced accuracy, ROC-AUC and PR-AUC."""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)


def best_threshold(y_true, scores):
    """Pick the threshold that maximizes F1 on the hallucinated class (tuned on validation only)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f1 = f1_score(y_true, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def compute_metrics(y_true, scores, threshold):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if len(set(y_true)) > 1 else None,
    }
