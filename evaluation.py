"""Metric calculations shared by every method: overall, per class and confusion matrix."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(gold, preds, labels: list) -> dict:
    """Returns overall metrics, per class metrics and a confusion matrix for one method.

    The labels list fixes the class order so that every method produces a
    matrix with the same layout. Rows of the matrix are gold labels and
    columns are predicted labels. Values are rounded to four decimals.
    """
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, labels=labels, average="macro", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        gold, preds, labels=labels, average=None, zero_division=0)
    per_class = {
        label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                "f1": round(float(f1[i]), 4), "support": int(support[i])}
        for i, label in enumerate(labels)
    }
    matrix = confusion_matrix(gold, preds, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(gold, preds)), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "rows_are": "gold", "columns_are": "predicted",
                             "matrix": matrix.tolist()},
    }


def confusion_frame(metrics: dict) -> pd.DataFrame:
    """Converts the stored confusion matrix into a labelled DataFrame for CSV export."""
    cm = metrics["confusion_matrix"]
    names = cm["labels"]
    return pd.DataFrame(cm["matrix"], index=[f"gold_{n}" for n in names], columns=[f"pred_{n}" for n in names])
