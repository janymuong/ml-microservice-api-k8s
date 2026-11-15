"""Classification metrics for sleep staging, in plain numpy (no scikit-learn)."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    """Rows are the true stage, columns the predicted stage."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def summarize(y_true, y_pred, names):
    """Accuracy, macro-F1, Cohen's kappa, per-stage precision/recall/F1 and the confusion matrix.

    Cohen's kappa is the headline number in sleep-staging papers: it discounts the
    agreement you would get by chance, which matters because N2 is ~40% of all epochs.
    """
    n = len(names)
    conf = confusion_matrix(y_true, y_pred, n)
    total = conf.sum()
    per_class = {}
    for i, name in enumerate(names):
        tp = int(conf[i, i])
        predicted, actual = int(conf[:, i].sum()), int(conf[i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "support": actual}
    observed = np.trace(conf) / total if total else 0.0
    expected = (conf.sum(0) * conf.sum(1)).sum() / total ** 2 if total else 0.0
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return {
        "accuracy": round(float(observed), 4),
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / n, 4),
        "kappa": round(float(kappa), 4),
        "epochs": int(total),
        "per_stage": per_class,
        "confusion_matrix": conf.tolist(),
    }
