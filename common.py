"""Shared helpers: data loading, label encoding, threshold tuning, metrics."""
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support

DATA_DIR = "data/go_emotions"


def load_data(data_dir=DATA_DIR):
    """Returns (DatasetDict with train/validation/test, list of label names)."""
    ds = load_from_disk(data_dir)
    label_names = ds["train"].features["labels"].feature.names
    return ds, label_names


def multi_hot(label_lists, n_classes):
    """[[0, 3], [27]] -> binary matrix (n_examples, n_classes)."""
    y = np.zeros((len(label_lists), n_classes), dtype=np.int64)
    for i, labs in enumerate(label_lists):
        labs = np.asarray(labs, dtype=np.int64)
        if labs.size:
            y[i, labs] = 1
    return y


def _avg(y_true, y_pred, average):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def tune_threshold(y_true, probs, grid=None):
    """Pick the global decision threshold that maximises macro-F1 (use VALIDATION data)."""
    if grid is None:
        grid = np.arange(0.05, 0.80, 0.025)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = _avg(y_true, (probs >= thr).astype(int), "macro")["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr, best_f1


def compute_metrics(y_true, probs, threshold, label_names):
    """Full metric dict: micro/macro/weighted/samples averages + per-emotion scores."""
    y_pred = (probs >= threshold).astype(int)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, name in enumerate(label_names)
    }
    return {
        "threshold": float(threshold),
        "micro": _avg(y_true, y_pred, "micro"),
        "macro": _avg(y_true, y_pred, "macro"),
        "weighted": _avg(y_true, y_pred, "weighted"),
        "samples": _avg(y_true, y_pred, "samples"),
        "per_class": per_class,
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
