"""Per-emotion decision thresholds, tuned on VALIDATION, evaluated once on TEST.
Uses the probabilities already saved by baseline_lr.py / train_bert.py (no GPU needed).

    python per_class_thresholds.py
Writes results/<model>/metrics_perclass.json and report_assets/comparison_perclass.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from common import load_data, multi_hot, save_json

GRID = np.arange(0.02, 0.90, 0.02)


def tune_per_class(y_val, p_val, fallback):
    """Best threshold per class on val. Classes with no val positives keep `fallback`."""
    thr = np.full(y_val.shape[1], fallback, dtype=float)
    for c in range(y_val.shape[1]):
        if y_val[:, c].sum() == 0:
            continue
        scores = [f1_score(y_val[:, c], (p_val[:, c] >= t).astype(int), zero_division=0) for t in GRID]
        thr[c] = GRID[int(np.argmax(scores))]
    return thr


def evaluate(y_true, probs, thr, names):
    y_pred = (probs >= thr[None, :]).astype(int)  # vector threshold broadcasts over classes
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    out = {"thresholds": {n: float(t) for n, t in zip(names, thr)}}
    for avg in ("micro", "macro", "weighted", "samples"):
        ap, ar, af, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[avg] = {"precision": float(ap), "recall": float(ar), "f1": float(af)}
    out["per_class"] = {
        n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, n in enumerate(names)
    }
    out["zero_f1_classes"] = [n for i, n in enumerate(names) if f[i] == 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/go_emotions")
    ap.add_argument("--lr_dir", default="results/lr")
    ap.add_argument("--bert_dir", default="results/bert")
    ap.add_argument("--out", default="report_assets")
    args = ap.parse_args()

    ds, names = load_data(args.data_dir)
    n = len(names)
    y_va = multi_hot(ds["validation"]["labels"], n)
    y_te = multi_hot(ds["test"]["labels"], n)

    results = {}
    for tag, d in (("LR", args.lr_dir), ("BERT", args.bert_dir)):
        d = Path(d)
        old = json.load(open(d / "metrics.json"))
        p_va, p_te = np.load(d / "val_probs.npy"), np.load(d / "test_probs.npy")
        thr = tune_per_class(y_va, p_va, fallback=old["threshold"])
        res = evaluate(y_te, p_te, thr, names)
        res["global_threshold_macro_f1"] = old["macro"]["f1"]
        save_json(res, d / "metrics_perclass.json")
        results[tag] = res
        print(
            f"{tag:5s} macro-F1: global thr {old['macro']['f1']:.3f} -> per-class thr "
            f"{res['macro']['f1']:.3f} | micro {res['micro']['f1']:.3f} | zero-F1 classes: "
            f"{res['zero_f1_classes'] or 'none'}"
        )

    header = ["Average", "LR P", "LR R", "LR F1", "BERT P", "BERT R", "BERT F1"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for avg in ("micro", "macro", "weighted", "samples"):
        row = [avg] + [
            f"{results[m][avg][k]:.3f}" for m in ("LR", "BERT") for k in ("precision", "recall", "f1")
        ]
        lines.append("| " + " | ".join(row) + " |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_perclass.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()