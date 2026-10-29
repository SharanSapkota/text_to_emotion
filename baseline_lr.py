"""Baseline: TF-IDF (word 1-2 grams) + one-vs-rest Logistic Regression on GoEmotions.

    python baseline_lr.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from common import compute_metrics, load_data, multi_hot, save_json, tune_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/go_emotions")
    ap.add_argument("--out", default="results/lr")
    ap.add_argument("--C", type=float, default=4.0, help="inverse regularisation strength")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    ds, names = load_data(args.data_dir)
    n = len(names)
    y_tr = multi_hot(ds["train"]["labels"], n)
    y_va = multi_hot(ds["validation"]["labels"], n)
    y_te = multi_hot(ds["test"]["labels"], n)

    t0 = time.time()
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000
    )
    X_tr = vec.fit_transform(ds["train"]["text"])
    X_va = vec.transform(ds["validation"]["text"])
    X_te = vec.transform(ds["test"]["text"])
    print(f"TF-IDF features: {X_tr.shape}")

    clf = OneVsRestClassifier(
        LogisticRegression(C=args.C, max_iter=1000, solver="liblinear", class_weight="balanced"),
        n_jobs=args.n_jobs,
    )
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    p_va = clf.predict_proba(X_va)
    p_te = clf.predict_proba(X_te)

    thr, val_f1 = tune_threshold(y_va, p_va)
    metrics = compute_metrics(y_te, p_te, thr, names)
    metrics.update(
        {
            "model": "TF-IDF + Logistic Regression",
            "val_macro_f1": val_f1,
            "train_seconds": train_time,
            "hyperparameters": {"C": args.C, "ngram_range": [1, 2], "min_df": 2},
        }
    )

    out = Path(args.out)
    save_json(metrics, out / "metrics.json")
    np.save(out / "val_probs.npy", p_va)
    np.save(out / "test_probs.npy", p_te)

    print(f"\nthreshold (tuned on val): {thr:.3f}   val macro-F1: {val_f1:.4f}")
    for avg in ("micro", "macro", "weighted"):
        m = metrics[avg]
        print(f"TEST {avg:9s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
