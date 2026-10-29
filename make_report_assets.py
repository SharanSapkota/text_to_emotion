"""Turn saved results into tables + figures for the report.

    python make_report_assets.py
Outputs go to report_assets/ : comparison table (md + csv), per-emotion F1 chart,
BERT training curve, and example predictions where the two models disagree.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_data, multi_hot


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_dir", default="results/lr")
    ap.add_argument("--bert_dir", default="results/bert")
    ap.add_argument("--data_dir", default="data/go_emotions")
    ap.add_argument("--out", default="report_assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lr, bert = load(Path(args.lr_dir) / "metrics.json"), load(Path(args.bert_dir) / "metrics.json")

    # ---- 1. overall comparison table -------------------------------------------------
    rows = []
    for avg in ("micro", "macro", "weighted", "samples"):
        rows.append(
            [avg]
            + [f"{lr[avg][k]:.3f}" for k in ("precision", "recall", "f1")]
            + [f"{bert[avg][k]:.3f}" for k in ("precision", "recall", "f1")]
        )
    header = ["Average", "LR P", "LR R", "LR F1", "BERT P", "BERT R", "BERT F1"]
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- 2. per-emotion F1 chart -----------------------------------------------------
    names = list(bert["per_class"].keys())
    order = sorted(names, key=lambda n: bert["per_class"][n]["f1"])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(y - 0.2, [lr["per_class"][n]["f1"] for n in order], 0.4, label="Logistic Regression")
    ax.barh(y + 0.2, [bert["per_class"][n]["f1"] for n in order], 0.4, label="BERT")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("F1-score (test set)")
    ax.set_title("Per-emotion F1: Logistic Regression vs BERT")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "per_emotion_f1.png", dpi=200)
    plt.close(fig)

    # per-emotion table (with support) for an appendix
    with open(out / "per_emotion.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emotion", "support", "LR_P", "LR_R", "LR_F1", "BERT_P", "BERT_R", "BERT_F1"])
        for n in names:
            a, b = lr["per_class"][n], bert["per_class"][n]
            w.writerow(
                [n, b["support"]]
                + [f"{a[k]:.3f}" for k in ("precision", "recall", "f1")]
                + [f"{b[k]:.3f}" for k in ("precision", "recall", "f1")]
            )

    # ---- 3. BERT training curve ------------------------------------------------------
    hist_path = Path(args.bert_dir) / "history.json"
    if hist_path.exists():
        hist = load(hist_path)
        ep = [h["epoch"] for h in hist]
        fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
        ax1.plot(ep, [h["train_loss"] for h in hist], "o-", color="tab:red", label="train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(ep, [h["val_macro_f1"] for h in hist], "s-", color="tab:blue", label="val macro-F1")
        ax2.set_ylabel("Validation macro-F1", color="tab:blue")
        ax1.set_xticks(ep)
        fig.tight_layout()
        fig.savefig(out / "bert_training_curve.png", dpi=200)
        plt.close(fig)

    # ---- 4. example predictions where the models disagree ----------------------------
    ds, label_names = load_data(args.data_dir)
    texts = ds["test"]["text"]
    y_true = multi_hot(ds["test"]["labels"], len(label_names))
    pred_lr = np.load(Path(args.lr_dir) / "test_probs.npy") >= lr["threshold"]
    pred_bt = np.load(Path(args.bert_dir) / "test_probs.npy") >= bert["threshold"]
    ok_lr = (pred_lr == y_true).all(axis=1)
    ok_bt = (pred_bt == y_true).all(axis=1)

    def fmt(i):
        gold = [label_names[j] for j in np.where(y_true[i])[0]]
        p_l = [label_names[j] for j in np.where(pred_lr[i])[0]]
        p_b = [label_names[j] for j in np.where(pred_bt[i])[0]]
        return f'"{texts[i]}"\n  gold: {gold}\n  LR:   {p_l}\n  BERT: {p_b}\n'

    lines = [f"Exact-match accuracy  LR: {ok_lr.mean():.3f}   BERT: {ok_bt.mean():.3f}\n"]
    lines.append("=== BERT correct, LR wrong ===")
    lines += [fmt(i) for i in np.where(ok_bt & ~ok_lr)[0][:10]]
    lines.append("=== LR correct, BERT wrong ===")
    lines += [fmt(i) for i in np.where(ok_lr & ~ok_bt)[0][:10]]
    lines.append("=== Both wrong ===")
    lines += [fmt(i) for i in np.where(~ok_lr & ~ok_bt)[0][:10]]
    (out / "example_predictions.txt").write_text("\n".join(lines))
    print(f"\nAll assets saved in {out}/")


if __name__ == "__main__":
    main()
