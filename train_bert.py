"""Fine-tune BERT (or any HF encoder) for multi-label emotion classification on GoEmotions.
Plain PyTorch loop (no Trainer) so it works across transformers versions.

    python train_bert.py --model_path models/bert-base-uncased --epochs 3
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import compute_metrics, load_data, multi_hot, save_json, tune_threshold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EncodedSet(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, y, max_len):
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.ids, self.mask, self.y = enc["input_ids"], enc["attention_mask"], y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.y[i]}


def make_collate(tokenizer):
    def collate(batch):
        feats = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        out = tokenizer.pad(feats, return_tensors="pt")
        out["labels"] = torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float)
        return out

    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    chunks = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        chunks.append(torch.sigmoid(model(**batch).logits.float()).cpu().numpy())
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/bert-base-uncased")
    ap.add_argument("--data_dir", default="data/go_emotions")
    ap.add_argument("--out", default="results/bert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    ds, names = load_data(args.data_dir)
    n = len(names)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    collate = make_collate(tok)

    y_tr = multi_hot(ds["train"]["labels"], n)
    y_va = multi_hot(ds["validation"]["labels"], n)
    y_te = multi_hot(ds["test"]["labels"], n)

    train_set = EncodedSet(tok, ds["train"]["text"], y_tr, args.max_len)
    val_set = EncodedSet(tok, ds["validation"]["text"], y_va, args.max_len)
    test_set = EncodedSet(tok, ds["test"]["text"], y_te, args.max_len)

    kw = dict(collate_fn=collate, num_workers=args.num_workers)
    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kw)
    val_dl = DataLoader(val_set, batch_size=128, **kw)
    test_dl = DataLoader(test_set, batch_size=128, **kw)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=n, problem_type="multi_label_classification"
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for nm, p in model.named_parameters() if not any(x in nm for x in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for nm, p in model.named_parameters() if any(x in nm for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and amp_dtype == torch.float16)

    history, best = [], {"val_macro_f1": -1.0}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            steps += 1

        p_va = predict(model, val_dl, device)
        thr, val_f1 = tune_threshold(y_va, p_va)
        train_loss = running / steps
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1, "threshold": thr}
        )
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f} (thr={thr:.3f})  elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

        if val_f1 > best["val_macro_f1"]:  # keep the best epoch's predictions (selected on VAL only)
            best = {
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "threshold": thr,
                "val_probs": p_va,
                "test_probs": predict(model, test_dl, device),
            }

    metrics = compute_metrics(y_te, best["test_probs"], best["threshold"], names)
    metrics.update(
        {
            "model": args.model_path,
            "best_epoch": best["epoch"],
            "val_macro_f1": best["val_macro_f1"],
            "train_seconds": time.time() - t0,
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_len": args.max_len,
                "seed": args.seed,
            },
        }
    )

    out = Path(args.out)
    save_json(metrics, out / "metrics.json")
    save_json(history, out / "history.json")
    np.save(out / "val_probs.npy", best["val_probs"])
    np.save(out / "test_probs.npy", best["test_probs"])

    print(f"\nBest epoch {best['epoch']}, threshold {best['threshold']:.3f}")
    for avg in ("micro", "macro", "weighted"):
        m = metrics[avg]
        print(f"TEST {avg:9s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
