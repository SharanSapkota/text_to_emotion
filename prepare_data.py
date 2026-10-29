"""Run ONCE on the login node (needs internet). Downloads GoEmotions + the BERT model
and saves them to disk so the compute-node job can run fully offline.

    python prepare_data.py
    python prepare_data.py --model distilbert-base-uncased
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--data_dir", default="data/go_emotions")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()

    print("Downloading GoEmotions (simplified: 27 emotions + neutral, multi-label)...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    ds.save_to_disk(args.data_dir)
    print(ds)
    print("Labels:", ds["train"].features["labels"].feature.names)

    out = Path(args.models_dir) / args.model.replace("/", "_")
    print(f"Downloading {args.model} -> {out}")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    AutoModel.from_pretrained(args.model).save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    main()
