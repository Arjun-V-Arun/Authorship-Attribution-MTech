#!/usr/bin/env python3
"""
Deep-learning baseline: frozen transformer embeddings + linear head
-------------------------------------------------------------------
Directly comparable to the classical char/word baselines because it uses the
SAME book-disjoint (leave-one-book-out) evaluation and the SAME LinearSVC head.
Only the FEATURES change: instead of TF-IDF, each 500-token snippet is encoded
once (no fine-tuning) by a frozen pretrained transformer and mean-pooled.

Why frozen, not fine-tuned: fine-tuning under leave-one-book-out means training
the transformer 15x (once per held-out book). That is impractical on CPU. Frozen
feature-extraction needs a single forward pass per snippet (cached to disk), so a
re-run is instant and the whole thing fits comfortably on a CPU laptop.

Scientific question: do content-heavy pretrained embeddings beat simple char
n-grams once topic leakage is removed? (Not obvious -- they may not.)

Models to try (swap with --model):
  bert-base-multilingual-cased   (mBERT, default, robust)
  google/muril-base-cased        (MuRIL, usually strongest on Indic)
  ai4bharat/indic-bert           (IndicBERT, smallest/fastest on CPU)

Install (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers scikit-learn numpy

Run (from repo root):
  python3 src/dl_baseline.py --raw_dir data/raw --model bert-base-multilingual-cased
"""
import argparse, glob, os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AUTHORS = ["dharamvir", "prem", "sarat", "vibhuti"]
SNIPPET_LEN = 500

def build_snippets(raw_dir):
    texts, y, groups = [], [], []
    for author in AUTHORS:
        for book_path in sorted(glob.glob(os.path.join(raw_dir, author, "*"))):
            book_id = f"{author}/{os.path.basename(book_path)}"
            with open(book_path, encoding="utf-8", errors="ignore") as f:
                toks = f.read().replace("\ufeff", "").split()
            for i in range(0, len(toks) - SNIPPET_LEN + 1, SNIPPET_LEN):
                texts.append(" ".join(toks[i:i + SNIPPET_LEN]))
                y.append(author)
                groups.append(book_id)
    return texts, np.array(y), np.array(groups)

def encode(texts, model_name, cache_path, max_len=256, batch_size=16):
    """Mean-pooled frozen embeddings, cached to .npy so re-runs are instant."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {model_name} (first run downloads weights)...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Encoding {len(texts)} snippets on {device} "
          f"(max_len={max_len}, batch={batch_size})...")

    embs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state           # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1)      # (B, T, 1)
            summed = (out * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            embs.append((summed / counts).cpu().numpy())    # masked mean pool
            if (start // batch_size) % 10 == 0:
                print(f"  {start+len(batch)}/{len(texts)}")
    X = np.vstack(embs)
    if cache_path:
        np.save(cache_path, X)
        print(f"Cached embeddings -> {cache_path}")
    return X

def head():
    # scale then linear SVM; matches the classical head, fair comparison
    return Pipeline([("scale", StandardScaler()), ("clf", LinearSVC())])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, y, groups = build_snippets(args.raw_dir)
    print(f"Built {len(texts)} snippets from {len(set(groups))} books\n")

    os.makedirs("embeddings", exist_ok=True)
    cache = f"embeddings/emb_{args.model.replace('/', '_')}.npy"
    X = encode(texts, args.model, cache, args.max_len, args.batch_size)
    print(f"Embedding matrix: {X.shape}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    logo = LeaveOneGroupOut()

    print("=" * 70)
    print(f"FROZEN {args.model} embeddings + LinearSVC")
    print("=" * 70)
    leaky = cross_val_score(head(), X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  [LEAKY ] shuffled 5-fold macro-F1 = "
          f"{leaky.mean():.4f} +/- {leaky.std():.4f}")
    pred = cross_val_predict(head(), X, y, groups=groups, cv=logo, n_jobs=-1)
    honest = f1_score(y, pred, average="macro")
    print(f"  [HONEST] leave-one-book-out macro-F1 = {honest:.4f}")
    print(f"  --> leakage inflation = {leaky.mean()-honest:+.4f}\n")
    print("  Book-disjoint report:")
    print(classification_report(y, pred, labels=AUTHORS, digits=4, zero_division=0))
    print("  Confusion matrix (rows=true, cols=pred), order:", AUTHORS)
    print(confusion_matrix(y, pred, labels=AUTHORS))

    print("\nCompare against your classical honest numbers:")
    print("  word TF-IDF  LOBO macro-F1 = 0.5162")
    print("  char n-gram  LOBO macro-F1 = 0.5707")

if __name__ == "__main__":
    main()
