#!/usr/bin/env python3
"""
Malayalam Authorship Attribution: from-scratch neural baselines.

Runs:
  1. Character/grapheme CNN
  2. Word-level BiLSTM
  3. Word-level GRU

Evaluation is whole-work stratified CV. A work is never split across folds.
This is deliberately compatible with the existing Sayahna corpus layout:

  sayahna-fiction/corpus/metadata.csv
  sayahna-fiction/corpus/texts_strict/<work_id>.txt
  sayahna-essays/corpus/metadata.csv
  sayahna-essays/corpus/texts_strict/<work_id>.txt

Example:
  python src/neural_models.py --root . --texts texts_strict --epochs 12 --repeats 3

GPU:
  The script uses CUDA automatically when available.
"""

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

MALAYALAM = re.compile(r"[\u0D00-\u0D7F]")
SEED = 20260922


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(root, corpus_name, text_dir, min_words, min_works):
    corpus = os.path.join(root, corpus_name)
    meta_path = os.path.join(corpus, "corpus", "metadata.csv")
    text_path = os.path.join(corpus, "corpus", text_dir)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    if not os.path.isdir(text_path):
        raise FileNotFoundError(text_path)

    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    rows = []

    for _, r in meta.iterrows():
        work_id = str(r["work_id"])
        p = os.path.join(text_path, work_id + ".txt")
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8", errors="ignore").read().strip()
        if len(text.split()) < min_words:
            continue
        work_type = str(r.get("work_type", "")).lower()
        if work_type in {"poetry", "poem", "drama"}:
            continue
        rows.append({"work_id": work_id, "author": str(r["author"]), "text": text})

    df = pd.DataFrame(rows)
    counts = df.author.value_counts()
    keep = counts[counts >= min_works].index
    df = df[df.author.isin(keep)].reset_index(drop=True)

    if df.author.nunique() < 2:
        raise RuntimeError("Need at least two authors after filtering.")

    return df


def graphemes(text):
    # Combining-mark aware segmentation is important for Malayalam.
    try:
        import regex
        return regex.findall(r"\X", text)
    except ImportError:
        return list(text)


def build_grapheme_vocab(texts, max_vocab=5000, min_freq=2):
    c = Counter()
    for t in texts:
        c.update(graphemes(t))
    vocab = ["<PAD>", "<UNK>"]
    vocab += [x for x, n in c.most_common() if n >= min_freq][:max_vocab - 2]
    return {x: i for i, x in enumerate(vocab)}


def build_word_vocab(texts, max_vocab=30000, min_freq=2):
    c = Counter()
    for t in texts:
        c.update(t.split())
    vocab = ["<PAD>", "<UNK>"]
    vocab += [x for x, n in c.most_common() if n >= min_freq][:max_vocab - 2]
    return {x: i for i, x in enumerate(vocab)}


def encode_sequence(text, vocab, max_len, char_level):
    toks = graphemes(text) if char_level else text.split()
    ids = [vocab.get(t, 1) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


class TextDataset:
    def __init__(self, texts, labels, vocab, max_len, char_level):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.char_level = char_level

    def tensors(self):
        import torch
        X = np.asarray([
            encode_sequence(t, self.vocab, self.max_len, self.char_level)
            for t in self.texts
        ], dtype=np.int64)
        y = np.asarray(self.labels, dtype=np.int64)
        return torch.tensor(X), torch.tensor(y)


def make_model(name, vocab_size, n_classes, emb_dim, hidden_dim, dropout):
    import torch
    import torch.nn as nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, 128, k) for k in (3, 5, 7)
            ])
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(128 * 3, n_classes)

        def forward(self, x):
            x = self.emb(x).transpose(1, 2)
            z = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
            z = self.drop(torch.cat(z, dim=1))
            return self.fc(z)

    class RNN(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            R = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = R(
                emb_dim, hidden_dim, batch_first=True,
                bidirectional=True, dropout=0.0
            )
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, n_classes)

        def forward(self, x):
            e = self.emb(x)
            out, _ = self.rnn(e)
            mask = (x != 0).unsqueeze(-1)
            out = out.masked_fill(~mask, -1e4)
            pooled = out.max(dim=1).values
            return self.fc(self.drop(pooled))

    if name == "charcnn":
        return CharCNN()
    if name == "bilstm":
        return RNN("lstm")
    if name == "bigru":
        return RNN("gru")
    raise ValueError(name)


def train_one(model, train_loader, val_loader, epochs, lr, device):
    import torch
    import torch.nn as nn

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1
    best_pred = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                gold.extend(yb.tolist())

        f1 = f1_score(gold, preds, average="macro", zero_division=0)
        acc = accuracy_score(gold, preds)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n),
            "val_macro_f1": f1,
            "val_accuracy": acc
        })
        if f1 > best_f1:
            best_f1 = f1
            best_pred = np.asarray(preds)

    return best_f1, best_pred, history


def run(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    all_results = []

    for corpus_name in args.corpora:
        df = load_corpus(
            args.root, corpus_name, args.texts,
            args.min_words, args.min_works
        )
        labels = sorted(df.author.unique())
        label2id = {x: i for i, x in enumerate(labels)}
        y = np.asarray([label2id[x] for x in df.author], dtype=np.int64)

        print(f"\n{'='*80}\n{corpus_name}: {len(df)} works, {len(labels)} authors\n{'='*80}")

        for model_name in args.models:
            for repeat in range(args.repeats):
                seed = SEED + repeat
                seed_everything(seed)

                # Vocabulary is fit only on the training fold.
                skf = StratifiedKFold(
                    n_splits=min(args.folds, min(Counter(y).values())),
                    shuffle=True, random_state=seed
                )
                fold_scores = []

                for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
                    train_texts = df.iloc[tr].text.tolist()
                    val_texts = df.iloc[va].text.tolist()

                    char_level = model_name == "charcnn"
                    vocab = (
                        build_grapheme_vocab(train_texts, args.vocab_size)
                        if char_level
                        else build_word_vocab(train_texts, args.vocab_size)
                    )

                    train_ds = TextDataset(
                        train_texts, y[tr], vocab, args.max_len, char_level
                    )
                    val_ds = TextDataset(
                        val_texts, y[va], vocab, args.max_len, char_level
                    )
                    Xtr, ytr = train_ds.tensors()
                    Xva, yva = val_ds.tensors()

                    bs = args.batch_size
                    train_loader = DataLoader(
                        TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True
                    )
                    val_loader = DataLoader(
                        TensorDataset(Xva, yva), batch_size=bs, shuffle=False
                    )

                    model = make_model(
                        model_name, len(vocab), len(labels),
                        args.emb_dim, args.hidden_dim, args.dropout
                    ).to(device)

                    f1, pred, history = train_one(
                        model, train_loader, val_loader,
                        args.epochs, args.lr, device
                    )
                    fold_scores.append(f1)

                    print(
                        f"{model_name:8s} seed={seed} fold={fold} "
                        f"best_macroF1={f1:.4f}"
                    )

                    if args.save_checkpoints:
                        out = os.path.join(
                            args.root, "results", "malayalam",
                            "neural", corpus_name.replace("sayahna-", "")
                        )
                        os.makedirs(out, exist_ok=True)
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "vocab": vocab,
                                "labels": labels,
                                "model_name": model_name,
                                "history": history,
                                "seed": seed,
                                "fold": fold,
                            },
                            os.path.join(
                                out, f"{model_name}_seed{seed}_fold{fold}.pt"
                            )
                        )

                mean_f1 = float(np.mean(fold_scores))
                std_f1 = float(np.std(fold_scores))
                print(
                    f"RESULT {corpus_name} {model_name} seed={seed}: "
                    f"{mean_f1:.4f} +/- {std_f1:.4f}"
                )
                all_results.append({
                    "corpus": corpus_name.replace("sayahna-", ""),
                    "model": model_name,
                    "seed": seed,
                    "cv_macro_f1": mean_f1,
                    "fold_std": std_f1,
                    "n_works": len(df),
                    "n_authors": len(labels),
                    "device": str(device),
                })

    outdir = os.path.join(args.root, "results", "malayalam", "neural")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        os.path.join(outdir, "neural_results.csv"), index=False
    )

    summary = (
        pd.DataFrame(all_results)
        .groupby(["corpus", "model"])
        .agg(
            macro_f1_mean=("cv_macro_f1", "mean"),
            macro_f1_std_across_seeds=("cv_macro_f1", "std"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(outdir, "neural_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="+",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_strict")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--min-works", type=int, default=5)
    ap.add_argument("--models", nargs="+",
                    default=["charcnn", "bilstm", "bigru"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1200,
                    help="graphemes for CharCNN; whitespace tokens for RNNs")
    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save-checkpoints", action="store_true")
    run(ap.parse_args())
