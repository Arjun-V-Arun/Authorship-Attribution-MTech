#!/usr/bin/env python3
"""
Work-level transformer authorship attribution for the Sayahna Malayalam corpora.

Important design choice:
  A transformer has a 512-token input limit. We therefore create windows INSIDE
  each work, but split by work_id first. All windows belonging to a work remain
  in exactly one fold. Predictions are aggregated back to work level by mean
  logits. This prevents the classic chunk leakage problem.

Default model:
  google/muril-base-cased

Alternative:
  ai4bharat/IndicBERTv2-SS

Example:
  python src/transformer_malayalam.py --root . \
      --model google/muril-base-cased --epochs 3 --folds 5 --repeats 1

For an RTX 4070-class laptop:
  --batch-size 2 --grad-accum 8 --max-length 512
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(root, corpus_name, text_dir, min_words, min_works):
    meta_path = os.path.join(
        root, corpus_name, "corpus", "metadata.csv"
    )
    text_path = os.path.join(
        root, corpus_name, "corpus", text_dir
    )
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
        if str(r.get("work_type", "")).lower() in {"poetry", "poem", "drama"}:
            continue
        rows.append({
            "work_id": work_id,
            "author": str(r["author"]),
            "text": text,
        })
    df = pd.DataFrame(rows)
    counts = df.author.value_counts()
    df = df[df.author.isin(counts[counts >= min_works].index)].reset_index(drop=True)
    return df


def make_windows(tokenizer, text, max_length, stride):
    """
    Return tokenizer-ready windows. No window identity is used for splitting.
    The parent work_id determines the fold.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    return [
        {k: enc[k][i] for k in ("input_ids", "attention_mask")}
        for i in range(enc["input_ids"].shape[0])
    ]


def run_fold(
    model_name, tokenizer, train_df, val_df, labels,
    args, device, seed, fold
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    label2id = {a: i for i, a in enumerate(labels)}

    class WindowDataset(Dataset):
        def __init__(self, frame):
            self.items = []
            for _, row in frame.iterrows():
                windows = make_windows(
                    tokenizer, row.text,
                    args.max_length, args.stride
                )
                for w in windows:
                    self.items.append((
                        row.work_id,
                        label2id[row.author],
                        w
                    ))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(batch):
        work_ids = [x[0] for x in batch]
        labels_b = torch.tensor([x[1] for x in batch])
        feats = {
            "input_ids": torch.stack([x[2]["input_ids"] for x in batch]),
            "attention_mask": torch.stack(
                [x[2]["attention_mask"] for x in batch]
            ),
        }
        return work_ids, labels_b, feats

    train_ds = WindowDataset(train_df)
    val_ds = WindowDataset(val_df)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=collate
    )

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=max(1, args.epochs)
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1
    best_work_preds = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (_, yb, feats) in enumerate(train_loader):
            yb = yb.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(**feats).logits
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Aggregate every window back to its parent work.
        model.eval()
        work_logits = defaultdict(list)
        work_gold = {}

        with torch.no_grad():
            for work_ids, yb, feats in val_loader:
                feats = {k: v.to(device) for k, v in feats.items()}
                logits = model(**feats).logits.float().cpu().numpy()
                for wid, yi, li in zip(work_ids, yb.tolist(), logits):
                    work_logits[wid].append(li)
                    work_gold[wid] = yi

        work_ids = sorted(work_logits)
        pred = np.asarray([
            np.mean(work_logits[w], axis=0).argmax()
            for w in work_ids
        ])
        gold = np.asarray([work_gold[w] for w in work_ids])
        f1 = f1_score(gold, pred, average="macro", zero_division=0)

        print(
            f"seed={seed} fold={fold} epoch={epoch}/{args.epochs} "
            f"work_macroF1={f1:.4f} windows_train={len(train_ds)} "
            f"windows_val={len(val_ds)}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_work_preds = (gold.copy(), pred.copy())

    return best_f1, best_work_preds


def run(args):
    import torch
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("model:", args.model)

    all_results = []

    for corpus_name in args.corpora:
        df = load_corpus(
            args.root, corpus_name, args.texts,
            args.min_words, args.min_works
        )
        labels = sorted(df.author.unique())
        label2id = {a: i for i, a in enumerate(labels)}
        y = np.asarray([label2id[a] for a in df.author])

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        for repeat in range(args.repeats):
            seed = args.seed + repeat
            seed_everything(seed)

            nfolds = min(args.folds, min(df.author.value_counts()))
            skf = StratifiedKFold(
                n_splits=nfolds, shuffle=True, random_state=seed
            )

            scores = []
            for fold, (tr, va) in enumerate(
                skf.split(np.zeros(len(df)), y), 1
            ):
                train_df = df.iloc[tr].reset_index(drop=True)
                val_df = df.iloc[va].reset_index(drop=True)

                f1, _ = run_fold(
                    args.model, tokenizer, train_df, val_df,
                    labels, args, device, seed, fold
                )
                scores.append(f1)

            mean = float(np.mean(scores))
            std = float(np.std(scores))
            print(
                f"\nRESULT {corpus_name} seed={seed}: "
                f"{mean:.4f} +/- {std:.4f}\n"
            )

            all_results.append({
                "corpus": corpus_name.replace("sayahna-", ""),
                "model": args.model,
                "seed": seed,
                "macro_f1": mean,
                "fold_std": std,
                "works": len(df),
                "authors": len(labels),
                "device": str(device),
            })

    out = os.path.join(
        args.root, "results", "malayalam", "transformer"
    )
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        os.path.join(out, "transformer_results.csv"), index=False
    )
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="+",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_strict")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--min-works", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260922)
    run(ap.parse_args())
