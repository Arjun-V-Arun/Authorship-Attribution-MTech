#!/usr/bin/env python3
"""
Fine-tune a transformer for Hindi authorship attribution (CPU-friendly, resumable)
----------------------------------------------------------------------------------
Book-disjoint split: for each author, one whole book is held out for validation
(so train/val never share a book -> no topic leakage). Every author appears in
both train and val.

Crash-safety / unattended-run features (as requested):
  * Per-epoch metrics appended to an .xlsx AND a .csv backup after EVERY epoch
    (atomic write: temp file -> os.replace), so a crash never corrupts results.
  * Model + optimizer + history checkpointed after every epoch
    (ckpt_<run>_latest.pt and ckpt_<run>_epochN.pt). Resume with --resume.
  * Training graph (loss + val macro-F1 vs epoch) redrawn and saved every epoch,
    so even an interrupted run leaves a usable PNG.

Install (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers scikit-learn pandas openpyxl matplotlib
  # IndicBERT/MuRIL also need: pip install sentencepiece

Run (from repo root):
  python3 src/train_finetune.py --raw_dir data/raw \
      --model distilbert-base-multilingual-cased --epochs 4 --run_name distil_ft
Resume after interruption:
  python3 src/train_finetune.py --raw_dir data/raw \
      --model distilbert-base-multilingual-cased --run_name distil_ft --resume
"""
import argparse, glob, os, time, json
import numpy as np
import pandas as pd

AUTHORS = ["dharamvir", "prem", "sarat", "vibhuti"]
LABEL2ID = {a: i for i, a in enumerate(AUTHORS)}
SNIPPET_LEN = 500

# ---------- data ----------
def build_snippets(raw_dir):
    rows = []
    for author in AUTHORS:
        for bp in sorted(glob.glob(os.path.join(raw_dir, author, "*"))):
            book_id = f"{author}/{os.path.basename(bp)}"
            with open(bp, encoding="utf-8", errors="ignore") as f:
                toks = f.read().replace("\ufeff", "").split()
            for i in range(0, len(toks) - SNIPPET_LEN + 1, SNIPPET_LEN):
                rows.append((" ".join(toks[i:i+SNIPPET_LEN]), author, book_id))
    return rows

def book_disjoint_split(rows, val_book_index=0):
    """Hold out one whole book per author (index into that author's sorted books)."""
    books_by_author = {}
    for _, a, b in rows:
        books_by_author.setdefault(a, set()).add(b)
    val_books = set()
    for a, bs in books_by_author.items():
        bs = sorted(bs)
        val_books.add(bs[val_book_index % len(bs)])
    train = [(t, a) for (t, a, b) in rows if b not in val_books]
    val   = [(t, a) for (t, a, b) in rows if b in val_books]
    return train, val, sorted(val_books)

# ---------- crash-safe logging (torch-free, unit-testable) ----------
def _atomic_write_df(df, path):
    root, ext = os.path.splitext(path)      # keep real ext so engine is inferred
    tmp = f"{root}.writing{ext}"
    if ext == ".xlsx":
        df.to_excel(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def log_epoch(history, out_dir, run_name):
    df = pd.DataFrame(history)
    _atomic_write_df(df, os.path.join(out_dir, f"results_{run_name}.xlsx"))
    _atomic_write_df(df, os.path.join(out_dir, f"results_{run_name}.csv"))  # backup

def save_graph(history, out_dir, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_macro_f1"], "s-", color="tab:blue", label="val macro-F1")
    if "val_acc" in df:
        ax2.plot(df["epoch"], df["val_acc"], "^--", color="tab:green", label="val acc")
    ax2.set_ylabel("val score", color="tab:blue"); ax2.set_ylim(0, 1)
    ax1.set_title(f"{run_name}: training curve")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"curve_{run_name}.png"), dpi=120)
    plt.close(fig)

# ---------- training (torch imported lazily) ----------
def run(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import f1_score, accuracy_score

    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = build_snippets(args.raw_dir)
    train, val, val_books = book_disjoint_split(rows, args.val_book_index)
    print(f"{len(rows)} snippets | train {len(train)} | val {len(val)}")
    print(f"held-out val books: {val_books}\n")

    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            text, author = self.data[i]
            enc = tok(text, truncation=True, max_length=args.max_len,
                      padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, LABEL2ID[author]

    def collate(b):
        feats = {k: torch.stack([x[0][k] for x in b]) for k in b[0][0]}
        labels = torch.tensor([x[1] for x in b])
        return feats, labels

    tr_loader = DataLoader(DS(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(DS(val), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(AUTHORS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history, start_epoch = [], 1
    latest = os.path.join(args.out_dir, f"ckpt_{args.run_name}_latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        history = ck["history"]; start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']} -> starting {start_epoch}\n")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        e0 = time.time()
        model.train(); tot, correct, loss_sum = 0, 0, 0.0
        for feats, labels in tr_loader:
            opt.zero_grad()
            logits = model(**feats).logits
            loss = loss_fn(logits, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item(); tot += len(labels)
        train_loss, train_acc = loss_sum / tot, correct / tot

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for feats, labels in va_loader:
                preds += model(**feats).logits.argmax(1).tolist(); gts += labels.tolist()
        val_f1 = f1_score(gts, preds, average="macro")
        val_acc = accuracy_score(gts, preds)
        dt = time.time() - e0

        history.append(dict(
            epoch=epoch, train_loss=round(train_loss, 4), train_acc=round(train_acc, 4),
            val_acc=round(val_acc, 4), val_macro_f1=round(val_f1, 4),
            epoch_time_sec=round(dt, 1), cumulative_min=round((time.time()-t0)/60, 2),
            model=args.model, lr=args.lr, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"epoch {epoch}/{args.epochs}  loss {train_loss:.4f}  "
              f"val_f1 {val_f1:.4f}  val_acc {val_acc:.4f}  ({dt:.0f}s)")

        # --- persist EVERYTHING after each epoch ---
        log_epoch(history, args.out_dir, args.run_name)
        save_graph(history, args.out_dir, args.run_name)
        ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                  epoch=epoch, history=history)
        torch.save(ck, latest)
        torch.save(ck, os.path.join(args.out_dir, f"ckpt_{args.run_name}_epoch{epoch}.pt"))

    print(f"\nDone. Best val macro-F1 = "
          f"{max(h['val_macro_f1'] for h in history):.4f}")
    print(f"Compare: word 0.5162 | char 0.5707 | frozen mBERT 0.5274")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--val_book_index", type=int, default=0)
    ap.add_argument("--run_name", default="ft_run")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--resume", action="store_true")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
