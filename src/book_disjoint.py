#!/usr/bin/env python3
"""
Book-disjoint authorship evaluation
-----------------------------------
Exposes the topic/book leakage in the 2014 setup and measures the HONEST,
leakage-free performance by never letting snippets from one book appear in
both train and test.

Two evaluation regimes on identical features:
  1. Shuffled StratifiedKFold   -> the leaky baseline (snippet-level shuffle).
  2. Leave-One-Book-Out (LOBO)  -> train on all OTHER books, test on the
                                   held-out whole book (LeaveOneGroupOut on book id).

Data: raw per-book files under <raw_dir>/<author>/<book>.
      Each book is whitespace/newline tokenized and cut into 500-token snippets,
      each snippet tagged with (author, book_id).

Run (from repo root):  python3 src/book_disjoint.py --raw_dir data/raw
"""
import argparse, glob, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

AUTHORS = ["dharamvir", "prem", "sarat", "vibhuti"]
SNIPPET_LEN = 500

def build_snippets(raw_dir):
    """Cut each raw book into 500-token snippets; tag with author and book id."""
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

def make_pipe(view):
    if view == "word":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                              max_features=5000, sublinear_tf=True)
    else:  # char
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              max_features=5000, sublinear_tf=True)
    return Pipeline([("tfidf", vec), ("clf", LinearSVC())])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, y, groups = build_snippets(args.raw_dir)
    books = sorted(set(groups))
    print(f"Built {len(texts)} snippets from {len(books)} books\n")
    for a in AUTHORS:
        nb = len({g for g in groups if g.startswith(a + "/")})
        print(f"  {a:12s}: {(y==a).sum():4d} snippets across {nb} books")
    print()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    logo = LeaveOneGroupOut()

    for view in ["word", "char"]:
        print("=" * 70)
        print(f"FEATURE VIEW: {view}")
        print("=" * 70)

        # 1. Leaky baseline: shuffled stratified CV
        leaky = cross_val_score(make_pipe(view), texts, y, cv=skf,
                                scoring="f1_macro", n_jobs=-1)
        print(f"  [LEAKY ] shuffled 5-fold macro-F1 = "
              f"{leaky.mean():.4f} +/- {leaky.std():.4f}")

        # 2. Honest: leave-one-book-out, aggregate out-of-fold predictions
        pred = cross_val_predict(make_pipe(view), texts, y,
                                 groups=groups, cv=logo, n_jobs=-1)
        from sklearn.metrics import f1_score
        honest = f1_score(y, pred, average="macro")
        print(f"  [HONEST] leave-one-book-out macro-F1 = {honest:.4f}")
        print(f"  --> leakage inflation = {leaky.mean()-honest:+.4f}\n")

        print(f"  Book-disjoint classification report [{view}]:")
        print(classification_report(y, pred, labels=AUTHORS, digits=4, zero_division=0))
        print("  Confusion matrix (rows=true, cols=pred), order:", AUTHORS)
        print(confusion_matrix(y, pred, labels=AUTHORS))
        print()

if __name__ == "__main__":
    main()
