#!/usr/bin/env python3
"""
Hindi Authorship Attribution - modern reproduction & baseline
-------------------------------------------------------------
Reproduces (and cleans up) the CS365-2014 IIT-K project by Shetty & Anand.

Data: data/snippets_2014/<author>.split/*  -- each file = 500 tokens, one token/line.
Authors: dharamvir, prem, sarat, vibhuti  (Tagore already excluded upstream:
         his texts are multi-translator translations, i.e. heterogeneous).

What this script adds over the 2014 version:
  * Proper stratified train/test split + 5-fold cross-validation (they used one 5:2 split).
  * A full 4-class confusion matrix + macro-F1 (they only did one-vs-all per author).
  * Two feature *views* to expose the topic-vs-style confound:
       (A) word TF-IDF  -> reproduces original (captures topic AND style)
       (B) char n-gram TF-IDF (3-5) -> more style-oriented, standard in stylometry
  * Three classifiers (LinearSVC, LogisticRegression, RandomForest).

Run (from repo root):  python3 src/baseline.py --data_dir data/snippets_2014
"""
import argparse, glob, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

AUTHORS = ["dharamvir", "prem", "sarat", "vibhuti"]

def load_corpus(data_dir):
    """Return (texts, labels). Each snippet file is one token per line -> join with spaces."""
    texts, labels = [], []
    for author in AUTHORS:
        folder = os.path.join(data_dir, f"{author}.split")
        files = sorted(glob.glob(os.path.join(folder, "*")))
        if not files:
            print(f"WARNING: no files in {folder}", file=sys.stderr)
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                # strip BOM + blank lines, rejoin tokens as whitespace-separated text
                toks = [t.strip().lstrip("\ufeff") for t in f if t.strip()]
            if toks:
                texts.append(" ".join(toks))
                labels.append(author)
    return texts, np.array(labels)

def make_vectorizer(view):
    if view == "word":       # reproduces original content-word BOW (topic + style)
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                               max_features=5000, sublinear_tf=True)
    if view == "char":       # char n-grams: closer to true style, topic-robust
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, sublinear_tf=True)
    raise ValueError(view)

def make_models():
    return {
        "LinearSVC":   LinearSVC(),
        "LogReg":      LogisticRegression(max_iter=2000),
        "RandomForest":RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/snippets_2014")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, y = load_corpus(args.data_dir)
    print(f"Loaded {len(texts)} snippets across {len(set(y))} authors")
    for a in AUTHORS:
        print(f"  {a:12s}: {(y==a).sum()} snippets")
    print()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for view in ["word", "char"]:
        print("=" * 66)
        print(f"FEATURE VIEW: {view}")
        print("=" * 66)
        for name, clf in make_models().items():
            pipe = Pipeline([("tfidf", make_vectorizer(view)), ("clf", clf)])
            scores = cross_val_score(pipe, texts, y, cv=cv,
                                     scoring="f1_macro", n_jobs=-1)
            print(f"  {name:13s}  5-fold macro-F1 = "
                  f"{scores.mean():.4f} +/- {scores.std():.4f}")

        # one detailed held-out report per view, using LinearSVC
        Xtr, Xte, ytr, yte = train_test_split(
            texts, y, test_size=0.3, stratify=y, random_state=args.seed)
        pipe = Pipeline([("tfidf", make_vectorizer(view)), ("clf", LinearSVC())])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        print(f"\n  --- LinearSVC held-out (30% test) report [{view}] ---")
        print(classification_report(yte, pred, digits=4))
        print("  Confusion matrix (rows=true, cols=pred), order:", AUTHORS)
        print(confusion_matrix(yte, pred, labels=AUTHORS))
        print()

if __name__ == "__main__":
    main()
