#!/usr/bin/env python3
"""
Rigor + ablation sprint (CPU-fast, reuses cached embeddings)
------------------------------------------------------------
Adds to the honest leave-one-book-out (LOBO) benchmark:
  * bootstrap 95% CIs on macro-F1 for every method
  * pairwise significance (paired bootstrap: is A>B real or noise?)
  * char n-gram ablation (2-4 / 3-5 / 4-6)
  * function-word stylometry view (top-frequency tokens ~ postpositions,
    pronouns, auxiliaries) + its leakage gap vs content features
  * frozen-embedding methods pulled from cached .npy if present

Run (from repo root):  python3 src/rigor_ablation.py --raw_dir data/raw
Needs: scikit-learn numpy pandas  (+ the embeddings/emb_*.npy caches you already built)
"""
import argparse, glob, os
from collections import Counter
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import f1_score

AUTHORS = ["dharamvir", "prem", "sarat", "vibhuti"]
SNIPPET_LEN = 500
RNG = np.random.default_rng(0)
N_BOOT = 1000

def build_snippets(raw_dir):
    texts, y, groups = [], [], []
    for author in AUTHORS:
        for bp in sorted(glob.glob(os.path.join(raw_dir, author, "*"))):
            book = f"{author}/{os.path.basename(bp)}"
            toks = open(bp, encoding="utf-8", errors="ignore").read().replace("\ufeff", "").split()
            for i in range(0, len(toks) - SNIPPET_LEN + 1, SNIPPET_LEN):
                texts.append(" ".join(toks[i:i+SNIPPET_LEN])); y.append(author); groups.append(book)
    return texts, np.array(y), np.array(groups)

def top_frequency_vocab(texts, k=200):
    c = Counter()
    for t in texts:
        c.update(t.split())
    return [w for w, _ in c.most_common(k)]

def text_pipe(vec):
    return Pipeline([("tfidf", vec), ("clf", LinearSVC(max_iter=5000))])

def emb_pipe():
    return Pipeline([("scale", StandardScaler()), ("clf", LinearSVC(max_iter=5000))])

def boot_ci(y_true, y_pred, n=N_BOOT):
    idx = np.arange(len(y_true))
    scores = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        scores.append(f1_score(y_true[s], y_pred[s], average="macro"))
    return np.percentile(scores, [2.5, 97.5])

def paired_sig(y_true, pa, pb, n=N_BOOT):
    """Fraction of bootstrap resamples where A does NOT beat B (one-sided p)."""
    idx = np.arange(len(y_true)); wins = 0
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        da = f1_score(y_true[s], pa[s], average="macro")
        db = f1_score(y_true[s], pb[s], average="macro")
        wins += (da > db)
    return 1.0 - wins / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    args = ap.parse_args()

    texts, y, groups = build_snippets(args.raw_dir)
    print(f"{len(texts)} snippets, {len(set(groups))} books\n")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    logo = LeaveOneGroupOut()
    fw_vocab = top_frequency_vocab(texts, 200)

    # method name -> (kind, spec). text kinds carry a vectorizer; emb kinds a cache path.
    methods = {
        "word 1-gram":     ("text", TfidfVectorizer(analyzer="word", ngram_range=(1,1), max_features=5000, sublinear_tf=True)),
        "func-word top200":("text", TfidfVectorizer(analyzer="word", vocabulary=fw_vocab, sublinear_tf=True)),
        "char 2-4":        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=5000, sublinear_tf=True)),
        "char 3-5":        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000, sublinear_tf=True)),
        "char 4-6":        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(4,6), max_features=5000, sublinear_tf=True)),
        "frozen mBERT":    ("emb",  "embeddings/emb_bert-base-multilingual-cased.npy"),
        "frozen MuRIL":    ("emb",  "embeddings/emb_google_muril-base-cased.npy"),
        "frozen IndicBERT":("emb",  "embeddings/emb_ai4bharat_indic-bert.npy"),
        "frozen MuRIL-large":("emb",  "embeddings/emb_google_muril-large-cased.npy"),
    }

    rows, oof = [], {}
    for name, (kind, spec) in methods.items():
        if kind == "text":
            leaky = cross_val_score(text_pipe(spec), texts, y, cv=skf, scoring="f1_macro", n_jobs=-1).mean()
            pred  = cross_val_predict(text_pipe(spec), texts, y, groups=groups, cv=logo, n_jobs=-1)
        else:
            if not os.path.exists(spec):
                print(f"  [skip] {name}: cache {spec} not found (run dl_baseline.py for it)")
                continue
            X = np.load(spec)
            leaky = cross_val_score(emb_pipe(), X, y, cv=skf, scoring="f1_macro", n_jobs=-1).mean()
            pred  = cross_val_predict(emb_pipe(), X, y, groups=groups, cv=logo, n_jobs=-1)
        honest = f1_score(y, pred, average="macro")
        lo, hi = boot_ci(y, pred)
        oof[name] = pred
        rows.append(dict(method=name, honest_macroF1=round(honest,4),
                         ci95=f"[{lo:.3f}, {hi:.3f}]", leaky_macroF1=round(leaky,4),
                         leakage_gap=round(leaky-honest,4)))
        print(f"  {name:18s} honest {honest:.4f}  95%CI [{lo:.3f},{hi:.3f}]  "
              f"leaky {leaky:.4f}  gap {leaky-honest:+.4f}")

    df = pd.DataFrame(rows).sort_values("honest_macroF1", ascending=False)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results_rigor.csv", index=False)
    print("\nRanking:\n", df.to_string(index=False))

    # pairwise significance for the headline comparisons, if available
    print("\nPairwise significance (one-sided p that A does NOT beat B):")
    for a, b in [("frozen MuRIL","char 3-5"), ("char 3-5","word 1-gram"),
                 ("char 3-5","func-word top200"), ("frozen MuRIL","frozen mBERT")]:
        if a in oof and b in oof:
            p = paired_sig(y, oof[a], oof[b])
            verdict = "significant" if p < 0.05 else "NOT significant"
            print(f"  {a} > {b}:  p={p:.3f}  ({verdict})")

    print("\nwrote results/results_rigor.csv")

if __name__ == "__main__":
    main()
