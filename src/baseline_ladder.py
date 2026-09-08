#!/usr/bin/env python3
"""
baseline_ladder.py — the model ladder, on cleaned text only.

Every result so far came from ONE model: char 3-5 gram TF-IDF + linear SVM.
This runs a ladder where each rung answers a question, not just adds a model.

    FLOOR      stratified random        what does guessing get?
    L1         Burrows' Delta           does function-word stylometry, built on
                                        English assumptions, transfer to an
                                        agglutinative language?
    L1         char 3-5 gram + SVM      the field's strongest general baseline
    L1         char 2-4 / char 4-6      how sensitive is that to n-gram size?
    L1         word 1-2 gram + SVM      what is lost by forcing whole-word units
                                        on agglutinative text?
    L1         stylometric + LogReg     do 30 interpretable, Malayalam-specific
                                        features carry the signal?
    L2         char TF-IDF + NB / LR / nearest centroid
                                        is the SVM doing the work, or the features?

Input is text only. Metadata supplies the author label and nothing else.

Usage:
    pip install pandas scikit-learn matplotlib regex
    python baseline_ladder.py --root . --texts texts_clean
    python baseline_ladder.py --root . --texts texts --corpora sayahna-fiction
    python baseline_ladder.py --root . --texts texts_strict --min-works 5

SUPERSEDED — kept for provenance only. See baseline_ladder-v2.py for the
current version; see src/README.md for why both are kept.

Output, per corpus, in <corpus>/corpus/prelim/ (all "_v1"-suffixed so a
re-run can't overwrite baseline_ladder-v2.py's current output):
    baseline_ladder_v1.txt         the table plus a per-author report
    confusion_<model>_v1.png       confusion matrix for the best model
    baseline_ladder_v1.csv         machine-readable, for the LaTeX table
And under --root/results/malayalam/:
    baseline-ladder-v1-superseded.md   both corpora side by side
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)

try:
    import regex as _re2
    HAVE_REGEX = True
except ImportError:
    HAVE_REGEX = False

SEED = 20260901

# ===================== CANONICAL EVALUATION SETTINGS =====================
# These MUST match low_baselines.py. Two scripts reporting different numbers
# for the same baseline is not something you can defend in a paper.
CANON = dict(sublinear_tf=True, min_df=2, max_features=100_000)
CANON_SVC = dict(C=1.0, class_weight="balanced", max_iter=5000)
MIN_DOC_WORDS = 100        # documents shorter than this are dropped
MIN_WORKS_PER_AUTHOR = 5   # authors with fewer works are dropped
MAX_FOLDS = 5
# =========================================================================
# sklearn's default word pattern shatters Malayalam: vowel signs and virama are
# combining marks that \w does not match, so സഞ്ജയൻ -> ['സഞ','ജയൻ'].
ML_TOKEN = r"\S+"

MAL = re.compile(r"[\u0D00-\u0D7F]")
VIRAMA = "\u0D4D"
CHILLU = re.compile(r"[\u0D7A-\u0D7F]")
SENT_END = re.compile(r"[.!?\u0964\u0965]+")
PUNCT = list(".,;:!?\u2018\u2019\u201c\u201d()\u2014\u2013-\u2026\u0964")


def graphemes(s):
    return _re2.findall(r"\X", s) if HAVE_REGEX else list(s)


# ------------------------------------------------------ Burrows' Delta

class BurrowsDelta(BaseEstimator, ClassifierMixin):
    """
    The canonical stylometric method (Burrows 2002): z-score the relative
    frequencies of the k most frequent words across the corpus, then assign a
    document to the author whose centroid is nearest in Manhattan distance.

    Included because its assumption -- that grammatical relations are carried
    by free function words -- is exactly what an agglutinative language breaks.
    Its performance here is a finding, not a formality.
    """

    def __init__(self, k=300):
        self.k = k

    def fit(self, X, y):
        freq = Counter()
        for d in X:
            freq.update(d.split())
        self.vocab_ = [w for w, _ in freq.most_common(self.k)]
        M = self._mat(X)
        self.mu_ = M.mean(0)
        self.sd_ = M.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (M - self.mu_) / self.sd_
        self.classes_ = np.array(sorted(set(y)))
        y = np.asarray(y)
        self.cent_ = np.vstack([Z[y == c].mean(0) for c in self.classes_])
        return self

    def _mat(self, X):
        idx = {w: i for i, w in enumerate(self.vocab_)}
        M = np.zeros((len(X), len(self.vocab_)))
        for r, d in enumerate(X):
            toks = d.split()
            n = max(1, len(toks))
            for t in toks:
                j = idx.get(t)
                if j is not None:
                    M[r, j] += 1
            M[r] /= n
        return M

    def predict(self, X):
        Z = (self._mat(X) - self.mu_) / self.sd_
        d = np.abs(Z[:, None, :] - self.cent_[None, :, :]).sum(2)
        return self.classes_[d.argmin(1)]


# -------------------------------------------- interpretable stylometry

def stylometric(docs):
    """~30 features you can read and argue about, several Malayalam-specific."""
    rows = []
    for d in docs:
        toks = d.split()
        n = max(1, len(toks))
        gr = graphemes(d)
        ng = max(1, len(gr))
        wl = [len(graphemes(t)) for t in toks] or [0]
        sents = [s for s in SENT_END.split(d) if s.strip()]
        sl = [len(s.split()) for s in sents] or [0]
        paras = [p for p in d.split("\n\n") if p.strip()]
        cnt = Counter(toks)
        hapax = sum(1 for v in cnt.values() if v == 1)

        f = [
            np.mean(wl), np.std(wl), np.percentile(wl, 90), max(wl),
            np.mean(sl), np.std(sl), np.median(sl),
            len(cnt) / n,                       # type-token ratio
            hapax / max(1, len(cnt)),           # hapax rate
            len(sents) / n * 100,
            np.mean([len(p.split()) for p in paras]) if paras else 0,
            d.count(VIRAMA) / ng * 100,         # conjunct / virama density
            len(CHILLU.findall(d)) / ng * 100,  # atomic chillu rate
            sum(1 for t in toks if len(graphemes(t)) >= 12) / n * 100,  # long words
            sum(1 for t in toks if len(graphemes(t)) <= 3) / n * 100,   # short words
            len(MAL.findall(d)) / ng * 100,     # Malayalam script share
            sum(c.isdigit() for c in d) / ng * 100,
            sum(c.isascii() and c.isalpha() for c in d) / ng * 100,     # Latin
        ]
        f += [d.count(p) / ng * 100 for p in PUNCT]
        rows.append(f)
    return np.asarray(rows, dtype=float)


# -------------------------------------------------------------- ladder

def build_ladder():
    tf = dict(CANON)
    return [
        ("FLOOR  stratified random", None,
         "what guessing gets"),

        ("L1  Burrows' Delta (300 MFW)", BurrowsDelta(k=300),
         "does English-style function-word stylometry transfer?"),

        ("L1  char 3-5 gram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **tf),
            LinearSVC(**CANON_SVC)),
         "the reference; every earlier number came from this"),

        ("L1  char 2-4 gram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), **tf),
            LinearSVC(**CANON_SVC)),
         "shorter n-grams: more morphology, less lexis"),

        ("L1  char 4-6 gram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), **tf),
            LinearSVC(**CANON_SVC)),
         "longer n-grams: closer to whole words"),

        ("L1  word unigram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                            token_pattern=ML_TOKEN, lowercase=False, **tf),
            LinearSVC(**CANON_SVC)),
         "lexical choice alone: which words, order ignored"),

        ("L1  word bigram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(2, 2),
                            token_pattern=ML_TOKEN, lowercase=False, **tf),
            LinearSVC(**CANON_SVC)),
         "phrasing alone: adjacent word pairs, no unigrams"),

        ("L1  word 1-2 gram + SVM", make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                            token_pattern=ML_TOKEN, lowercase=False, **tf),
            LinearSVC(**CANON_SVC)),
         "both together: does the bigram add anything over unigrams?"),

        ("L1  stylometric + LogReg", make_pipeline(
            FunctionTransformer(stylometric, validate=False),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
         "can ~30 readable features do it?"),

        ("L2  char 3-5 + LogReg", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **tf),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
         "is the SVM or the features doing the work?"),

        ("L2  char 3-5 + ComplementNB", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **tf),
            ComplementNB(alpha=0.3)),
         "generative comparison; Complement not Multinomial, which cannot "
         "weight classes and collapses onto the large ones"),

        ("L2  char 3-5 + NearestCentroid", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **tf),
            NearestCentroid()),
         "distance-based, the modern analogue of Delta"),
    ]


# ---------------------------------------------------------------- data

def load(corpus_dir, textdir, min_works):
    meta = os.path.join(corpus_dir, "corpus", "metadata.csv")
    tdir = os.path.join(corpus_dir, "corpus", textdir)
    if not os.path.exists(meta):
        return None, f"no metadata.csv in {corpus_dir}"
    if not os.path.isdir(tdir):
        return None, f"no corpus/{textdir} in {corpus_dir}"
    df = pd.read_csv(meta, dtype=str).fillna("")
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(tdir, str(r.work_id) + ".txt")
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read().strip()
        if len(t.split()) < MIN_DOC_WORDS:
            continue
        rows.append({"work_id": str(r.work_id), "author": str(r.author),
                     "text": t,
                     "work_type": str(r.get("work_type", "")).lower()})
    d = pd.DataFrame(rows)
    for c in ("work_id", "author", "text", "work_type"):
        if c in d:
            d[c] = d[c].astype(object)
    if d.empty:
        return None, "no usable texts"

    # A work_id appearing twice, or two works with identical text, means one
    # file is carrying two different author labels. That makes the task
    # unlearnable and is worth catching loudly rather than reading as a hard task.
    dup_id = d.work_id.duplicated(keep=False)
    if dup_id.any():
        bad = d[dup_id].groupby("work_id").author.nunique()
        conflict = bad[bad > 1]
        print(f"  !! {int(dup_id.sum())} duplicate work_id rows"
              + (f", {len(conflict)} with CONFLICTING authors" if len(conflict) else ""))
        d = d.drop_duplicates(subset="work_id", keep="first")
    d["_h"] = d.text.map(lambda t: hash(t))
    clash = d.groupby("_h").author.nunique()
    clash = clash[clash > 1]
    if len(clash):
        print(f"  !! {len(clash)} identical texts carry more than one author label "
              "— labels are unreliable, fix the corpus before trusting any score")
    d = d.drop(columns=["_h"])
    d = d[~d.work_type.isin(["poetry", "poem", "drama"])]
    vc = d.author.value_counts()
    d = d[d.author.isin(vc[vc >= min_works].index)].reset_index(drop=True)
    if d.author.nunique() < 2:
        return None, f"fewer than 2 authors with >= {min_works} works"
    return d, None


def confusion_png(y, pred, labels, path, title):
    cm = confusion_matrix(y, pred, labels=labels)
    cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.62),
                                    max(6, len(labels) * 0.55)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.045, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def run_corpus(name, corpus_dir, a):
    out = []

    def w(s=""):
        print(s, flush=True)
        out.append(s)

    d, err = load(corpus_dir, a.texts, a.min_works)
    if d is None:
        print(f"  skipping {name}: {err}")
        return None

    # .values on a pandas string column can return a StringArray or
    # ArrowStringArray; some sklearn versions fail to index those with an
    # integer array ("only integer scalar arrays can be converted to a scalar
    # index"). A plain object-dtype numpy array works on every version.
    X = np.asarray(d["text"].tolist(), dtype=object)
    y = np.asarray(d["author"].tolist(), dtype=object)
    labels = sorted(set(y))
    chance = 1.0 / len(labels)
    k = int(min(MAX_FOLDS, pd.Series(y).value_counts().min()))

    w("=" * 78)
    w(f"BASELINE LADDER — {name}")
    w("=" * 78)
    w(f"text: {corpus_dir}/corpus/{a.texts}   (text only; metadata supplies labels)")
    w(f"canonical: max_features={CANON['max_features']}, min_df={CANON['min_df']}, "
      f"min_doc_words={MIN_DOC_WORDS}, min_works={a.min_works}")
    w(f"{len(d)} works, {len(labels)} authors, {sum(len(t.split()) for t in X):,} words")
    w(f"works/author: min {pd.Series(y).value_counts().min()}, "
      f"median {int(pd.Series(y).value_counts().median())}, "
      f"max {pd.Series(y).value_counts().max()}")
    w(f"{k}-fold stratified CV, macro-F1 (+/- std across folds), chance = {chance:.3f}")
    w("")

    results, preds = [], {}
    rng = np.random.RandomState(SEED)
    cv = StratifiedKFold(k, shuffle=True, random_state=SEED)

    for label, model, why in build_ladder():
        # 'a' is in scope for --debug
        if model is None:
            p = np.asarray(pd.Series(y).sample(frac=1, random_state=SEED).tolist(),
                           dtype=object)
            f1, acc, sd = f1_score(y, p, average="macro"), accuracy_score(y, p), 0.0
        else:
            try:
                p = cross_val_predict(model, X, y, cv=cv)
            except Exception as e:
                if getattr(a, "debug", False):
                    raise
                w(f"  {label:<34} FAILED: {type(e).__name__}: {str(e)[:70]}")
                continue
            f1 = f1_score(y, p, average="macro")
            acc = accuracy_score(y, p)
            folds = []
            for tr, te in cv.split(X, y):
                folds.append(f1_score(y[te], p[te], average="macro"))
            sd = float(np.std(folds))
        preds[label] = p
        results.append((label, f1, acc, sd, why))
        w(f"  {label:<34} macro-F1={f1:.3f} (+/-{sd:.3f})  acc={acc:.3f}")
        w(f"  {'':<34} {why}")

    prelim = os.path.join(corpus_dir, "corpus", "prelim")
    os.makedirs(prelim, exist_ok=True)

    w("")
    w("-" * 78)
    ranked = sorted([r for r in results if not r[0].startswith("FLOOR")],
                    key=lambda r: -r[1])
    if not ranked:
        w("  every model failed — nothing to report.")
        w("  Re-run with --debug to see the full traceback.")
        with open(os.path.join(corpus_dir, "corpus", "prelim",
                               "baseline_ladder_v1.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        return None
    best = ranked[0]
    w(f"  best: {best[0]}  macro-F1={best[1]:.3f}")
    delta = next((r for r in results if "Delta" in r[0]), None)
    ref = next((r for r in results if "char 3-5 gram + SVM" in r[0]), None)
    sty = next((r for r in results if "stylometric" in r[0]), None)
    wrd = next((r for r in results if "word 1-2" in r[0]), None)
    if delta and ref:
        w(f"  Burrows' Delta vs char n-grams : {delta[1]:.3f} vs {ref[1]:.3f} "
          f"({delta[1]-ref[1]:+.3f})")
    if wrd and ref:
        w(f"  word 1-2 vs char units         : {wrd[1]:.3f} vs {ref[1]:.3f} "
          f"({wrd[1]-ref[1]:+.3f})")
    uni = next((r for r in results if "word unigram" in r[0]), None)
    big = next((r for r in results if "word bigram" in r[0]), None)
    if uni and big:
        w(f"  unigram vs bigram              : {uni[1]:.3f} vs {big[1]:.3f} "
          f"({big[1]-uni[1]:+.3f} for phrasing)")
        w("    unigram >> bigram  -> the signal is lexical choice")
        w("    bigram close/above -> phrasing carries real weight")
    if uni and wrd:
        w(f"  adding bigrams to unigrams     : {uni[1]:.3f} -> {wrd[1]:.3f} "
          f"({wrd[1]-uni[1]:+.3f})")
    if sty:
        w(f"  ~30 readable features alone    : {sty[1]:.3f}  "
          f"({sty[1]/chance:.0f}x chance)")
    w("-" * 78)

    prelim = os.path.join(corpus_dir, "corpus", "prelim")
    os.makedirs(prelim, exist_ok=True)

    w("")
    w(f"Per-author report — {best[0]}")
    w(classification_report(y, preds[best[0]], zero_division=0))

    # NOTE: this is the superseded v1 ladder script (see src/README.md) —
    # every output name below carries a "_v1" suffix so a re-run can't
    # clobber the current, unsuffixed output of baseline_ladder-v2.py.
    tag = re.sub(r"[^\w]+", "_", best[0]).strip("_")
    cpath = os.path.join(prelim, f"confusion_{tag}_v1.png")
    confusion_png(y, preds[best[0]], labels, cpath, f"{name} — {best[0]}")
    w(f"confusion matrix: {cpath}")

    with open(os.path.join(prelim, "baseline_ladder_v1.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(prelim, "baseline_ladder_v1.csv"), "w",
              encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "macro_f1", "accuracy", "fold_std", "hypothesis"])
        for r in results:
            wr.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", r[4]])
    print(f"\nwritten: {os.path.join(prelim, 'baseline_ladder_v1.txt')}")
    return {"name": name, "results": results, "chance": chance,
            "n": len(d), "authors": len(labels)}


def main(a):
    summaries = []
    for c in a.corpora:
        cdir = os.path.join(a.root, c)
        if not os.path.isdir(cdir):
            print(f"  skipping {c}: not found")
            continue
        s = run_corpus(c.replace("sayahna-", ""), cdir, a)
        if s:
            summaries.append(s)
        print()

    if len(summaries) < 1:
        return
    md = ["# Baseline ladder", "",
          f"Text version: `corpus/{a.texts}`. Macro-F1, stratified k-fold CV.", ""]
    head = "| Model | " + " | ".join(s["name"] for s in summaries) + " | Hypothesis |"
    md += [head, "|---" * (len(summaries) + 2) + "|"]
    names = [r[0] for r in summaries[0]["results"]]
    for i, nm in enumerate(names):
        cells, why = [], ""
        for s in summaries:
            hit = next((r for r in s["results"] if r[0] == nm), None)
            cells.append(f"{hit[1]:.3f}" if hit else "—")
            if hit:
                why = hit[4]
        md.append(f"| {nm} | " + " | ".join(cells) + f" | {why} |")
    md.append("| _chance_ | " + " | ".join(f"_{s['chance']:.3f}_" for s in summaries)
              + " | |")
    md += ["", "| Corpus | Works | Authors |", "|---|---|---|"]
    for s in summaries:
        md.append(f"| {s['name']} | {s['n']} | {s['authors']} |")

    # NOTE: this is the superseded v1 ladder script (see src/README.md) —
    # writes to the "-v1-superseded" file so a re-run can't clobber the
    # current results/malayalam/baseline-ladder.md produced by baseline_ladder-v2.py.
    out_dir = os.path.join(a.root, "results", "malayalam")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "baseline-ladder-v1-superseded.md")
    open(p, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten: {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="*",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_clean",
                    help="texts, texts_clean or texts_strict")
    ap.add_argument("--min-works", type=int, default=MIN_WORKS_PER_AUTHOR)
    ap.add_argument("--debug", action="store_true",
                    help="re-raise the first model error with a full traceback")
    main(ap.parse_args())
