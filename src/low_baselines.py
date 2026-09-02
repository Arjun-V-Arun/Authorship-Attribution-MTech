#!/usr/bin/env python3
"""
low_baselines.py — how much of that 0.98 is actually authorial style?

Run this in the SAME folder as corpus/metadata.csv. Writes corpus/prelim/low-baselines.txt

The standard baseline holds out whole works, so there is no chunk leakage.
But a classifier can still reach 0.98 without learning anything about style,
by exploiting three shortcuts:

    NAMED ENTITIES  an author's recurring people and places identify them
    TOPIC           an author's subject matter identifies them
    PROVENANCE      one author = one digitisation batch = one era's
                    orthography, one typesetter, one editorial pass

Each run below removes one shortcut. The score AFTER removal is the honest
estimate of how well style alone identifies the author. The DROP tells you
how much the headline number was borrowing from the shortcut.

Usage:
    pip install pandas scikit-learn numpy
    python low_baselines.py
    python low_baselines.py --quick        # fewer runs, for a fast look
    python low_baselines.py --max-authors 12
"""

import argparse
import os
import re
import sys
import glob
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

TEXTS = "corpus/texts"
PRELIM = os.path.join("corpus", "prelim")
OUTFILE = os.path.join(PRELIM, "low-baselines.txt")

SEED = 20260828
MASK = "\u25a1"                       # □ stands in for a removed token
EXCLUDE_TYPES = {"poetry", "poem", "drama"}

OUT = []


def w(s=""):
    print(s, flush=True)
    OUT.append(s)


def rule(c="-"):
    w(c * 76)


# --------------------------------------------------------------------- data

def load(max_authors=0, min_works=4):
    files = sorted(glob.glob("corpus/metadata*.csv"))
    if not files:
        sys.exit("no corpus/metadata*.csv here — run the downloader first")
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files],
                   ignore_index=True).fillna("")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)

    texts = {}
    for wid in df.work_id:
        p = os.path.join(TEXTS, str(wid) + ".txt")
        if os.path.exists(p):
            texts[wid] = open(p, encoding="utf-8").read()
    df = df[df.work_id.isin(texts)].copy()
    df["text"] = df.work_id.map(texts)

    df["wtype"] = df.work_type.str.lower().str.strip()
    df = df[~df.wtype.isin(EXCLUDE_TYPES)]
    if "section_label" in df:
        df = df[~df.section_label.str.lower().isin({"poem", "drama"})]

    vc = df.author.value_counts()
    df = df[df.author.isin(vc[vc >= min_works].index)]
    if max_authors:
        keep = df.author.value_counts().head(max_authors).index
        df = df[df.author.isin(keep)]
    return df.reset_index(drop=True)


def tokens(s):
    return s.split()


# ------------------------------------------------------------- transforms

def mask_single_author_tokens(df):
    """
    Remove every token that appears in the works of only ONE author.
    These are the giveaways: personal names, place names, an author's
    private vocabulary. What survives is shared language.
    """
    seen = defaultdict(set)
    for a, t in zip(df.author, df.text):
        for tok in set(tokens(t)):
            seen[tok].add(a)
    doomed = {tok for tok, auths in seen.items() if len(auths) == 1}
    out = [" ".join(MASK if tok in doomed else tok for tok in tokens(t))
           for t in df.text]
    return out, len(doomed), len(seen)


def mask_rare_tokens(df, min_df=5):
    """Remove tokens appearing in fewer than min_df documents."""
    dfreq = Counter()
    for t in df.text:
        dfreq.update(set(tokens(t)))
    doomed = {tok for tok, c in dfreq.items() if c < min_df}
    out = [" ".join(MASK if tok in doomed else tok for tok in tokens(t))
           for t in df.text]
    return out, len(doomed), len(dfreq)


def function_words_only(df, k=200):
    """
    Keep ONLY the k most frequent tokens in the corpus; mask everything else.
    Malayalam has no standard stopword list, so corpus frequency is the proxy
    for function words. This is the classic content-free stylometry setting.
    """
    freq = Counter()
    for t in df.text:
        freq.update(tokens(t))
    keep = {tok for tok, _ in freq.most_common(k)}
    out = [" ".join(tok if tok in keep else MASK for tok in tokens(t))
           for t in df.text]
    return out, len(keep)


def shuffle_words(df, rng):
    """Destroy word order, keep vocabulary. If the score holds up, the model
    was never using syntax or phrasing — only which words appear."""
    out = []
    for t in df.text:
        toks = tokens(t)
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def truncate(series, n):
    return series.map(lambda t: " ".join(tokens(t)[:n]))


def balance(df, n, rng):
    """Same number of works per author — otherwise the big authors carry the score."""
    parts = [g.sample(min(len(g), n), random_state=SEED) for _, g in df.groupby("author")]
    return pd.concat(parts).reset_index(drop=True)


# ------------------------------------------------------------------- model

def pipe(analyzer="char_wb", ngram=(3, 5), max_features=60_000):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                        max_features=max_features, sublinear_tf=True, min_df=2),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000))


def cv_score(X, y, name, analyzer="char_wb", ngram=(3, 5), note=""):
    y = pd.Series(list(y))
    X = pd.Series(list(X))
    k = int(min(5, y.value_counts().min()))
    if k < 2 or y.nunique() < 2:
        w(f"  {name:<44} skipped (need >=2 works for every author)")
        return None
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    pred = cross_val_predict(pipe(analyzer, ngram), X, y, cv=cv)
    f1 = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    w(f"  {name:<44} {k}-fold  macro-F1={f1:.3f}  acc={acc:.3f}  {note}")
    return f1


def chance_level(y):
    return 1.0 / pd.Series(list(y)).nunique()


# ------------------------------------------------------------ diagnostics

def top_features(df, n_per_author=12):
    """
    What is the model actually keying on? Word-level, so the features are
    readable. If you see names and topic nouns, the score is topic. If you
    see suffixes, particles and function words, it is style.
    """
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                          max_features=60_000, sublinear_tf=True, min_df=2)
    X = vec.fit_transform(df.text)
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000)
    clf.fit(X, df.author)
    names = np.array(vec.get_feature_names_out())
    w("")
    w("Top discriminative WORDS per author (word-level linear SVM)")
    w("Read these yourself. Names and topic nouns mean topic classification;")
    w("particles, suffixes and function words mean style.")
    w("")
    classes = clf.classes_
    coefs = clf.coef_ if len(classes) > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    for i, a in enumerate(classes):
        top = names[np.argsort(coefs[i])[::-1][:n_per_author]]
        w(f"  {a}")
        w(f"      {'  '.join(top)}")


def provenance_check(df):
    """Does a non-author variable line up with the author?"""
    w("")
    w("Provenance cross-check — does a non-author variable track the author?")
    for col in ["digital_edition_year", "source_pub_year", "licence_url",
                "encoded_by", "typeset_by", "publisher"]:
        if col not in df or df[col].replace("", np.nan).nunique(dropna=True) < 2:
            continue
        ct = pd.crosstab(df.author, df[col])
        o = ct.values.astype(float)
        n = o.sum()
        e = np.outer(o.sum(1), o.sum(0)) / n
        chi2 = np.nansum(np.where(e > 0, (o - e) ** 2 / e, 0.0))
        denom = n * (min(o.shape) - 1)
        v = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        flag = ("<<< tracks the author; a model can use this instead of style"
                if v >= 0.7 else "")
        w(f"  author x {col:<24} Cramer's V = {v:.2f}   {flag}")


# ---------------------------------------------------------------------- main

def main(a):
    os.makedirs(PRELIM, exist_ok=True)
    rng = random.Random(SEED)
    df = load(a.max_authors)

    w("=" * 76)
    w("LOW BASELINES — what survives when the shortcuts are removed")
    w("=" * 76)
    w("")
    w(f"corpus: {len(df)} works, {df.author.nunique()} authors, "
      f"{df.word_count.sum():,} words")
    w(f"works per author: min {df.author.value_counts().min()}, "
      f"median {int(df.author.value_counts().median())}, "
      f"max {df.author.value_counts().max()}")
    w(f"chance level (macro-F1 of random guessing) ≈ {chance_level(df.author):.3f}")
    w("")
    w("Whole works are held out in every run, so there is no chunk leakage.")
    w("The question here is different: WHICH SIGNAL is the model using?")
    w("")

    # ---------------------------------------------------------- 0. sanity
    rule("=")
    w("0. SANITY CHECKS")
    rule("=")
    perm = []
    for i in range(1 if a.quick else 3):
        y = list(df.author)
        rng.shuffle(y)
        s = cv_score(df.text, y, f"labels shuffled (run {i+1})")
        if s is not None:
            perm.append(s)
    if perm:
        ch = chance_level(df.author)
        w(f"  -> permutation mean macro-F1 = {np.mean(perm):.3f}   chance = {ch:.3f}")
        w("     Should be near chance. Somewhat above is normal when documents form")
        w("     strong clusters. Far above (say >2x chance) means the pipeline leaks.")

    # ---------------------------------------------------- 1. the reference
    w("")
    rule("=")
    w("1. REFERENCE — what your current baselines.txt reports")
    rule("=")
    ref = cv_score(df.text, df.author, "char 3-5 gram, full text")
    bal = balance(df, a.cap, rng)
    ref_bal = cv_score(bal.text, bal.author,
                       f"char 3-5 gram, capped at {a.cap} works/author",
                       note="removes the big-author advantage")

    # -------------------------------------------------- 2. remove the props
    w("")
    rule("=")
    w("2. REMOVE NAMED ENTITIES AND PRIVATE VOCABULARY")
    rule("=")
    masked, n_doomed, n_total = mask_single_author_tokens(df)
    w(f"  masked {n_doomed:,} of {n_total:,} token types "
      f"({n_doomed/max(1,n_total):.0%}) that only ever appear in one author")
    m1 = cv_score(masked, df.author, "single-author tokens masked")
    if ref and m1:
        w(f"  -> DROP {ref:.3f} -> {m1:.3f}  ({m1-ref:+.3f})")
        w("     A large drop means the score was mostly names and private topics.")

    rare, n_rare, _ = mask_rare_tokens(df, min_df=a.min_df)
    w("")
    w(f"  masked {n_rare:,} token types occurring in fewer than {a.min_df} documents")
    m2 = cv_score(rare, df.author, f"rare tokens (df<{a.min_df}) masked")

    # ------------------------------------------------ 3. content-free style
    w("")
    rule("=")
    w("3. CONTENT-FREE STYLOMETRY — only the commonest words survive")
    rule("=")
    w("  Malayalam has no standard stopword list, so corpus frequency stands in.")
    for k in ([200] if a.quick else [100, 300, 1000]):
        fw, nkeep = function_words_only(df, k)
        s = cv_score(fw, df.author, f"top-{k} tokens only (rest masked)",
                     note=f"vocab={nkeep}")
        if s and ref and k == 300:
            w(f"  -> {s:.3f} against {ref:.3f} on full text. If this stays high,")
            w("     the signal is genuinely stylistic rather than topical.")

    # ---------------------------------------------------- 4. word order
    w("")
    rule("=")
    w("4. DESTROY WORD ORDER — keep vocabulary, lose syntax and phrasing")
    rule("=")
    shuf = shuffle_words(df, rng)
    s4 = cv_score(shuf, df.author, "words shuffled within each work")
    if ref and s4:
        w(f"  -> {ref:.3f} -> {s4:.3f}  ({s4-ref:+.3f})")
        w("     Little change means the model never used word order or phrasing —")
        w("     it is a bag-of-words topic model wearing a stylometry costume.")

    # ------------------------------------------------------- 5. text length
    w("")
    rule("=")
    w("5. HOW MUCH TEXT DOES IT ACTUALLY NEED?")
    rule("=")
    for n in ([250, 1000] if a.quick else [100, 250, 500, 1000, 2000]):
        cv_score(truncate(df.text, n), df.author, f"first {n} words only")

    # -------------------------------------------- 6. the hard, honest setting
    w("")
    rule("=")
    w("6. COMBINED — balanced, entity-masked, length-equalised")
    rule("=")
    w("  This is the closest thing here to an honest difficulty estimate.")
    bal2 = balance(df, a.cap, rng)
    bmask, _, _ = mask_single_author_tokens(bal2)
    bmask = [" ".join(tokens(t)[:500]) for t in bmask]
    hard = cv_score(bmask, bal2.author,
                    f"capped {a.cap}/author + masked + first 500 words")
    if ref and hard:
        w("")
        w(f"  HEADLINE  {ref:.3f}")
        w(f"  HONEST    {hard:.3f}")
        w(f"  GAP       {hard-ref:+.3f}")
        w(f"  chance    {chance_level(bal2.author):.3f}")

    # -------------------------------------------------------- 7. diagnostics
    w("")
    rule("=")
    w("7. WHAT IS THE MODEL LOOKING AT?")
    rule("=")
    provenance_check(df)
    top_features(df, 10 if a.quick else 12)

    w("")
    rule("=")
    w("HOW TO READ THIS")
    rule("=")
    w("  Sections 2 and 4 are the important ones.")
    w("  If masking single-author tokens barely moves the score, and shuffling")
    w("  word order barely moves it either, then the model is doing topic")
    w("  classification and the honest number is the one in section 6.")
    w("  If the score survives both, you have real stylistic signal and that")
    w("  is a result worth writing up.")
    w("")
    w("  Report section 1 AND section 6 in any paper. Reporting only section 1")
    w("  is what the existing Indic AA literature does, and it is why its")
    w("  numbers are not comparable to anything.")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\nwritten: {OUTFILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer runs")
    ap.add_argument("--cap", type=int, default=8, help="works per author when balancing")
    ap.add_argument("--min-df", type=int, default=5, help="rare-token threshold")
    ap.add_argument("--max-authors", type=int, default=0, help="keep only the N largest authors")
    main(ap.parse_args())
