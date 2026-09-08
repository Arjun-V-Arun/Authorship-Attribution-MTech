#!/usr/bin/env python3
"""
merge_and_crossgenre.py — the cross-genre experiment.

Five authors appear in BOTH the fiction and the essay corpus. That makes a
controlled experiment possible: train on an author's fiction, test on the same
author's non-fiction, and compare against within-genre performance on exactly
the same authors. No Indic authorship study has reported this.

    1. merge both corpora, deduplicating by work_id and by content hash
    2. unify author names across them ("Zacharia" / "Paul Zacharia")
    3. label each work fiction or non-fiction from its TEI work_type
    4. report which authors have enough of both
    5. run: within-fiction, within-nonfiction, fiction->nonfiction,
       nonfiction->fiction; each raw and entity-masked

Usage:
    python merge_and_crossgenre.py --root C:\\...\\Authorship-Attribution-MTech
    python merge_and_crossgenre.py --texts texts_strict --min-per-genre 4

Writes:
    sayahna-crossgenre/metadata.csv     the combined corpus
    sayahna-crossgenre/texts/           one file per work
    sayahna-crossgenre/crossgenre.txt   the results
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

FICTION_TYPES = {"novel", "novella", "short_story", "story", "fiction"}
NONFICTION_TYPES = {"essay", "article", "criticism", "memoir", "biography",
                    "speech", "interview", "travelogue", "letters",
                    "thoolikachithram"}
DROP_TYPES = {"poetry", "poem", "drama"}

# author strings that denote the same person across the two corpora
ALIASES = {
    "zacharia": "Paul Zacharia",
    "paul zacharia": "Paul Zacharia",
}

MASK = "\u25a1"
SEED = 20260831
OUT = []


def w(s=""):
    print(s, flush=True)
    OUT.append(s)


def canon(name):
    k = re.sub(r"[^a-z ]", "", str(name).lower()).strip()
    if k in ALIASES:
        return ALIASES[k]
    return str(name).strip()


def genre_of(row):
    t = str(row.get("work_type", "")).lower().strip()
    if t in DROP_TYPES:
        return "drop"
    if t in FICTION_TYPES:
        return "fiction"
    if t in NONFICTION_TYPES:
        return "nonfiction"
    lab = str(row.get("section_label", "")).lower().strip()
    if lab in {"poem", "drama"}:
        return "drop"
    if lab == "story":
        return "fiction"
    if lab == "essay":
        return "nonfiction"
    return "unknown"


def load_one(folder, textdir):
    meta = os.path.join(folder, "corpus", "metadata.csv")
    tdir = os.path.join(folder, "corpus", textdir)
    if not os.path.exists(meta):
        w(f"  !! no metadata.csv in {folder}")
        return None
    if not os.path.isdir(tdir):
        w(f"  !! {tdir} not found — falling back to corpus/texts")
        tdir = os.path.join(folder, "corpus", "texts")
    df = pd.read_csv(meta, dtype=str).fillna("")
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(tdir, str(r.work_id) + ".txt")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        if len(txt.split()) < 100:
            continue
        d = r.to_dict()
        d["text"] = txt
        d["source_corpus"] = os.path.basename(folder.rstrip("/\\"))
        rows.append(d)
    out = pd.DataFrame(rows)
    w(f"  {folder}: {len(out)} works from corpus/{textdir}")
    return out


def mask_author_unique(texts, authors):
    seen = defaultdict(set)
    for a, t in zip(authors, texts):
        for tok in set(t.split()):
            seen[tok].add(a)
    doomed = {tok for tok, s in seen.items() if len(s) == 1}
    return [" ".join(MASK if tok in doomed else tok for tok in t.split())
            for t in texts]


def run(tr_X, tr_y, te_X, te_y, name):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score

    common = sorted(set(tr_y) & set(te_y))
    keep_tr = [i for i, y in enumerate(tr_y) if y in common]
    keep_te = [i for i, y in enumerate(te_y) if y in common]
    if len(common) < 2 or len(keep_te) < 5:
        w(f"  {name:<40} skipped (too few shared authors)")
        return None
    Xtr = [tr_X[i] for i in keep_tr]; ytr = [tr_y[i] for i in keep_tr]
    Xte = [te_X[i] for i in keep_te]; yte = [te_y[i] for i in keep_te]
    m = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        max_features=60_000, sublinear_tf=True, min_df=2),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000))
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    f1 = f1_score(yte, p, average="macro")
    acc = accuracy_score(yte, p)
    w(f"  {name:<40} authors={len(common)}  train={len(ytr):<4} test={len(yte):<4} "
      f"macro-F1={f1:.3f}  acc={acc:.3f}")
    return f1


def cv(X, y, name):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score

    y = pd.Series(y)
    k = int(min(5, y.value_counts().min()))
    if k < 2:
        w(f"  {name:<40} skipped")
        return None
    m = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        max_features=60_000, sublinear_tf=True, min_df=2),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000))
    pred = cross_val_predict(m, pd.Series(X), y,
                             cv=StratifiedKFold(k, shuffle=True, random_state=SEED))
    f1 = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    w(f"  {name:<40} authors={y.nunique()}  works={len(y):<4} {k}-fold      "
      f"macro-F1={f1:.3f}  acc={acc:.3f}")
    return f1


def main(a):
    fic = os.path.join(a.root, a.fiction)
    ess = os.path.join(a.root, a.essays)

    w("=" * 78)
    w("MERGE AND CROSS-GENRE ATTRIBUTION")
    w("=" * 78)
    w(f"text version: corpus/{a.texts}")
    w("")

    parts = [p for p in (load_one(fic, a.texts), load_one(ess, a.texts)) if p is not None]
    if not parts:
        sys.exit("nothing loaded — check --root, --fiction, --essays")
    df = pd.concat(parts, ignore_index=True).fillna("")

    # ---- unify authors, dedupe, label genre
    df["author"] = df.author.map(canon)
    before = len(df)
    df = df.drop_duplicates(subset="work_id", keep="first")
    by_id = before - len(df)
    if "sha256" in df:
        n = len(df)
        df = df.drop_duplicates(subset="sha256", keep="first")
        by_hash = n - len(df)
    else:
        by_hash = 0
    df["genre"] = df.apply(genre_of, axis=1)

    w("")
    w(f"merged: {before} works -> {len(df)} after dedup "
      f"({by_id} same work_id, {by_hash} same content)")
    w(f"genres: " + ", ".join(f"{k}={v}" for k, v in df.genre.value_counts().items()))

    df = df[df.genre.isin(["fiction", "nonfiction"])].reset_index(drop=True)

    # ---- who has both?
    w("")
    w("-" * 78)
    w("AUTHORS BY GENRE")
    w("-" * 78)
    ct = pd.crosstab(df.author, df.genre)
    for g in ("fiction", "nonfiction"):
        if g not in ct:
            ct[g] = 0
    ct = ct[["fiction", "nonfiction"]].sort_values(
        ["nonfiction", "fiction"], ascending=False)
    w(ct.to_string())

    both = ct[(ct.fiction >= a.min_per_genre) & (ct.nonfiction >= a.min_per_genre)]
    w("")
    w(f"authors with >= {a.min_per_genre} works in BOTH genres: {len(both)}")
    if len(both):
        w("  " + ", ".join(both.index))
    if len(both) < 2:
        w("")
        w("  Not enough for the cross-genre experiment. Lower --min-per-genre,")
        w("  or check that author names match across the two corpora.")

    # ---- write the merged corpus
    mdir = os.path.join(a.root, "sayahna-crossgenre")
    tdir = os.path.join(mdir, "texts")
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
    os.makedirs(tdir, exist_ok=True)
    for _, r in df.iterrows():
        open(os.path.join(tdir, f"{r.work_id}.txt"), "w", encoding="utf-8").write(r.text)
    df.drop(columns=["text"]).to_csv(os.path.join(mdir, "metadata.csv"), index=False)
    w("")
    w(f"written: {os.path.join(mdir, 'metadata.csv')}  ({len(df)} works)")

    # ---- the experiment
    if len(both) >= 2:
        sub = df[df.author.isin(both.index)].reset_index(drop=True)
        F = sub[sub.genre == "fiction"]
        N = sub[sub.genre == "nonfiction"]

        w("")
        w("=" * 78)
        w("CROSS-GENRE EXPERIMENT")
        w("=" * 78)
        w(f"{len(both)} authors, {len(F)} fiction works, {len(N)} non-fiction works")
        w(f"chance macro-F1 for {len(both)} authors = {1/len(both):.3f}")
        w("")
        w("Within-genre rows are the CONTROL: same authors, same models, so any")
        w("gap to the cross-genre rows is the cost of changing genre alone.")
        w("")

        results = {}
        w("  -- raw text")
        results["within_f"] = cv(list(F.text), list(F.author), "within fiction (CV)")
        results["within_n"] = cv(list(N.text), list(N.author), "within non-fiction (CV)")
        results["f2n"] = run(list(F.text), list(F.author),
                             list(N.text), list(N.author), "fiction -> non-fiction")
        results["n2f"] = run(list(N.text), list(N.author),
                             list(F.text), list(F.author), "non-fiction -> fiction")

        w("")
        w("  -- author-unique tokens masked (removes names and private topics)")
        allm = mask_author_unique(list(sub.text), list(sub.author))
        sub2 = sub.copy()
        sub2["mtext"] = allm
        F2 = sub2[sub2.genre == "fiction"]
        N2 = sub2[sub2.genre == "nonfiction"]
        results["m_within_f"] = cv(list(F2.mtext), list(F2.author), "within fiction (CV)")
        results["m_within_n"] = cv(list(N2.mtext), list(N2.author), "within non-fiction (CV)")
        results["m_f2n"] = run(list(F2.mtext), list(F2.author),
                               list(N2.mtext), list(N2.author), "fiction -> non-fiction")
        results["m_n2f"] = run(list(N2.mtext), list(N2.author),
                               list(F2.mtext), list(F2.author), "non-fiction -> fiction")

        w("")
        w("  -- first 500 words only (equalises document length)")
        cut = lambda s: [" ".join(t.split()[:500]) for t in s]
        results["t_f2n"] = run(cut(list(F.text)), list(F.author),
                               cut(list(N.text)), list(N.author), "fiction -> non-fiction")
        results["t_n2f"] = run(cut(list(N.text)), list(N.author),
                               cut(list(F.text)), list(F.author), "non-fiction -> fiction")

        w("")
        w("-" * 78)
        wf, f2n = results.get("within_f"), results.get("f2n")
        wn, n2f = results.get("within_n"), results.get("n2f")
        if wf and f2n:
            w(f"  COST OF GENRE SHIFT (fiction -> non-fiction): "
              f"{wf:.3f} -> {f2n:.3f}  ({f2n-wf:+.3f})")
        if wn and n2f:
            w(f"  COST OF GENRE SHIFT (non-fiction -> fiction): "
              f"{wn:.3f} -> {n2f:.3f}  ({n2f-wn:+.3f})")
        mf2n = results.get("m_f2n")
        if f2n and mf2n:
            w(f"  cross-genre, entity-masked                  : "
              f"{f2n:.3f} -> {mf2n:.3f}  ({mf2n-f2n:+.3f})")
            w("")
            w("  If the masked cross-genre score stays well above chance, what")
            w("  survives a change of genre AND the removal of private vocabulary")
            w("  is authorial style in the strict sense. That is the claim to make.")
        w("-" * 78)

    outp = os.path.join(a.root, "sayahna-crossgenre", "crossgenre.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fiction", default="sayahna-fiction")
    ap.add_argument("--essays", default="sayahna-essays")
    ap.add_argument("--texts", default="texts_strict",
                    help="texts, texts_clean or texts_strict")
    ap.add_argument("--min-per-genre", type=int, default=4,
                    help="works needed in each genre for an author to be included")
    main(ap.parse_args())
