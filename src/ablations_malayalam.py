#!/usr/bin/env python3
"""
Research ablations for the Malayalam AA benchmark.

Experiments:
  A. Document-length ablation:
       first N words from each work, with N in {250, 500, 1000, 2000, 4000, full}
     Uses the canonical word 1-2 gram + SVM and char 3-5 gram + SVM.

  B. Candidate-author scaling:
       randomly sample K authors while requiring enough works per author.
     Repeats the sampling and evaluates with the same work-level CV.

The purpose is NOT to hunt for the highest number. These experiments ask:
  - How much text is required before authorship becomes stable?
  - How does the closed-set problem change as the candidate pool grows?

No text from one work is ever split across train/test.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def load(root, corpus_name, texts, min_words, min_works):
    meta = pd.read_csv(
        os.path.join(root, corpus_name, "corpus", "metadata.csv"),
        dtype=str
    ).fillna("")
    rows = []
    for _, r in meta.iterrows():
        p = os.path.join(
            root, corpus_name, "corpus", texts, str(r.work_id) + ".txt"
        )
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8", errors="ignore").read().strip()
        if len(t.split()) < min_words:
            continue
        if str(r.get("work_type", "")).lower() in {"poetry", "poem", "drama"}:
            continue
        rows.append({
            "work_id": str(r.work_id),
            "author": str(r.author),
            "text": t
        })
    df = pd.DataFrame(rows)
    c = df.author.value_counts()
    return df[df.author.isin(c[c >= min_works].index)].reset_index(drop=True)


def build_models():
    common = dict(
        sublinear_tf=True,
        min_df=2,
        max_features=100_000
    )
    svc = dict(C=1.0, class_weight="balanced", max_iter=5000)

    return {
        "char_3_5_svm": make_pipeline(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                **common
            ),
            LinearSVC(**svc)
        ),
        "word_1_2_svm": make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                token_pattern=r"\S+",
                lowercase=False,
                ngram_range=(1, 2),
                **common
            ),
            LinearSVC(**svc)
        ),
    }


def evaluate(df, text_col, repeats, folds, seed):
    X = np.asarray(df[text_col].tolist(), dtype=object)
    y = np.asarray(df.author.tolist(), dtype=object)
    k = min(folds, int(df.author.value_counts().min()))
    out = []

    for name, model in build_models().items():
        scores = []
        for r in range(repeats):
            cv = StratifiedKFold(
                k, shuffle=True, random_state=seed + r
            )
            pred = cross_val_predict(model, X, y, cv=cv)
            scores.append(f1_score(y, pred, average="macro"))
        out.append({
            "model": name,
            "macro_f1_mean": np.mean(scores),
            "macro_f1_std": np.std(scores),
        })
    return out


def length_ablation(df, args, corpus):
    limits = [250, 500, 1000, 2000, 4000, None]
    rows = []

    for limit in limits:
        key = "full" if limit is None else str(limit)
        tmp = df.copy()
        tmp["eval_text"] = tmp.text.map(
            lambda x: " ".join(x.split()[:limit]) if limit else x
        )

        for r in evaluate(
            tmp, "eval_text",
            args.repeats, args.folds, args.seed
        ):
            r.update({
                "experiment": "document_length",
                "corpus": corpus,
                "length_words": key,
            })
            rows.append(r)
        print("length complete:", corpus, key)

    return rows


def author_scaling(df, args, corpus):
    rng = random.Random(args.seed)
    counts = df.author.value_counts()
    possible = sorted(counts[counts >= args.min_works].index)

    # Use a stable ladder. Do not fabricate levels larger than the corpus.
    ks = [k for k in [5, 8, 10, 12, 16, 20] if k <= len(possible)]
    rows = []

    for k in ks:
        for rep in range(args.author_repeats):
            authors = rng.sample(possible, k)
            sub = df[df.author.isin(authors)].copy().reset_index(drop=True)

            for r in evaluate(
                sub, "text",
                args.repeats, args.folds, args.seed + rep * 100
            ):
                r.update({
                    "experiment": "author_scaling",
                    "corpus": corpus,
                    "n_authors": k,
                    "author_sample_repeat": rep,
                })
                rows.append(r)

            print(
                f"author scaling: {corpus} authors={k} repeat={rep}"
            )

    return rows


def main(args):
    all_rows = []

    for corpus_name in args.corpora:
        corpus = corpus_name.replace("sayahna-", "")
        df = load(
            args.root, corpus_name, args.texts,
            args.min_words, args.min_works
        )

        print(
            f"\n{corpus}: {len(df)} works, "
            f"{df.author.nunique()} authors"
        )

        all_rows.extend(length_ablation(df, args, corpus))
        all_rows.extend(author_scaling(df, args, corpus))

    out = os.path.join(
        args.root, "results", "malayalam", "ablations"
    )
    os.makedirs(out, exist_ok=True)

    result = pd.DataFrame(all_rows)
    result.to_csv(
        os.path.join(out, "ablation_results.csv"), index=False
    )

    if not result.empty:
        result.groupby(
            ["experiment", "corpus", "model"]
        ).agg(
            mean_f1=("macro_f1_mean", "mean"),
            std_f1=("macro_f1_mean", "std")
        ).reset_index().to_csv(
            os.path.join(out, "ablation_summary.csv"),
            index=False
        )

    print("\nWrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="+",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_strict")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--min-works", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--author-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260922)
    main(ap.parse_args())
