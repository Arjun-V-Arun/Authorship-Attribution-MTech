#!/usr/bin/env python3
"""
morphology_stats.py — why do word features beat character features here,
and why does Burrows' Delta fail?

A single hypothesis explains both results:

    In an agglutinative language each lemma fragments across many inflected
    forms. So (a) the top-k word FORMS cover much less running text than in
    an analytic language, which is exactly the assumption Burrows' Delta
    rests on; and (b) any individual word form carries more morphological
    information, which is why word-level features are strong.

This script measures that directly. It computes nothing model-related — only
corpus statistics — so the numbers are stable and cheap.

    coverage(k)   share of running tokens falling in the k most frequent types
                  THE number for Delta: coverage(300)
    Heaps' beta   vocabulary growth exponent V = K * N^beta.
                  Higher beta = vocabulary keeps growing = more inflection.
    hapax rate    share of types seen exactly once
    word length   in GRAPHEME CLUSTERS, not code points (Malayalam differs)

Run it on your Malayalam corpora AND on your Hindi corpus. Same pipeline,
same code, so the comparison needs no citation and no external reference
number -- Dravidian agglutinative against Indo-Aryan fusional, measured
identically.

Usage:
    python morphology_stats.py --root . \\
        --corpora sayahna-fiction sayahna-essays --texts texts_strict

    # add Hindi: point at any folder holding corpus/metadata.csv + corpus/<texts>
    python morphology_stats.py --root . \\
        --corpora sayahna-fiction sayahna-essays hindi-corpus --texts texts

Output, at --root:
    morphology_stats.md         the table
    coverage_curve.png          coverage(k) for every corpus on one axes
    vocab_growth.png            Heaps' law, log-log
"""

import argparse
import math
import os
import re
import sys
import glob
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import regex as _re2
    HAVE_REGEX = True
except ImportError:
    HAVE_REGEX = False

KS = [10, 50, 100, 300, 1000, 3000, 10000]
DROP_TYPES = {"poetry", "poem", "drama"}


def graphemes(s):
    return _re2.findall(r"\X", s) if HAVE_REGEX else list(s)


def load_texts(corpus_dir, textdir):
    meta = os.path.join(corpus_dir, "corpus", "metadata.csv")
    tdir = os.path.join(corpus_dir, "corpus", textdir)
    if not os.path.isdir(tdir):
        alt = os.path.join(corpus_dir, "corpus", "texts")
        if os.path.isdir(alt):
            print(f"  ({textdir} missing, using texts/)")
            tdir = alt
        else:
            return None, None, f"no corpus/{textdir}"
    authors = {}
    if os.path.exists(meta):
        df = pd.read_csv(meta, dtype=str).fillna("")
        idcol = "work_id" if "work_id" in df else df.columns[0]
        for _, r in df.iterrows():
            wt = str(r.get("work_type", "")).lower()
            if wt in DROP_TYPES:
                continue
            authors[str(r[idcol])] = str(r.get("author", ""))
    texts, auth = [], []
    for p in sorted(glob.glob(os.path.join(tdir, "*.txt"))):
        wid = os.path.splitext(os.path.basename(p))[0]
        if authors and wid not in authors:
            continue
        t = open(p, encoding="utf-8").read()
        if len(t.split()) < 100:
            continue
        texts.append(t)
        auth.append(authors.get(wid, ""))
    if not texts:
        return None, None, "no usable texts"
    return texts, auth, None


def coverage_curve(counter, total, ks=KS):
    """Share of running tokens covered by the k most frequent types."""
    ordered = [c for _, c in counter.most_common()]
    cum = np.cumsum(ordered)
    out = {}
    for k in ks:
        out[k] = float(cum[min(k, len(cum)) - 1] / total) if len(cum) else 0.0
    return out, cum


def heaps_beta(tokens, points=25):
    """
    Fit V = K * N^beta on log-log. beta near 0.5 is unremarkable; higher means
    the vocabulary keeps expanding with corpus size, which is what heavy
    inflection produces.
    """
    n = len(tokens)
    if n < 2000:
        return float("nan"), [], []
    idx = np.unique(np.logspace(math.log10(500), math.log10(n), points).astype(int))
    seen, V, N = set(), [], []
    prev = 0
    for cut in idx:
        for t in tokens[prev:cut]:
            seen.add(t)
        prev = cut
        V.append(len(seen))
        N.append(cut)
    b = np.polyfit(np.log(N), np.log(V), 1)[0]
    return float(b), N, V


def analyse(name, texts):
    tokens = []
    for t in texts:
        tokens += t.split()
    total = len(tokens)
    cnt = Counter(tokens)
    cov, _ = coverage_curve(cnt, total)
    beta, N, V = heaps_beta(tokens)
    wl = [len(graphemes(w)) for w in tokens[:200_000]]
    hapax = sum(1 for v in cnt.values() if v == 1)
    return {
        "name": name, "tokens": total, "types": len(cnt),
        "ttr": len(cnt) / total,
        "hapax_share_of_types": hapax / len(cnt),
        "coverage": cov, "beta": beta, "N": N, "V": V,
        "mean_word_graphemes": float(np.mean(wl)),
        "median_word_graphemes": float(np.median(wl)),
    }


def main(a):
    stats = []
    for c in a.corpora:
        cdir = os.path.join(a.root, c)
        if not os.path.isdir(cdir):
            print(f"  skipping {c}: not found")
            continue
        print(f"reading {c} ...")
        texts, _, err = load_texts(cdir, a.texts)
        if texts is None:
            print(f"  skipping {c}: {err}")
            continue
        s = analyse(c.replace("sayahna-", ""), texts)
        stats.append(s)
        print(f"  {s['tokens']:,} tokens, {s['types']:,} types, "
              f"coverage(300)={s['coverage'][300]:.1%}, beta={s['beta']:.3f}")

    if not stats:
        sys.exit("nothing to analyse")

    md = ["# Vocabulary and morphology statistics", "",
          f"Text version: `corpus/{a.texts}`. Corpus statistics only — no models.", ""]

    md += ["## Coverage: share of running tokens in the k most frequent word types", "",
           "| k | " + " | ".join(s["name"] for s in stats) + " |",
           "|---" * (len(stats) + 1) + "|"]
    for k in KS:
        md.append(f"| top {k:,} | " +
                  " | ".join(f"{s['coverage'][k]:.1%}" for s in stats) + " |")
    md += ["",
           "**`top 300` is the number that matters for Burrows' Delta**, which "
           "assumes the most frequent word forms cover most of a text. In an "
           "analytic language they largely do. Under heavy inflection each lemma "
           "splits across many forms, so the same 300 slots reach far less text.",
           ""]

    md += ["## Vocabulary growth and word shape", "",
           "| Statistic | " + " | ".join(s["name"] for s in stats) + " |",
           "|---" * (len(stats) + 1) + "|"]
    rows = [
        ("Tokens", lambda s: f"{s['tokens']:,}"),
        ("Types", lambda s: f"{s['types']:,}"),
        ("Type-token ratio", lambda s: f"{s['ttr']:.4f}"),
        ("Hapax share of types", lambda s: f"{s['hapax_share_of_types']:.1%}"),
        ("Heaps' beta", lambda s: f"{s['beta']:.3f}"),
        ("Mean word length (graphemes)", lambda s: f"{s['mean_word_graphemes']:.2f}"),
        ("Median word length (graphemes)", lambda s: f"{s['median_word_graphemes']:.0f}"),
    ]
    for label, fn in rows:
        md.append(f"| {label} | " + " | ".join(fn(s) for s in stats) + " |")

    md += ["", "## How to read this", "",
           "A low `top 300` coverage together with a high Heaps' beta and a high "
           "hapax share is the signature of heavy inflection: the vocabulary keeps "
           "growing because word forms, not lemmas, are being counted.",
           "",
           "That single fact predicts both results in the baseline ladder:",
           "",
           "- **Delta underperforms.** Its 300 most-frequent-word features reach "
           "only a small share of the text, so it discards most of the evidence.",
           "- **Word features beat character features.** Each inflected form "
           "carries morphological information that character n-grams have to "
           "reconstruct piecemeal across the same span.",
           "",
           "Run this on a fusional Indo-Aryan corpus (your Hindi set) with the "
           "same command. If Hindi shows higher top-300 coverage and a lower "
           "beta, the explanation holds across language families and the claim "
           "stops being about Malayalam alone.",
           ""]

    # ---- coverage curve
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    for s in stats:
        ax.plot(KS, [s["coverage"][k] * 100 for k in KS], marker="o", label=s["name"])
    ax.axvline(300, color="#888888", ls="--", lw=1)
    ax.annotate("Burrows' Delta\nuses 300", (300, 8), fontsize=9, color="#555555",
                ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("k  (most frequent word types)")
    ax.set_ylabel("% of running tokens covered")
    ax.set_title("Vocabulary coverage", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    os.makedirs(a.output, exist_ok=True)

    p1 = os.path.join(a.output, "coverage_curve.png")
    fig.savefig(p1, dpi=150, facecolor="white")
    plt.close(fig)

    # ---- Heaps' law
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    for s in stats:
        if s["N"]:
            ax.plot(s["N"], s["V"], marker=".",
                    label=f"{s['name']}  (beta={s['beta']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("distinct word types")
    ax.set_title("Vocabulary growth (Heaps' law)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    p2 = os.path.join(a.output, "vocab_growth.png")
    fig.savefig(p2, dpi=150, facecolor="white")
    plt.close(fig)

    out = os.path.join(a.output, "morphology_stats.md")
    open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"written: {out}\n         {p1}\n         {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="*",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_strict")
    ap.add_argument("--output", default=None,
                    help="folder for output files (default: --root)")

    args = ap.parse_args()
    if args.output is None:
        args.output = args.root

    main(args)
