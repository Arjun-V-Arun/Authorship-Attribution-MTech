#!/usr/bin/env python3
"""
corpus_overview.py — figures and baselines for the downloaded corpus.

Usage:
    pip install pandas matplotlib scikit-learn
    python corpus_overview.py

Writes everything into corpus/prelim/ :
    distribution_pies.png     documents by author, and by work type
    work_lengths.png          every work's length, grouped by author,
                              coloured by work type (log scale)
    work_lengths_linear.png   the same on a linear axis
    baselines.txt             baseline results, saved as text
    corpus_summary.csv        per-author summary table

Charts use the English author and title fields, so no Malayalam font is needed.
"""

import os
import sys
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

META = "corpus/metadata.csv"
TEXTS = "corpus/texts"
PRELIM = os.path.join("corpus", "prelim")

SEED = 20260827
TRUNC_WORDS = 700
MIN_DOCS_FOR_CV = 3

# colour per work type — colourblind-safe, consistent across every figure
TYPE_COLOURS = {
    "short_story": "#4C72B0", "story": "#4C72B0", "novel": "#DD8452",
    "essay": "#55A868", "poem": "#C44E52", "poetry": "#C44E52",
    "drama": "#8172B3", "article": "#937860", "memoir": "#DA8BC3",
    "speech": "#8C8C8C", "thoolikachithram": "#CCB974",
}
FALLBACK = "#64B5CD"


def colour(t):
    return TYPE_COLOURS.get(str(t).strip().lower(), FALLBACK)


def load():
    if not os.path.exists(META):
        sys.exit(f"{META} not found — run download_corpus.py first")
    df = pd.read_csv(META, dtype=str).fillna("")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)

    texts = {}
    for wid in df.work_id:
        p = os.path.join(TEXTS, wid + ".txt")
        if os.path.exists(p):
            texts[wid] = open(p, encoding="utf-8").read()
    df = df[df.work_id.isin(texts)].copy()
    df["text"] = df.work_id.map(texts)

    df["wtype"] = df.work_type.replace("", np.nan).fillna(
        df.section_label.str.lower()).str.lower().str.strip()
    df["label"] = np.where(df.title_en.str.strip() != "",
                           df.title_en.str.strip(), df.work_id)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------- figure 1
def pies(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.patch.set_facecolor("white")

    # -- by author, weighted by number of works
    a = df.author.value_counts()
    cmap = plt.get_cmap("tab20")
    acols = [cmap(i % 20) for i in range(len(a))]
    wedges, _, autotexts = axes[0].pie(
        a.values, colors=acols, startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 3.5 else "",
        pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.6),
        textprops=dict(fontsize=9.5, color="white", fontweight="bold"))
    axes[0].set_title("Documents by author", fontsize=15, fontweight="bold", pad=18)
    axes[0].legend(wedges, [f"{n}  ({v})" for n, v in zip(a.index, a.values)],
                   loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=9.5)
    axes[0].text(0, 0, f"{len(df)}\nworks", ha="center", va="center",
                 fontsize=15, fontweight="bold", color="#333333")

    # -- by work type
    t = df.wtype.value_counts()
    tcols = [colour(x) for x in t.index]
    wedges2, _, _ = axes[1].pie(
        t.values, colors=tcols, startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 3.5 else "",
        pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.6),
        textprops=dict(fontsize=9.5, color="white", fontweight="bold"))
    axes[1].set_title("Documents by work type", fontsize=15, fontweight="bold", pad=18)
    axes[1].legend(wedges2, [f"{n}  ({v})" for n, v in zip(t.index, t.values)],
                   loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=9.5)
    axes[1].text(0, 0, f"{df.wtype.nunique()}\ntypes", ha="center", va="center",
                 fontsize=15, fontweight="bold", color="#333333")

    fig.suptitle("Malayalam authorship corpus — composition",
                 fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             f"{len(df)} works · {df.author.nunique()} authors · "
             f"{df.word_count.sum():,} words · source: Sayahna Foundation",
             ha="center", fontsize=10, color="#666666")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


# ------------------------------------------------------------------- figure 2
def lengths(df, path, log=True):
    d = df.sort_values(["author", "word_count"], ascending=[True, False]).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(max(18, n * 0.21), 11.5))
    fig.patch.set_facecolor("white")

    x = np.arange(n)
    bars = ax.bar(x, d.word_count, color=[colour(t) for t in d.wtype],
                  edgecolor="white", linewidth=0.5, width=0.82, zorder=3)

    # alternating bands, with author names staggered over two rows so that a
    # long name above a narrow band cannot be misread as belonging to its
    # neighbour, plus a bracket marking each band's true extent
    start = 0
    for i, (author, grp) in enumerate(d.groupby("author", sort=False)):
        end = start + len(grp)
        if i % 2 == 0:
            ax.axvspan(start - 0.5, end - 0.5, color="#000000", alpha=0.055, zorder=0)
        row = i % 2                      # 0 = lower label row, 1 = upper
        ybar = 1.015 + row * 0.055
        ytxt = ybar + 0.012
        # bracket spanning exactly this author's bars
        ax.annotate("", xy=(start - 0.45, ybar), xytext=(end - 0.55, ybar),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=1.4),
                    annotation_clip=False)
        ax.annotate(f"{textwrap.fill(author, 22)}  ({len(grp)})",
                    xy=((start + end - 1) / 2, ytxt),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#222222", annotation_clip=False)
        if start:
            ax.axvline(start - 0.5, color="#AAAAAA", lw=0.9, zorder=1)
        start = end

    ax.set_xticks(x)
    ax.set_xticklabels(d.label, rotation=90, fontsize=6.2)
    ax.set_xlim(-0.8, n - 0.2)
    if log:
        ax.set_yscale("log")
        ax.set_ylabel("Word count  (log scale)", fontsize=12)
    else:
        ax.set_ylabel("Word count", fontsize=12)
    ax.set_xlabel("Work", fontsize=12, labelpad=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # label the longest work in each author block so the eye has anchors
    for _, grp in d.groupby("author", sort=False):
        i = grp.word_count.idxmax()
        if d.loc[i, "word_count"] >= d.word_count.median() * 1.8:
            ax.annotate(f"{d.loc[i,'word_count']:,}", (i, d.loc[i, "word_count"]),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#333333", fontweight="bold")

    present = sorted(d.wtype.unique(), key=lambda t: -int((d.wtype == t).sum()))
    ax.legend(handles=[Patch(facecolor=colour(t),
                             label=f"{t}  ({int((d.wtype == t).sum())})")
                       for t in present],
              loc="lower center", bbox_to_anchor=(0.5, 1.135), ncol=len(present),
              frameon=False, fontsize=10.5)

    ax.set_title("Work length by author and work type"
                 + ("  (log scale — short works stay legible next to novels)" if log else ""),
                 fontsize=16, fontweight="bold", y=1.20)
    fig.text(0.5, 0.005,
             f"median {int(d.word_count.median()):,} words · "
             f"range {d.word_count.min():,}–{d.word_count.max():,}",
             ha="center", fontsize=9.5, color="#666666")
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.045, right=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


# ------------------------------------------------------------------ baselines
def baselines(df, path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    out = []

    def w(s=""):
        print(s)
        out.append(s)

    def run(sub, name, truncate=None):
        vc = sub.author.value_counts()
        sub = sub[sub.author.isin(vc[vc >= MIN_DOCS_FOR_CV].index)]
        if sub.author.nunique() < 2 or len(sub) < 10:
            w(f"  {name:<34} skipped (too few works)")
            return None
        k = int(min(5, sub.author.value_counts().min()))
        X = sub.text
        if truncate:
            X = X.map(lambda t: " ".join(t.split()[:truncate]))
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            max_features=100_000, sublinear_tf=True, min_df=2),
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        pred = cross_val_predict(pipe, X, sub.author, cv=cv)
        f1 = f1_score(sub.author, pred, average="macro")
        acc = accuracy_score(sub.author, pred)
        w(f"  {name:<34} authors={sub.author.nunique():<3} works={len(sub):<4} "
          f"{k}-fold CV   macro-F1={f1:.3f}  acc={acc:.3f}")
        return f1, sub, pred

    w("=" * 74)
    w("BASELINES — character 3-5 gram TF-IDF + linear SVM")
    w("=" * 74)
    w()
    w("One work = one document, so no chunk leakage is possible by construction.")
    w("Stratified k-fold cross-validation, not a single split, because the corpus")
    w("is small and a single split would be noisy.")
    w()

    # sanity floor
    vc = df.author.value_counts()
    d0 = df[df.author.isin(vc[vc >= MIN_DOCS_FOR_CV].index)]
    k = int(min(5, d0.author.value_counts().min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    pl = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                           d0[["word_count"]], d0.author, cv=cv)
    f1_len = f1_score(d0.author, pl, average="macro")
    w(f"  {'(a) length only — sanity floor':<34} macro-F1={f1_len:.3f}   "
      + ("<<< length is a shortcut; trust (d)" if f1_len > 0.30 else "length is not a shortcut"))

    r_all = run(df, "(b) all works, all types")

    prose = df[~df.wtype.isin(["poem", "poetry", "drama"])]
    r_prose = run(prose, "(c) prose only (no poem/drama)")

    r_trunc = run(prose, f"(d) prose, first {TRUNC_WORDS} words only",
                  truncate=TRUNC_WORDS)

    # cross-genre: train on stories, test on essays, same authors
    w()
    st = df[df.section_label.str.lower() == "story"]
    es = df[df.section_label.str.lower() == "essay"]
    both = sorted(set(st.author) & set(es.author))
    if len(both) >= 2 and len(es[es.author.isin(both)]) >= 8:
        from sklearn.pipeline import make_pipeline as mp
        tr, te = st[st.author.isin(both)], es[es.author.isin(both)]
        pipe = mp(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  max_features=100_000, sublinear_tf=True, min_df=2),
                  LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
        pipe.fit(tr.text, tr.author)
        p = pipe.predict(te.text)
        w(f"  {'(e) train story -> test essay':<34} authors={len(both):<3} "
          f"train={len(tr):<4} test={len(te):<4} "
          f"macro-F1={f1_score(te.author, p, average='macro'):.3f}")
    else:
        w(f"  {'(e) train story -> test essay':<34} skipped — only {len(both)} "
          f"author(s) have both story and essay")

    w()
    w("-" * 74)
    if r_all and r_prose:
        w(f"  poems/drama removed        : {r_all[0]:.3f} -> {r_prose[0]:.3f}  "
          f"({r_prose[0]-r_all[0]:+.3f})")
    if r_prose and r_trunc:
        w(f"  length equalised at {TRUNC_WORDS}w   : {r_prose[0]:.3f} -> {r_trunc[0]:.3f}  "
          f"({r_trunc[0]-r_prose[0]:+.3f})")
    w("-" * 74)
    w("  The differences between rows matter more than any single number.")

    if r_prose:
        w()
        w("Per-author detail, run (c):")
        w(classification_report(r_prose[1].author, r_prose[2], zero_division=0))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n  {path}")


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(PRELIM, exist_ok=True)
    df = load()
    print(f"{len(df)} works · {df.author.nunique()} authors · "
          f"{df.word_count.sum():,} words\n")

    summary = df.groupby("author").agg(
        works=("work_id", "count"),
        words=("word_count", "sum"),
        median_words=("word_count", "median"),
        shortest=("word_count", "min"),
        longest=("word_count", "max"),
        types=("wtype", lambda s: ", ".join(sorted(set(s)))),
    ).sort_values("works", ascending=False)
    summary.to_csv(os.path.join(PRELIM, "corpus_summary.csv"))
    print(summary.to_string())
    print()

    print("writing figures:")
    pies(df, os.path.join(PRELIM, "distribution_pies.png"))
    lengths(df, os.path.join(PRELIM, "work_lengths.png"), log=True)
    lengths(df, os.path.join(PRELIM, "work_lengths_linear.png"), log=False)
    print(f"  {os.path.join(PRELIM, 'corpus_summary.csv')}")

    print("\nrunning baselines (first sklearn import can be slow — do not Ctrl+C)\n")
    baselines(df, os.path.join(PRELIM, "baselines.txt"))
    print(f"\neverything is in {PRELIM}/")


if __name__ == "__main__":
    main()
