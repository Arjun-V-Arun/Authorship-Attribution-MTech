#!/usr/bin/env python3
"""
morpheme_experiment.py — the causal test of the inflection hypothesis.

THE ARGUMENT SO FAR (correlational)
    Malayalam's top 300 word forms cover ~20-26% of running text; Heaps' beta
    is ~0.87; 77% of types are hapax. Burrows' Delta, which assumes the top-k
    forms cover most of a text, scores 0.656. Word features beat character
    features. All consistent with inflectional fragmentation -- but consistent
    is not the same as demonstrated.

THE TEST (causal, and entirely within Malayalam)
    If fragmentation is the cause, then splitting words into morphemes should
    collapse the vocabulary, raise top-k coverage, and REPAIR Delta. If Delta
    does not improve on segmented text, the explanation is wrong and should be
    withdrawn.

    Unsupervised segmentation (Morfessor) is used deliberately: no Malayalam
    morphological analyser of research quality is available, and an
    unsupervised method cannot be accused of encoding the answer.

WHAT IT RUNS
    coverage and Heaps' beta   raw vs punctuation-stripped vs segmented
    Burrows' Delta             raw vs segmented          <- the decisive row
    word 1-2 gram + SVM        raw vs segmented
    morpheme 1-3 gram + SVM    segmented only            <- a new feature type
    char 3-5 gram + SVM        raw, as the fixed reference

Usage:
    pip install morfessor scikit-learn pandas matplotlib regex
    python morpheme_experiment.py --root . --corpora sayahna-fiction sayahna-essays \\
        --texts texts_strict
    python morpheme_experiment.py --root . --corpora sayahna-fiction --quick

Output, at --root:
    morpheme_experiment.md
    segmentation_examples.txt     100 words with their segmentations, to read
and per corpus:
    corpus/segmented/<id>.txt     the segmented text, reusable
"""

import argparse
import os
import re
import sys
import glob
import math
from collections import Counter

import unicodedata

import numpy as np
import pandas as pd

SEED = 20260828
KS = [100, 300, 1000, 10000]
DROP_TYPES = {"poetry", "poem", "drama"}
MIN_DOC_WORDS = 100
MIN_WORKS = 5
CANON = dict(sublinear_tf=True, min_df=2, max_features=100_000)
CANON_SVC = dict(C=1.0, class_weight="balanced", max_iter=5000)

def merge_combining(pieces):
    """
    Merge any piece starting with a combining mark into the piece before it,
    so nothing is split below a grapheme cluster.

    WHETHER THIS IS CORRECT FOR MALAYALAM IS NOT OBVIOUS, which is why it is a
    flag and not the default. Two different cases look identical to code:

        \u0d38 + \u0d48            BAD.  A vowel sign cut from its consonant.
                              No morpheme boundary, just half a syllable.
        \u0d38\u0d42\u0d15\u0d4d\u0d37\u0d3f\u0d1a\u0d4d\u0d1a + \u0d3f\u0d30\u0d41\u0d28\u0d4d\u0d28   GOOD. Verb stem + auxiliary
                              \u0d07\u0d30\u0d41\u0d28\u0d4d\u0d28. Sandhi renders \u0d1a\u0d4d + \u0d07 as \u0d1a\u0d3f, so a real
                              morpheme boundary sits INSIDE a cluster.

    Separating them needs a morphological analyser. Run the experiment both
    ways and report the difference rather than assuming.
    """
    out = []
    for p in pieces:
        body = p[1:] if p.startswith("\u2581") else p
        starts_combining = bool(body) and (
            unicodedata.category(body[0]) in ("Mn", "Mc")
            or body[0] in ("\u200c", "\u200d")
        )
        if starts_combining and out:
            out[-1] += body
        else:
            out.append(p)
    return out


PUNCT = re.compile(r"[.,;:!?()\[\]{}\"'\u2018\u2019\u201c\u201d\u2014\u2013\u2026"
                   r"\u0964\u0965\-/]+")

OUT = []


def w(s=""):
    print(s, flush=True)
    OUT.append(s)


def strip_punct(t):
    return " ".join(x for x in PUNCT.sub(" ", t).split() if x)


# --------------------------------------------------------------- corpus io

def load(corpus_dir, textdir):
    meta = os.path.join(corpus_dir, "corpus", "metadata.csv")
    tdir = os.path.join(corpus_dir, "corpus", textdir)
    if not os.path.isdir(tdir):
        return None, f"no corpus/{textdir}"
    authors = {}
    if os.path.exists(meta):
        df = pd.read_csv(meta, dtype=str).fillna("")
        for _, r in df.iterrows():
            if str(r.get("work_type", "")).lower() in DROP_TYPES:
                continue
            authors[str(r["work_id"])] = str(r.get("author", ""))
    rows = []
    for p in sorted(glob.glob(os.path.join(tdir, "*.txt"))):
        wid = os.path.splitext(os.path.basename(p))[0]
        if authors and wid not in authors:
            continue
        t = open(p, encoding="utf-8").read()
        if len(t.split()) < MIN_DOC_WORDS:
            continue
        rows.append({"work_id": wid, "author": authors.get(wid, ""), "text": t})
    d = pd.DataFrame(rows)
    if d.empty:
        return None, "no texts"
    vc = d.author.value_counts()
    d = d[d.author.isin(vc[vc >= MIN_WORKS].index)].reset_index(drop=True)
    for c in d.columns:
        d[c] = d[c].astype(object)
    return d, None


# ------------------------------------------------------------- statistics

def coverage(tokens):
    c = Counter(tokens)
    total = len(tokens)
    cum = np.cumsum([n for _, n in c.most_common()])
    out = {k: float(cum[min(k, len(cum)) - 1] / total) if len(cum) else 0.0
           for k in KS}
    hapax = sum(1 for v in c.values() if v == 1)
    return out, len(c), total, hapax / max(1, len(c))


def heaps(tokens, points=20):
    n = len(tokens)
    if n < 2000:
        return float("nan")
    idx = np.unique(np.logspace(math.log10(500), math.log10(n), points).astype(int))
    seen, V, N, prev = set(), [], [], 0
    for cut in idx:
        seen.update(tokens[prev:cut])
        prev = cut
        V.append(len(seen))
        N.append(cut)
    return float(np.polyfit(np.log(N), np.log(V), 1)[0])


# ---------------------------------------------------------- segmentation

def train_sp(texts, vocab_size, workdir, protect_graphemes=False):
    """
    Unigram subword model with an EXPLICIT vocabulary size. Smaller vocab =
    smaller units. Sweeping it traces the word -> morpheme -> character axis
    directly, which is what the hypothesis is about.
    """
    import sentencepiece as spm
    os.makedirs(workdir, exist_ok=True)
    corpus = os.path.join(workdir, "sp_input.txt")
    # sentencepiece silently drops lines longer than ~4 KB, so write short
    # chunks rather than one line per document (which yields an empty corpus).
    CHUNK = 40
    with open(corpus, "w", encoding="utf-8") as f:
        n_lines = 0
        for t in texts:
            toks = strip_punct(t).split()
            for i in range(0, len(toks), CHUNK):
                piece = " ".join(toks[i:i + CHUNK]).strip()
                if piece:
                    f.write(piece + "\n")
                    n_lines += 1
    if n_lines == 0:
        raise RuntimeError("no text to train the subword model on")
    prefix = os.path.join(workdir, f"sp{vocab_size}")
    try:
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=prefix, vocab_size=vocab_size,
            model_type="unigram", character_coverage=0.9995, minloglevel=3)
    except RuntimeError as e:
        m = re.search(r"<= (\d+)", str(e))
        if not m:
            raise
        cap = int(m.group(1))
        print(f"    vocab {vocab_size} too large, falling back to {cap}")
        vocab_size = cap
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=prefix, vocab_size=vocab_size,
            model_type="unigram", character_coverage=0.9995, minloglevel=3)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")

    def encode(t, protect=protect_graphemes):
        pieces = sp.encode(strip_punct(t), out_type=str)
        if protect:
            pieces = merge_combining(pieces)
        return " ".join(pieces)

    return encode, vocab_size


def train_segmenter(texts, corpusweight=None, seed=SEED):
    try:
        import morfessor
    except ImportError:
        sys.exit("morfessor not installed.  pip install morfessor")
    counts = Counter()
    for t in texts:
        counts.update(strip_punct(t).split())
    words = [(wd, n) for wd, n in counts.items() if len(wd) > 1]
    model = (morfessor.BaselineModel(corpusweight=corpusweight)
             if corpusweight else morfessor.BaselineModel())
    # morfessor's load_data signature differs across releases: some expect
    # (count, atoms), others (count, compound, atoms). Try both.
    for shape in ("pair", "triple"):
        try:
            if shape == "pair":
                model.load_data([(n, tuple(wd)) for wd, n in words])
            else:
                model.load_data([(n, wd, tuple(wd)) for wd, n in words])
            break
        except ValueError:
            model = (morfessor.BaselineModel(corpusweight=corpusweight)
                     if corpusweight else morfessor.BaselineModel())
            continue
    else:
        sys.exit("could not load data into morfessor; check its version")
    model.train_batch()
    cache = {}

    def seg(word):
        if word not in cache:
            try:
                cache[word] = model.viterbi_segment(word)[0]
            except Exception:
                cache[word] = [word]
        return cache[word]

    return seg, len(counts)


def segment_text(t, seg, marker="\u2039"):
    """Word-initial morphemes get a marker so boundaries stay visible."""
    out = []
    for wd in strip_punct(t).split():
        parts = seg(wd)
        if not parts:
            continue
        out.append(marker + parts[0])
        out.extend(parts[1:])
    return " ".join(out)


# ---------------------------------------------------------------- models

def cv_macro(X, y, model, repeats=3):
    """
    Manual stratified k-fold. Deliberately not cross_val_predict: that requires
    the full sklearn estimator protocol, which changes between versions, and
    BurrowsDelta is a hand-written estimator. fit/predict is all we need.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.base import clone
    y = np.asarray(list(y), dtype=object)
    X = np.asarray(list(X), dtype=object)
    k = int(min(5, pd.Series(y).value_counts().min()))
    if k < 2:
        return float("nan"), float("nan")
    sc = []
    for i in range(repeats):
        cv = StratifiedKFold(k, shuffle=True, random_state=SEED + i)
        pred = np.empty(len(y), dtype=object)
        for tr, te in cv.split(X, y):
            try:
                m = clone(model)
            except Exception:
                m = model.__class__(**model.get_params()) \
                    if hasattr(model, "get_params") else model
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        sc.append(f1_score(y, pred, average="macro"))
    return float(np.mean(sc)), float(np.std(sc))


def make_models():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    ML = r"\S+"

    def word_ng(lo, hi):
        return make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(lo, hi),
                            token_pattern=ML, lowercase=False, **CANON),
            LinearSVC(**CANON_SVC))

    def char_ng(lo, hi):
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi), **CANON),
            LinearSVC(**CANON_SVC))

    return word_ng, char_ng


# ------------------------------------------------------------------ main

class BurrowsDelta:
    """Same implementation as the ladder, inlined so this script stands alone."""

    def __init__(self, k=300):
        self.k = k

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **p):
        self.k = p.get("k", self.k)
        return self

    def fit(self, X, y):
        freq = Counter()
        for d in X:
            freq.update(d.split())
        self.vocab_ = [t for t, _ in freq.most_common(self.k)]
        M = self._mat(X)
        self.mu_, self.sd_ = M.mean(0), M.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (M - self.mu_) / self.sd_
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.cent_ = np.vstack([Z[y == c].mean(0) for c in self.classes_])
        return self

    def _mat(self, X):
        idx = {t: i for i, t in enumerate(self.vocab_)}
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
        return self.classes_[np.abs(Z[:, None, :] - self.cent_[None, :, :]).sum(2).argmin(1)]


def main(a):
    word_ng, char_ng = make_models()
    reps = 1 if a.quick else 3
    all_stats = []

    for c in a.corpora:
        cdir = os.path.join(a.root, c)
        if not os.path.isdir(cdir):
            print(f"skipping {c}: not found")
            continue
        d, err = load(cdir, a.texts)
        if d is None:
            print(f"skipping {c}: {err}")
            continue
        name = c.replace("sayahna-", "")

        w("=" * 78)
        w(f"MORPHEME EXPERIMENT — {name}")
        w("=" * 78)
        w(f"{len(d)} works, {d.author.nunique()} authors")
        w("")

        raw = list(d.text)
        nop = [strip_punct(t) for t in raw]

        work = os.path.join(cdir, "corpus", "_sp")
        variants = {}
        for vs in a.vocab_sizes:
            print(f"  training subword model, vocab={vs} ...")
            enc, actual = train_sp(raw, vs, work, a.protect_graphemes)
            variants[actual] = [enc(t) for t in raw]
        if a.compare_graphemes:
            best = sorted(variants)[len(variants) // 2]
            print(f"  also building grapheme-protected variant at v={best} ...")
            enc_p, _ = train_sp(raw, best, work, protect_graphemes=True)
            variants_protected = [enc_p(t) for t in raw]
        else:
            variants_protected = None
        segd = variants[sorted(variants)[0]]          # smallest units

        sdir = os.path.join(cdir, "corpus", "segmented")
        os.makedirs(sdir, exist_ok=True)
        for wid, t in zip(d.work_id, segd):
            open(os.path.join(sdir, f"{wid}.txt"), "w", encoding="utf-8").write(t)

        # ---- vocabulary statistics
        w("-" * 78)
        w("1. VOCABULARY — does segmentation collapse the type inventory?")
        w("-" * 78)
        rows = []
        pairs = [("raw (whitespace)", raw), ("punctuation stripped", nop)]
        pairs += [(f"subword vocab={v:,}", variants[v]) for v in sorted(variants)]
        for label, texts in pairs:
            toks = " ".join(texts).split()
            cov, ntyp, ntok, hap = coverage(toks)
            b = heaps(toks)
            rows.append((label, cov, ntyp, ntok, hap, b))
        w(f"  {'version':<24}{'types':>10}{'top100':>9}{'top300':>9}"
          f"{'top1k':>9}{'hapax':>8}{'beta':>8}")
        for label, cov, ntyp, ntok, hap, b in rows:
            w(f"  {label:<24}{ntyp:>10,}{cov[100]:>8.1%}{cov[300]:>8.1%}"
              f"{cov[1000]:>8.1%}{hap:>7.0%}{b:>8.3f}")
        base, strip_r = rows[0], rows[1]
        segr = min(rows[2:], key=lambda r: r[2]) if len(rows) > 2 else rows[-1]
        w("")
        w(f"  punctuation alone accounts for "
          f"{(base[2]-strip_r[2])/base[2]:.1%} of the raw type count")
        cut = (strip_r[2] - segr[2]) / max(1, strip_r[2])
        w(f"  at the smallest vocabulary, segmentation cuts types by {cut:.1%} "
          f"and lifts top-300 coverage {strip_r[1][300]:.1%} -> {segr[1][300]:.1%}")
        w("  (the sweep above matters more than any single row: coverage rises")
        w("   monotonically as the unit shrinks)")
        if cut < 0.05:
            w("")
            w("  !! Segmentation barely changed the vocabulary, so sections 2 and 3")
            w("     below test nothing. Morfessor kept words whole. Re-run with")
            w("     --corpusweight 0.1 (or 0.01) to make it split more.")

        # ---- the decisive experiment
        w("")
        w("-" * 78)
        w("2. DOES SEGMENTATION REPAIR BURROWS' DELTA?")
        w("-" * 78)
        w("  If inflectional fragmentation is what breaks Delta, this is where")
        w("  it shows. No improvement means the explanation is wrong.")
        w("")
        res = {}
        delta_runs = [("Delta (300 MFW), raw words", nop, BurrowsDelta(300))]
        for v in sorted(variants):
            delta_runs.append((f"Delta (300), subword v={v:,}",
                               variants[v], BurrowsDelta(300)))
        if variants_protected is not None:
            delta_runs.append(("Delta (300), grapheme-protected",
                               variants_protected, BurrowsDelta(300)))
        delta_runs.append(("Delta (1000), raw words", nop, BurrowsDelta(1000)))
        for label, texts, model in delta_runs:
            m, s = cv_macro(texts, d.author, model, reps)
            res[label] = m
            w(f"  {label:<34} macro-F1={m:.3f} (+/-{s:.3f})")
        dr = res.get("Delta (300 MFW), raw words")
        sub_keys = [k for k in res if k.startswith("Delta (300), subword")]
        ds = max((res[k] for k in sub_keys), default=None)
        if dr and ds:
            w("")
            w(f"  >>> DELTA, raw -> segmented: {dr:.3f} -> {ds:.3f} ({ds-dr:+.3f})")
            if dr > 0.95:
                w("      UNINFORMATIVE: raw Delta is already at ceiling, so there is")
                w("      no headroom for segmentation to improve into. This test says")
                w("      nothing either way. Make the task harder (more authors, or")
                w("      truncate documents) before drawing a conclusion.")
            elif ds - dr > 0.03:
                w("      Segmentation repairs Delta. The fragmentation explanation is")
                w("      supported causally, not just by correlation. This is the")
                w("      result that turns the coverage statistics into a mechanism.")
            elif ds - dr < -0.03:
                w("      Segmentation HURTS Delta. The fragmentation explanation does")
                w("      not hold; report the coverage statistics as description only")
                w("      and drop the causal claim.")
            else:
                w("      No real change. The explanation is not supported causally.")
                w("      Report coverage as a descriptive statistic, not a mechanism.")

        # ---- morphemes as a feature type in their own right
        w("")
        w("-" * 78)
        w("3. ARE MORPHEMES A USEFUL FEATURE FOR ATTRIBUTION?")
        w("-" * 78)
        feat_runs = [("char 3-5 gram, raw (reference)", raw, char_ng(3, 5)),
                     ("word 1-2 gram, raw", nop, word_ng(1, 2))]
        for v in sorted(variants):
            feat_runs.append((f"subword 1-2 gram, v={v:,}", variants[v], word_ng(1, 2)))
        feat_runs.append((f"subword 1-3 gram, v={sorted(variants)[0]:,}",
                          segd, word_ng(1, 3)))
        for label, texts, model in feat_runs:
            m, s = cv_macro(texts, d.author, model, reps)
            res[label] = m
            w(f"  {label:<34} macro-F1={m:.3f} (+/-{s:.3f})")

        all_stats.append((name, rows, res))
        w("")

    # ---- readable segmentation samples, so you can judge quality yourself
    if all_stats:
        cdir = os.path.join(a.root, a.corpora[0])
        d, _ = load(cdir, a.texts)
        if d is not None:
            enc, _ = train_sp(list(d.text), sorted(a.vocab_sizes)[0],
                              os.path.join(cdir, "corpus", "_sp"),
                              a.protect_graphemes)
            seg = lambda wd: enc(wd).split()
            cnt = Counter(strip_punct(" ".join(list(d.text)[:60])).split())
            lines = ["Unsupervised subword segmentation samples (sentencepiece,",
                     "unigram model). Judge these yourself: a good split separates",
                     "stem from inflection, a bad one cuts mid-stem.",
                     "",
                     "NOTE: a piece may legitimately begin with a vowel sign.",
                     "Sandhi puts real morpheme boundaries inside grapheme",
                     "clusters, so not every such split is an error.",
                     ""]
            for wd, n in cnt.most_common(60):
                lines.append(f"  {wd:<28} -> {' + '.join(seg(wd))}   (x{n})")
            for wd, n in [x for x in cnt.items() if len(x[0]) > 12][:40]:
                lines.append(f"  {wd:<28} -> {' + '.join(seg(wd))}   (x{n})")
            p = os.path.join(a.root, "segmentation_examples.txt")
            open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            print(f"\nwritten: {p}")

    out = os.path.join(a.root, "morpheme_experiment.md")
    open(out, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="*",
                    default=["sayahna-fiction", "sayahna-essays"])
    ap.add_argument("--texts", default="texts_strict")
    ap.add_argument("--quick", action="store_true", help="1 seed instead of 3")
    ap.add_argument("--vocab-sizes", type=int, nargs="*",
                    default=[2000, 8000, 32000],
                    help="subword vocabulary sizes to sweep. Smaller = smaller units")
    ap.add_argument("--protect-graphemes", action="store_true",
                    help="never split inside a grapheme cluster. See the note on "
                         "merge_combining: this is NOT obviously right for Malayalam")
    ap.add_argument("--compare-graphemes", action="store_true",
                    help="run protected and unprotected side by side and report both")
    ap.add_argument("--segmenter", choices=["sentencepiece", "morfessor"],
                    default="sentencepiece")
    ap.add_argument("--corpusweight", type=float, default=None,
                    help="Morfessor MDL trade-off. Lower = splits more. Try 0.1 "
                         "or 0.01 if it leaves words whole (check the types row).")
    main(ap.parse_args())
