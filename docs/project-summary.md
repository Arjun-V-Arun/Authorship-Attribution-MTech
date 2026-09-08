# Project summary

Compiled 2026-09-07 (Hindi benchmark); updated as the Malayalam benchmark progresses.

## Goal

Build a leakage-free authorship attribution (AA) benchmark: first reproduce
and correct a flawed 2014 Hindi undergraduate project, then extend the
corrected methodology to a new Malayalam benchmark (with a planned
quantum-classical hybrid component).

## Part 1 — Hindi benchmark (complete)

Authors: Dharamvir Bharti, Munshi Premchand, Sarat Chandra Chattopadhyay,
Vibhuti Narain Rai. Data from hindisamay.com via the original 2014 project.

**Key finding:** the 2014 project's shuffled cross-validation reported ~99%
macro-F1 for every method. This was an artifact of **leakage** — snippets
from the same book appeared in both train and test, so models memorized
topic/vocabulary rather than writing style.

**Fix:** leave-one-book-out (book-disjoint) evaluation. Honest results
([`results/results_rigor.csv`](../results/results_rigor.csv)), with 95%
bootstrap CIs and leaky-vs-honest gap:

| Method | Honest F1 | Leaky F1 | Gap |
|---|---|---|---|
| frozen MuRIL | 0.610 | 0.981 | 0.371 |
| frozen MuRIL-large | 0.602 | 0.983 | 0.382 |
| char n-gram (4-6) | 0.584 | 0.999 | 0.415 |
| char n-gram (2-4) | 0.572 | 0.997 | 0.425 |
| char n-gram (3-5) | 0.571 | 0.997 | 0.426 |
| frozen mBERT | 0.527 | 0.884 | 0.356 |
| word 1-gram (2014 setup) | 0.516 | 0.994 | 0.478 |
| function-word top-200 | 0.386 | 0.747 | 0.361 |

Every method drops 0.36-0.48 macro-F1 under honest evaluation — the leakage,
not the method, drove the original ~99% numbers.

Fine-tuning attempts (mBERT, DistilBERT, book-disjoint split): both overfit
almost immediately — training accuracy climbs to ~95-98% by epoch 3-4 while
validation macro-F1 falls from ~0.35 (epoch 1) to ~0.27-0.31, i.e. worse than
several frozen-embedding baselines. Frozen MuRIL remains the best honest
performer overall.

Deliverables: `src/baseline.py`, `book_disjoint.py`, `dl_baseline.py`,
`train_finetune.py`, `rigor_ablation.py`, `fig_ci.py`, `run_experiments.py`;
cached embeddings; tracked results in `results/`; full checkpoints/logs kept
locally (gitignored).

## Part 2 — Malayalam benchmark (in progress)

New corpus built via a custom crawler (`src/sayahna_crawler.py`) against the
Sayahna Foundation digital library (books.sayahna.org), covering two genres:

| Corpus | Works | Authors |
|---|---|---|
| `sayahna-fiction/` | 114 | 12 |
| `sayahna-essays/` | 599-610 | 16-17 |

Three text-cleaning levels produced per corpus: raw / clean (site furniture,
repeated author bios removed) / strict (also strips paragraphs that name the
author) — see [`results/malayalam/comparison.md`](../results/malayalam/comparison.md).

Baseline ladder ([`results/malayalam/baseline-ladder.md`](../results/malayalam/baseline-ladder.md)),
macro-F1 on `texts_strict`, stratified CV — see that file for the current
numbers (superseded run kept alongside as `baseline-ladder-v1-superseded.md`
for provenance).

These numbers are inflated by the same class of leakage seen in Part 1
(book/topic and easy metadata shortcuts). A dedicated diagnostic sweep
([`results/malayalam/comparison.md`](../results/malayalam/comparison.md)) —
masking author-naming tokens, balancing works per author, truncating to
first-N-words, permutation tests, chance baselines — converges on a genuinely
honest COMBINED estimate of macro-F1 ≈ 0.75 (fiction) and ≈ 0.78 (essays),
well above chance (~0.06-0.08) but far below the raw char-n-gram numbers
above.

Cross-genre experiment (`src/merge_and_crossgenre.py`,
[`sayahna-crossgenre/`](../sayahna-crossgenre/)): intended to test whether a
model trained on an author's fiction can identify the same author's
non-fiction (novel test — no prior Indic AA study reports this). After
merging and deduplicating both corpora (728 → 673 works), only 1 author
(Vallathol Vasudevamenon) has >=4 works in both genres — not enough to run
the experiment yet. Needs a bigger essay/fiction overlap before this can
proceed (see [`docs/assessment-and-roadmap.md`](assessment-and-roadmap.md)
for the cross-topic replacement experiment).

Literature review ([`docs/malayalam-litreview.md`](malayalam-litreview.md)):
~140 sources surveyed and status-tagged (verified vs. reference-harvested).
Headline finding: there is no published, dedicated AA study on Malayalam
literary prose — Malayalam appears only as one row in a multilingual
benchmark (Kim et al. 2025) on Wikipedia text, not literature. Every other
major Indian language (Bengali, Urdu, Hindi, Telugu, Kannada, Tamil,
Marathi, Assamese) already has one. This is the gap the Malayalam benchmark
fills. The review also lays out a reading plan and justification track for a
planned quantum-classical hybrid component (Tier 3), to be framed without
relying on a speedup claim.

## Current state / next steps

- **Hindi benchmark:** essentially finished; honest baseline established,
  fine-tuning shown not to help under leave-one-book-out evaluation.
- **Malayalam benchmark:** corpus crawled and cleaned at 3 levels, baseline
  ladder and leakage diagnostics done; still needed: an honest (book/work-
  disjoint) evaluation of the Malayalam baseline ladder itself (comparable to
  Part 1's `book_disjoint.py`), a larger fiction/essay author overlap to
  unblock the cross-genre experiment, transformer baselines (MuRIL/IndicBERT)
  on Malayalam, and the quantum-classical hybrid component.

See [`docs/assessment-and-roadmap.md`](assessment-and-roadmap.md) for a
detailed, ranked assessment of what to do next and how this compares to
published Indic AA work.
