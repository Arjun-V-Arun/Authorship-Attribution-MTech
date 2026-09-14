# Authorship Attribution (M.Tech)

A leakage-free authorship attribution (AA) benchmark. The real contribution
is Malayalam; Hindi was the warmup that established the methodology.

**Hindi (warmup, complete).** Reproduces a prior 2014 undergraduate
project's results, shows that those results were inflated by topic/book
leakage in the evaluation setup, and measures the *honest* performance of
classical, stylometric, and transformer-based approaches once that leakage
is removed. Four authors: **Dharamvir Bharti**, **Munshi Premchand**,
**Sarat Chandra Chattopadhyay**, **Vibhuti Narain Rai**. This part exists to
work out and de-risk the leave-one-book-out / book-disjoint evaluation
methodology before applying it somewhere the answer wasn't already known —
it is not the paper.

**Malayalam (main focus).** The first dedicated authorship attribution
corpus, evaluation protocol, and published baseline for Malayalam, a
morphologically rich Dravidian language with no prior AA work. Built from
TEI-encoded digital editions crawled from the Sayahna Foundation: 114
fiction works by 12 authors, 599 essays by 16 authors. Four findings, in
increasing order of how much they generalize beyond this corpus:

1. **A corpus and a leakage-controlled protocol.** A document is a whole
   work, so chunk-level leakage across train/test is impossible by
   construction.
2. **A contamination finding, and a protocol that survives it.**
   Publisher-supplied author biographies are embedded inside `<text><body>`
   in the source markup — not in the header, where they'd be easy to strip.
   Left in place, they inflate a standard char-n-gram baseline by up to
   0.070 macro-F1. The leakage-controlled protocol returns the *same*
   0.752 macro-F1 (fiction) before and after their removal — it was not
   designed against this specific defect and turned out to be immune to it
   anyway, which is the strongest evidence for the protocol in the whole
   project.
3. **Word features beat character n-grams**, on both subcorpora — the
   opposite of the usual finding for English, where char n-grams are the
   strongest general-purpose representation.
4. **A mechanism, tested causally.** Burrows' Delta (the canonical
   stylometric baseline) badly underperforms here — 0.752 / 0.656 macro-F1
   against 0.888 / 0.919 for char n-grams — because it assumes the 300 most
   frequent *word forms* cover most of a text. Under Malayalam's inflectional
   morphology they cover only 26% (fiction) / 20% (essays) of running text.
   Re-running Delta over SentencePiece subword units instead of raw word
   forms recovers +0.132 / +0.114 macro-F1, with the same optimum
   (an 8,000-unit vocabulary) in both subcorpora — turning a correlational
   observation (low coverage) into a causal one (fixing coverage fixes the
   method).

Along the way, two Malayalam-specific tokenization bugs were found in
widely-used tooling (scikit-learn's default word pattern silently discards
Malayalam vowel signs and viramas; `char_wb` n-grams make word-order
ablations a no-op) — see [`docs/assessment-and-roadmap.md`](docs/assessment-and-roadmap.md).

An unpublished paper draft collating these Malayalam findings is kept
outside the repo (not code, not reproducible from this checkout alone) —
the source of truth for numbers is the results files linked below.

Start with [`docs/project-summary.md`](docs/project-summary.md) for the
full research narrative, or
[`docs/assessment-and-roadmap.md`](docs/assessment-and-roadmap.md) for a
detailed, ranked assessment of the Malayalam work and what to do next.

## Repository structure

| Path | What's in it |
|---|---|
| [`src/`](src/) | All code, for both parts. **Start here** to run anything — see [`src/README.md`](src/README.md) for what each script does. |
| [`docs/`](docs/) | Research write-ups: project summary, the Malayalam assessment/roadmap, and the literature review. |
| [`data/`](data/) | Hindi source text (`raw/`, `snippets_2014/`) and small Malayalam reference material (`malayalam/`). See [`data/README.md`](data/README.md). |
| [`sayahna-fiction/`](sayahna-fiction/), [`sayahna-essays/`](sayahna-essays/) | The Malayalam corpora, each self-contained: raw/clean/strict text, TEI XML, metadata, and per-corpus diagnostic results in `corpus/prelim/`. See each folder's README. |
| [`sayahna-crossgenre/`](sayahna-crossgenre/) | Generated merge of the two Malayalam corpora for the (currently blocked) cross-genre experiment. |
| [`embeddings/`](embeddings/) | Cached frozen-transformer embeddings (`.npy`) for the Hindi benchmark, so re-running the rigor/ablation study doesn't require re-encoding. |
| [`results/`](results/) | Small, human-readable result artifacts: Hindi results at the top level, Malayalam results in [`results/malayalam/`](results/malayalam/), and the two morpheme/subword-segmentation experiment runs in `results/morpheme-experiment*`. |
| [`legacy-cs365-2014/`](legacy-cs365-2014/) | The original 2014 IIT-K CS365 course project this work builds on and corrects — downloaded archives, preprocessing scripts, and the original (Python 2) ML scripts. Kept for provenance, not maintained. |
| `logs/`, `runs/` | Local-only, gitignored. Recreated by `src/run_experiments.py` / `src/train_finetune.py` (checkpoints here can run into GBs, far past what git/GitHub can hold). |
| `data/malayalam-crawl-cache/` | Local-only, gitignored. The full, unfiltered Sayahna library crawl cache (a superset of the curated fiction/essay corpora), kept locally so the crawler doesn't have to hit the site again. Regenerate with `src/sayahna_crawler.py`. |
| `envAA/` | Local Python virtual environment, gitignored. |

## Setup

```
python -m venv envAA
envAA\Scripts\activate            # Windows; use `source envAA/bin/activate` on Linux/Mac
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, see requirements.txt
```

## Reproducing the Hindi warmup

Run from the repo root so the default data paths (`data/raw`,
`data/snippets_2014`) resolve:

```
python src/baseline.py            # reproduces the (leaky) 2014-style evaluation
python src/book_disjoint.py       # exposes the leakage: shuffled CV vs. leave-one-book-out
python src/dl_baseline.py         # frozen transformer embeddings + linear head, honest eval
python src/train_finetune.py --run_name my_run --epochs 4   # fine-tune a transformer
python src/rigor_ablation.py      # bootstrap CIs, significance tests, ablations
python src/fig_ci.py              # plot results/results_rigor.csv -> results/fig_ci.png
```

### Headline result

The 2014 project's shuffled cross-validation reported ~99% macro-F1 for
every method — because snippets from the same book leak across train/test,
so the model is largely just memorizing topic/vocabulary, not writing
style. Under a leave-one-book-out (honest) evaluation, from
[`results/results_rigor.csv`](results/results_rigor.csv):

| Method | Honest macro-F1 | 95% CI | Leaky macro-F1 | Leakage gap |
|---|---|---|---|---|
| frozen MuRIL | 0.610 | [0.592, 0.628] | 0.981 | 0.371 |
| frozen MuRIL-large | 0.602 | [0.584, 0.619] | 0.983 | 0.382 |
| char n-gram (4-6) | 0.584 | [0.569, 0.599] | 0.999 | 0.415 |
| char n-gram (2-4) | 0.572 | [0.559, 0.585] | 0.997 | 0.425 |
| char n-gram (3-5) | 0.571 | [0.557, 0.584] | 0.997 | 0.426 |
| frozen mBERT | 0.527 | [0.511, 0.547] | 0.884 | 0.356 |
| word 1-gram (2014 setup) | 0.516 | [0.498, 0.533] | 0.994 | 0.478 |
| function-word top-200 | 0.386 | [0.369, 0.404] | 0.747 | 0.361 |

Every method's leaky-vs-honest gap is 0.36–0.48 macro-F1 — the leakage, not
the method, was driving the original ~99% numbers. This is the same failure
mode the Malayalam work below checks for from the start, plus a second one
(biography contamination) that Hindi didn't have.

## Reproducing the Malayalam benchmark

```
# crawl (run from inside each genre folder so the corpus/ output lands correctly)
cd sayahna-fiction && python ../src/sayahna_crawler.py --indexes shortstory && cd ..
cd sayahna-essays  && python ../src/sayahna_crawler.py --indexes article    && cd ..

# clean, then run the full diagnostic sequence across both corpora
.\run_diagnostics.ps1

# aggregate results
python src/collect_results.py --root .
python src/baseline_ladder-v2.py --root . --texts texts_strict
python src/morphology_stats.py --root . --output results/malayalam/morphology_stats/sayahna-both

# subword segmentation / Burrows' Delta repair experiment
python src/morpheme_experiment_v2.py
```

See [`src/README.md`](src/README.md) for what each script does and the
current status of the several script versions (`low_baselines.py` / `_v2` /
`_v3`, `baseline_ladder.py` / `-v2`) — some diagnostic scripts have more than
one iteration kept for provenance, and it's worth confirming which one you
mean to run before a new diagnostic pass.

### Headline results

**Baseline ladder** ([`results/malayalam/baseline-ladder.md`](results/malayalam/baseline-ladder.md)),
macro-F1 on `texts_strict`, stratified CV:

| Model | Fiction | Essays |
|---|---|---|
| chance | 0.083 | 0.062 |
| Burrows' Delta (300 MFW) | 0.752 | 0.656 |
| char 3-5 gram + SVM | 0.888 | 0.919 |
| word 1-2 gram + SVM | 0.928 | 0.953 |

Word features beat char n-grams on both subcorpora — the inverse of the
usual English result.

**Leakage diagnostic** ([`results/malayalam/comparison.md`](results/malayalam/comparison.md)):
the raw char 3-5 gram number above is itself inflated, the same way the
Hindi numbers were — by book/topic leakage and, distinctively, by
author-biography paragraphs embedded in the source text (removing them
costs the standard protocol 0.070 macro-F1 in fiction, 0.033 in essays). A
leakage-controlled, entity-masked, length-equalised protocol converges on
macro-F1 ≈ **0.752** (fiction) and ≈ **0.776–0.788** (essays), identical
before and after biography removal.

**Burrows' Delta repair** ([`results/morpheme-experiment-v2_2026-09-11_1132/morpheme_experiment.md`](results/morpheme-experiment-v2_2026-09-11_1132/morpheme_experiment.md)):
Delta's weakness traces to vocabulary coverage under inflection (top-300
word forms cover only 26.0% / 19.6% of running text). Segmenting into
SentencePiece subword units before running Delta recovers **+0.132**
(fiction: 0.735 → 0.867) and **+0.114** (essays: 0.678 → 0.793) macro-F1,
peaking at an 8,000-unit vocabulary in both subcorpora — a causal repair,
not just a correlation with the coverage statistics.

See [`docs/assessment-and-roadmap.md`](docs/assessment-and-roadmap.md) for
the full analysis, including the two Malayalam-specific tokenization bugs
found in standard NLP tooling along the way.

## Data & copyright note

`data/`, `legacy-cs365-2014/archives/`, `sayahna-fiction/`, and
`sayahna-essays/` contain the literary texts used for this benchmark.
Premchand and Sarat Chandra Chattopadhyay's works are in the public domain
in India; Dharamvir Bharti and Vibhuti Narain Rai's works are still under
copyright. They're included here for academic/research reproducibility,
sourced (via the 2014 project) from [hindisamay.com](https://www.hindisamay.com).
The Malayalam corpus is sourced from the Sayahna Foundation
([sayahna.org](https://sayahna.org)) under its Creative Commons licensing.

## Adding new experiments

- **A new script for an existing benchmark** goes in `src/`, following the
  existing convention of a module docstring explaining why it exists, what
  it reads, and what it writes.
- **A new dataset or language** should follow the Malayalam layout: its own
  top-level `<name>/corpus/` (or a `data/<name>/` entry if it's reference
  material rather than corpus text), its own README, and its results under
  `results/<name>/`.
- **New results** for an existing benchmark go in that benchmark's `results/`
  subfolder, not at the repo root — root is reserved for setup files
  (`README.md`, `requirements.txt`, `.gitignore`) and the top-level corpus/
  code/docs/results/data folders themselves.
