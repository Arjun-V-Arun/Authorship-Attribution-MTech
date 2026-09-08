# Authorship Attribution (M.Tech)

A leakage-free authorship attribution (AA) benchmark, built in two parts.

**Part 1 — Hindi (complete).** Reproduces a prior 2014 undergraduate
project's results, shows that those results were inflated by topic/book
leakage in the evaluation setup, and measures the *honest* performance of
classical, stylometric, and transformer-based approaches once that leakage
is removed. Four authors: **Dharamvir Bharti**, **Munshi Premchand**,
**Sarat Chandra Chattopadhyay**, **Vibhuti Narain Rai**.

**Part 2 — Malayalam (in progress).** A new corpus — the first dedicated
Malayalam AA benchmark — crawled from the Sayahna Foundation digital
library, covering fiction (114 works, 12 authors) and essays (~600 works,
16-17 authors). Applies the same leakage-control methodology, and goes
further: it finds and quantifies a subtler leak (author biographies embedded
in the source text) and a Malayalam-specific tokenization bug in standard
NLP tooling.

Start with [`docs/project-summary.md`](docs/project-summary.md) for the full
research narrative, or [`docs/assessment-and-roadmap.md`](docs/assessment-and-roadmap.md)
for a detailed, ranked assessment of the Malayalam work and what to do next.

## Repository structure

| Path | What's in it |
|---|---|
| [`src/`](src/) | All code, for both parts. **Start here** to run anything — see [`src/README.md`](src/README.md) for what each script does. |
| [`docs/`](docs/) | Research write-ups: project summary, the Malayalam assessment/roadmap, and the literature review. |
| [`data/`](data/) | Hindi source text (`raw/`, `snippets_2014/`) and small Malayalam reference material (`malayalam/`). See [`data/README.md`](data/README.md). |
| [`sayahna-fiction/`](sayahna-fiction/), [`sayahna-essays/`](sayahna-essays/) | The Malayalam corpora, each self-contained: raw/clean/strict text, TEI XML, metadata, and per-corpus diagnostic results in `corpus/prelim/`. See each folder's README. |
| [`sayahna-crossgenre/`](sayahna-crossgenre/) | Generated merge of the two Malayalam corpora for the (currently blocked) cross-genre experiment. |
| [`embeddings/`](embeddings/) | Cached frozen-transformer embeddings (`.npy`) for the Hindi benchmark, so re-running the rigor/ablation study doesn't require re-encoding. |
| [`results/`](results/) | Small, human-readable result artifacts: Hindi results at the top level, Malayalam results in [`results/malayalam/`](results/malayalam/). |
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

## Reproducing the Hindi benchmark

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
the method, was driving the original ~99% numbers.

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
```

See [`src/README.md`](src/README.md) for what each script does and the
current status of the several script versions (`low_baselines.py` / `_v2` /
`_v3`, `baseline_ladder.py` / `-v2`) — some diagnostic scripts have more than
one iteration kept for provenance, and it's worth confirming which one you
mean to run before a new diagnostic pass.

### Headline result

Baseline ladder ([`results/malayalam/baseline-ladder.md`](results/malayalam/baseline-ladder.md)),
macro-F1 on `texts_strict`, stratified CV — char and word n-gram SVMs reach
0.88–0.95, well above Burrows' Delta (English-style function-word
stylometry, 0.66–0.79) and well above chance (0.06–0.08). But a diagnostic
sweep ([`results/malayalam/comparison.md`](results/malayalam/comparison.md))
shows these numbers are inflated the same way the Hindi ones were — by
book/topic leakage and, distinctively, by author-biography paragraphs
embedded in the source text. The honest, leakage-controlled estimate
converges on macro-F1 ≈ 0.75 (fiction) and ≈ 0.78 (essays). See
[`docs/assessment-and-roadmap.md`](docs/assessment-and-roadmap.md) for the
full analysis, including a Malayalam-specific tokenization bug found in
standard NLP tooling along the way.

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
