# src/ — current code

Two benchmarks live here: the completed **Hindi** benchmark (Part 1) and the
in-progress **Malayalam** benchmark (Part 2). All scripts are meant to be run
**from the repo root** (`python src/<script>.py`), since their default data
paths are relative to the current directory, not to the script's own
location. See [`docs/project-summary.md`](../docs/project-summary.md) for the
research narrative and [`docs/assessment-and-roadmap.md`](../docs/assessment-and-roadmap.md)
for what's next.

## Part 1 — Hindi (`data/raw`, `data/snippets_2014`, `embeddings/`, `results/`)

| Script | Purpose |
|---|---|
| `baseline.py` | Reproduces the 2014 setup on `data/snippets_2014` (word/char TF-IDF + LinearSVC/LogReg/RandomForest, shuffled 5-fold CV). This is the *leaky* baseline. |
| `book_disjoint.py` | The core fix: cuts `data/raw` into 500-token snippets tagged by book, then compares shuffled CV (leaky) against leave-one-book-out CV (honest) on identical features. |
| `dl_baseline.py` | Same book-disjoint evaluation, but with frozen pretrained transformer embeddings (mBERT / MuRIL / IndicBERT) instead of TF-IDF. Caches embeddings to `embeddings/`. |
| `train_finetune.py` | Actually fine-tunes a transformer (book-disjoint train/val split), with per-epoch checkpointing to `runs/` so long unattended runs can resume with `--resume`. |
| `rigor_ablation.py` | Bootstrap 95% CIs, pairwise significance tests, char n-gram width ablation, and a function-word-only view — reads any `embeddings/emb_*.npy` caches it finds. Writes `results/results_rigor.csv`. |
| `run_experiments.py` | Orchestrates several of the above back-to-back unattended, logging each to `logs/` and `results/run_manifest.csv`. |
| `fig_ci.py` | Plots `results/results_rigor.csv` into `results/fig_ci.png`. |

## Part 2 — Malayalam (`sayahna-fiction/`, `sayahna-essays/`, `sayahna-crossgenre/`, `results/malayalam/`)

| Script | Purpose |
|---|---|
| `sayahna_crawler.py` | Builds the corpus from books.sayahna.org: TEI XML → `corpus/xml`, `corpus/texts`, `corpus/bios`, `corpus/metadata.csv`, `corpus/authors.csv`. Run with the target genre folder (e.g. `sayahna-fiction/`) as the working directory — see its docstring. An unfiltered whole-library crawl is cached locally at `data/malayalam-crawl-cache/` (gitignored, regenerate on demand; not needed to reproduce results). |
| `find_boilerplate_v2.py` | **Current.** Finds repeated author-bio / site-furniture paragraphs; `--strip` writes `corpus/texts_clean/`, `--strip --strip-named` additionally strips paragraphs naming the author and writes `corpus/texts_strict/`. Superseded `find_boilerplate.py` (no `--strip-named`) kept for provenance. |
| `low_baselines.py` | Diagnostic sweep: how much of the headline macro-F1 survives once named entities, topic, and provenance shortcuts are removed. Writes `corpus/prelim/low-baselines*.txt`. **Two earlier iterations, `low_baselines_v2.py` and `low_baselines_v3.py`, are kept alongside it** — `run_diagnostics.ps1` currently calls `low_baselines_v3.py` specifically, while `low_baselines.py` itself was edited most recently (2026-09-08). **Before your next diagnostic run, confirm which of the three is the one you intend to run** — the edit history suggests `low_baselines.py` is the latest line of work, but it has not been reconciled against `v3`'s changes. |
| `baseline_ladder.py` / `baseline_ladder-v2.py` | The model ladder (FLOOR / Delta / char n-grams / word n-grams / stylometric features / L2 ablations). `-v2.py` is the current version — it produced the un-suffixed `corpus/prelim/baseline_ladder.csv/txt` and [`results/malayalam/baseline-ladder.md`](../results/malayalam/baseline-ladder.md). The original `baseline_ladder.py` run is kept as the `_v1` outputs and [`results/malayalam/baseline-ladder-v1-superseded.md`](../results/malayalam/baseline-ladder-v1-superseded.md). |
| `collect_results.py` | Turns the six `low-baselines*.txt` files (raw/clean/strict × fiction/essays) into [`results/malayalam/comparison.md`](../results/malayalam/comparison.md) / `comparison.csv`. Run after `run_diagnostics.ps1`. |
| `morphology_stats.py` | Corpus-only statistics (no model) explaining why word features beat character features and why Burrows' Delta underperforms: top-k coverage and Heaps' law vocabulary growth. Outputs written to `results/malayalam/morphology_stats/<label>/` (pass `--output`). |
| `merge_and_crossgenre.py` | Merges `sayahna-fiction/` and `sayahna-essays/`, deduplicates, and looks for authors with enough works in both genres to run a cross-genre experiment. Currently blocked — see [`docs/project-summary.md`](../docs/project-summary.md#part-2--malayalam-benchmark-in-progress). Writes to `sayahna-crossgenre/`. |

`run_diagnostics.ps1` (repo root) drives the diagnostic sequence
(`find_boilerplate_v2.py` → `low_baselines_v3.py`) across both corpora
unattended; see its header comment for usage.

See each script's module docstring for exact CLI flags and an example
invocation.
