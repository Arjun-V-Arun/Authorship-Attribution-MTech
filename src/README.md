# src/ — current code

The M.Tech work: reproduces the 2014 project, diagnoses why its numbers were
inflated, and builds an honest benchmark on top of the fix. All scripts are
meant to be run **from the repo root** (`python src/<script>.py`), since their
default data paths (`data/raw`, `data/snippets_2014`, `embeddings/`,
`results/`) are relative to the current directory, not to the script's own
location.

| Script | Purpose |
|---|---|
| `baseline.py` | Reproduces the 2014 setup on `data/snippets_2014` (word/char TF-IDF + LinearSVC/LogReg/RandomForest, shuffled 5-fold CV). This is the *leaky* baseline. |
| `book_disjoint.py` | The core fix: cuts `data/raw` into 500-token snippets tagged by book, then compares shuffled CV (leaky) against leave-one-book-out CV (honest) on identical features. |
| `dl_baseline.py` | Same book-disjoint evaluation, but with frozen pretrained transformer embeddings (mBERT / MuRIL / IndicBERT) instead of TF-IDF. Caches embeddings to `embeddings/`. |
| `train_finetune.py` | Actually fine-tunes a transformer (book-disjoint train/val split), with per-epoch checkpointing to `runs/` so long unattended runs can resume with `--resume`. |
| `rigor_ablation.py` | Bootstrap 95% CIs, pairwise significance tests, char n-gram width ablation, and a function-word-only view — reads any `embeddings/emb_*.npy` caches it finds. Writes `results/results_rigor.csv`. |
| `run_experiments.py` | Orchestrates several of the above back-to-back unattended, logging each to `logs/` and `results/run_manifest.csv`. |
| `fig_ci.py` | Plots `results/results_rigor.csv` into `results/fig_ci.png`. |

See each script's module docstring for exact CLI flags and an example
invocation.
