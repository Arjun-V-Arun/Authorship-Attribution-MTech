# Authorship Attribution (M.Tech)

A leakage-free Hindi authorship attribution benchmark. It reproduces a prior
2014 undergraduate project's results, shows that those results were inflated
by topic/book leakage in the evaluation setup, and then measures the *honest*
performance of classical, stylometric, and transformer-based approaches once
that leakage is removed.

Four authors: **Dharamvir Bharti**, **Munshi Premchand**, **Sarat Chandra
Chattopadhyay**, **Vibhuti Narain Rai**.

## Folder guide

| Folder | What's in it |
|---|---|
| [`src/`](src/) | The current code: classical baselines, the leakage analysis, transformer fine-tuning, and the statistical rigor/ablation study. **Start here.** |
| [`data/`](data/) | The text data those scripts read: raw per-book texts and pre-split 500-token snippets. |
| [`embeddings/`](embeddings/) | Cached frozen-transformer embeddings (`.npy`), so re-running the rigor/ablation study doesn't require re-encoding. |
| [`results/`](results/) | Small, human-readable result artifacts: figures, CSVs, a sample run log. |
| [`legacy-cs365-2014/`](legacy-cs365-2014/) | The original 2014 IIT-K CS365 course project this work builds on and corrects — downloaded archives, preprocessing scripts, and the original (Python 2) ML scripts. Kept for provenance, not maintained. |
| `logs/`, `runs/` | Local-only, gitignored. Recreated by `src/run_experiments.py` / `src/train_finetune.py` (checkpoints here can run into GBs, far past what git/GitHub can hold). |
| `envAA/` | Local Python virtual environment, gitignored. |

## Quickstart

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, see requirements.txt

# Run from the repo root so the default data paths (data/raw, data/snippets_2014) resolve:
python src/baseline.py            # reproduces the (leaky) 2014-style evaluation
python src/book_disjoint.py       # exposes the leakage: shuffled CV vs. leave-one-book-out
python src/dl_baseline.py         # frozen transformer embeddings + linear head, honest eval
python src/train_finetune.py --run_name my_run --epochs 4   # fine-tune a transformer
python src/rigor_ablation.py      # bootstrap CIs, significance tests, ablations
python src/fig_ci.py              # plot results/results_rigor.csv -> results/fig_ci.png
```

## Headline result

The 2014 project's shuffled cross-validation reported ~99% macro-F1 for every
method — because snippets from the same book leak across train/test, so the
model is largely just memorizing topic/vocabulary, not writing style. Under a
leave-one-book-out (honest) evaluation, from [`results/results_rigor.csv`](results/results_rigor.csv):

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

## Data & copyright note

`data/` and `legacy-cs365-2014/archives/` contain the literary texts used for
this benchmark. Premchand and Sarat Chandra Chattopadhyay's works are in the
public domain in India; Dharamvir Bharti and Vibhuti Narain Rai's works are
still under copyright. They're included here for academic/research
reproducibility, sourced (via the 2014 project) from [hindisamay.com](https://www.hindisamay.com).
