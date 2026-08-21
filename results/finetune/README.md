# results/finetune/ — fine-tuning run artifacts

Copied out of the (gitignored) `runs/` and `logs/` folders after running
`src/run_experiments.py`, keeping only what's small and human-readable. The
model checkpoints (`ckpt_*.pt`, 1.6–2.1GB each) stay local-only in `runs/` —
regenerate them with `src/train_finetune.py --run_name <name> --resume`.

| Run | Model | Files |
|---|---|---|
| `distil_ft` | `distilbert-base-multilingual-cased` | `curve_distil_ft.png`, `results_distil_ft.csv`/`.xlsx`, `finetune_distilmbert.log` |
| `mbert_ft` | `bert-base-multilingual-cased` | `curve_mbert_ft.png`, `results_mbert_ft.csv`/`.xlsx`, `finetune_mbert.log` |

`frozen_indicbert.log` and `frozen_muril.log` are from the frozen-embedding
jobs (`src/dl_baseline.py`), not fine-tuning runs, but were logged alongside
these by the same `run_experiments.py` batch.
