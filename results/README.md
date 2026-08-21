# results/ — tracked result artifacts

Small, human-readable outputs worth keeping in version control (as opposed to
`runs/`, which holds multi-GB model checkpoints and is gitignored).

| File | Produced by | What it is |
|---|---|---|
| `results_rigor.csv` | `src/rigor_ablation.py` | Honest (leave-one-book-out) macro-F1, 95% bootstrap CIs, and leaky-vs-honest leakage gap for every method compared. |
| `fig_ci.png` | `src/fig_ci.py` | Plot of `results_rigor.csv` — the headline figure in the root README. |
| `run_manifest.csv` | `src/run_experiments.py` | Ledger of unattended experiment runs: job name, start/end time, duration, status. |
| `sample-run-output_2026-08-11.txt` | `src/baseline.py` | A captured stdout log from a real run, kept as a reference for what "working correctly" looks like. |
| `finetune/` | `src/train_finetune.py` (copied out of `runs/`) | Per-epoch training curves and metrics for the two fine-tuning runs actually completed (DistilBERT, mBERT) — the small artifacts only; checkpoints stay in the gitignored `runs/`. |
