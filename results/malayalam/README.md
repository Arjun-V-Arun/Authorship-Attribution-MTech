# results/malayalam/ — Malayalam benchmark results

Cross-corpus (fiction + essays) summaries and comparisons. Per-corpus
diagnostic output (the raw `low-baselines*.txt`, per-condition CSVs, and
confusion-matrix figures) stays with its corpus, in
[`sayahna-fiction/corpus/prelim/`](../../sayahna-fiction/corpus/prelim/) and
[`sayahna-essays/corpus/prelim/`](../../sayahna-essays/corpus/prelim/).

| File | Produced by | What it is |
|---|---|---|
| `baseline-ladder.md` | `src/baseline_ladder-v2.py` (current) | The model ladder — FLOOR, Burrows' Delta, char/word n-gram SVMs, stylometric features, L2 ablations — macro-F1 on `texts_strict` for both corpora. |
| `baseline-ladder-v1-superseded.md` | `src/baseline_ladder.py` (earlier version) | The same ladder from an earlier run, kept for provenance. Numbers differ from `baseline-ladder.md`; treat the latter as current. |
| `comparison.md` / `comparison.csv` | `src/collect_results.py`, after `run_diagnostics.ps1` | The leakage-diagnostic sweep (masking, balancing, truncation, permutation) across raw/clean/strict text for both corpora, converging on the honest COMBINED macro-F1 estimate. |
| `morphology_stats/` | `src/morphology_stats.py` | Corpus-only statistics (top-k coverage, Heaps' law vocabulary growth) explaining the word-vs-character and Delta results. `sayahna-both/` covers just the Malayalam corpora; `sayahna_both_and_hindi/` adds the Hindi corpus for a cross-language comparison. |

See [`docs/project-summary.md`](../../docs/project-summary.md) and
[`docs/assessment-and-roadmap.md`](../../docs/assessment-and-roadmap.md) for
interpretation.
