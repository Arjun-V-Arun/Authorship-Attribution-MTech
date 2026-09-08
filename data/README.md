# data/ — text data used by src/

## `raw/<author>/<book>`

Raw, mostly-unprocessed per-book text files, one folder per author:
`dharamvir`, `prem`, `sarat`, `vibhuti` (plus a leftover `rnt` folder from the
2014 project that none of the current scripts read — the current `AUTHORS`
list in every `src/*.py` script is just the four above; Tagore-authored
material was excluded upstream in the 2014 project since his available texts
are heterogeneous multi-translator translations).

This is the data `src/book_disjoint.py`, `src/dl_baseline.py`,
`src/train_finetune.py`, and `src/rigor_ablation.py` read directly: each book
is tokenized and cut into 500-token snippets, tagged with `(author, book_id)`,
so a book-disjoint (leave-one-book-out) split is possible.

## `snippets_2014/<author>.split/*`

Pre-split 500-token snippet files (one token per line), in the exact format
the original 2014 project produced via
`legacy-cs365-2014/preprocessing-scripts/split-folder.zsh`. This is what
`src/baseline.py` reads to reproduce the original (leaky, shuffled-CV)
evaluation — snippets here are **not** tagged with book of origin, which is
exactly the information gap that made the original evaluation leaky.

Both are plain-text data checked into this repo for reproducibility; see the
root [README](../README.md#data--copyright-note) for a copyright note.

## `malayalam/`

Small reference material for the Malayalam benchmark (Part 2): the corpus
availability sheet and a scratch notebook. This is **not** where the
Malayalam corpus itself lives — that's [`sayahna-fiction/`](../sayahna-fiction/)
and [`sayahna-essays/`](../sayahna-essays/) at the repo root, each self-contained
with its own `corpus/` subfolder (raw/clean/strict text, TEI XML, metadata).
See [`docs/project-summary.md`](../docs/project-summary.md) for how those were
built.

| File | What it is |
|---|---|
| `corpus-availability-sheet.xlsx` | Tracks which Sayahna authors/works were considered and why they were included or excluded. |
| `granthappura.ipynb` | Scratch notebook used while exploring an alternative Malayalam source (granthappura). |
