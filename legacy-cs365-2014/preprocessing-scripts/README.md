# preprocessing-scripts/ — 2014 zsh preprocessing pipeline

From the original CS365 2014 project (zsh, macOS/Linux only):

- `doc2txt.zsh` — converts a `.doc` file to `.txt`.
- `split-folder.zsh` — strips punctuation/Hindi danda from every file in a
  folder, concatenates them, and splits the result into 500-word-per-file
  snippets (`split -l 500`). Usage: `split-folder corpus/<author>/`.
- `split-corpus.zsh` — runs `split-folder.zsh` across the whole corpus.

This is the exact process that produced the snippet format still used today
in [`data/snippets_2014/`](../../data/README.md) and read by
`src/baseline.py`.
