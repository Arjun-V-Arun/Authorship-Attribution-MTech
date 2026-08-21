# legacy-cs365-2014/ — the original downloaded project

Everything in this folder was **downloaded**, not written for this M.Tech
work. It's the original CS365 (2014, IIT Kanpur) course project by Shetty &
Anand, retrieved from:

https://cse.iitk.ac.in/users/cs365/2014/_submissions/srijans/project/

It's kept for provenance and comparison — this M.Tech project builds directly
on it, reproduces its results (see `src/baseline.py`), and then shows that
those results were inflated by topic/book leakage in its evaluation
methodology (see `src/book_disjoint.py` and the root README's headline
result). None of the code here is maintained; it's Python 2 and won't run
as-is under a modern interpreter.

- [`archives/`](archives/) — the original downloaded `.tar.gz` bundles.
- [`preprocessing-scripts/`](preprocessing-scripts/) — zsh scripts that turned raw text into 500-token snippet files.
- [`ml-scripts/`](ml-scripts/) — the original Python 2 BOW/SVM/clustering pipeline.

The 500-token snippet *data* these scripts produced is still actively used by
the current code — see [`data/snippets_2014/`](../data/README.md) — only the
old processing/modeling *scripts* live here.
