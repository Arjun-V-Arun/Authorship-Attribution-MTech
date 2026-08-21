# ml-scripts/ — original 2014 modeling pipeline (Python 2)

The original CS365 2014 BOW/clustering/SVM pipeline. Written for Python 2
(`print` statements, `nltk`) — does not run under a modern Python 3
interpreter without porting. Kept for reference/comparison only; the current,
maintained equivalent is [`src/baseline.py`](../../src/baseline.py) (word/char
TF-IDF + LinearSVC/LogReg/RandomForest) and [`src/book_disjoint.py`](../../src/book_disjoint.py)
(honest, leakage-free evaluation).

- `BOW-uni.py`, `BOW-bi.py`, `BOW-tri.py` — unigram/bigram/trigram
  bag-of-words feature construction.
- `cluster.py` — k-means clustering (4 clusters) on a feature-vector array.
- `count.py` — precision/recall statistics for a clusters array vs. vectors array.
- `pca.py` — PCA dimensionality reduction on a vector array.
- `plot.py` — plots the first three PCA components.
- `svm.py` — generic SVM on a feature matrix `X` and labels `y`.
- `dharamvir_svm.py`, `prem_svm.py`, `sarat_svm.py`, `vibhuti_svm.py` — per-author one-vs-all SVM runs.
- `unsupervised.py` — runs the unsupervised clustering pipeline end-to-end.
- `corpus.txt` — a large (~12MB) concatenation of the raw corpus, used as input to some of the scripts above.

### What the original approach did

Supervised: build a TF-IDF-style feature matrix + label vector on the
training set, fit an SVM, evaluate on a held-out test set.

Unsupervised (bigram/trigram): strip punctuation, cut into 500-word snippets,
count n-grams to find the top 2000, build a bag-of-words per snippet using
those n-grams, PCA down to 20 dimensions, k-means with 4 clusters, then score
against ground truth with precision/recall.

The evaluation in both cases shuffled snippets across train/test without
regard to which book they came from — the topic/book leakage that
[`src/book_disjoint.py`](../../src/book_disjoint.py) identifies and corrects.
