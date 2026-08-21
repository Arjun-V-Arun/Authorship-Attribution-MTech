# embeddings/ — cached frozen-transformer embeddings

`.npy` files, one per model, containing mean-pooled sentence embeddings for
every 500-token snippet in `data/raw`. Produced (and consumed) by
`src/dl_baseline.py`; consumed by `src/rigor_ablation.py` when present.

They're checked in so the rigor/ablation study runs instantly without
re-encoding, but they're fully regenerable:

```
python src/dl_baseline.py --model bert-base-multilingual-cased
python src/dl_baseline.py --model google/muril-base-cased
python src/dl_baseline.py --model google/muril-large-cased
python src/dl_baseline.py --model ai4bharat/indic-bert
```

Naming: `emb_<model-name-with-slashes-as-underscores>.npy`.
