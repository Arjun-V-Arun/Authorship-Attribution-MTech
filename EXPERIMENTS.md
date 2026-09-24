# Malayalam AA Experimental Suite

These scripts are designed to sit beside the existing `src/` code in
`Authorship-Attribution-MTech`.

## What this adds

### 1. From-scratch neural models

`neural_models.py`

- Character/grapheme CNN
- BiLSTM
- BiGRU
- 5-fold whole-work stratified CV
- 3 random seeds by default
- vocabularies are fitted on the training fold only
- macro-F1 and accuracy
- optional checkpoints

The CharCNN operates on Unicode grapheme clusters, which is safer for
Malayalam than treating combining marks as independent characters.

### 2. Transformer fine-tuning

`transformer_malayalam.py`

Default:

`google/muril-base-cased`

Alternative:

`ai4bharat/IndicBERTv2-SS`

The important evaluation detail is that transformer windows are created
*after* the work-level train/validation split. Every window from a work
inherits the fold of that work. Validation predictions are averaged over
all windows of the work before calculating macro-F1.

This is deliberate. Randomly splitting transformer chunks would recreate
the leakage problem the project is explicitly designed to avoid.

### 3. Ablations

`ablations_malayalam.py`

Document length:

- 250
- 500
- 1000
- 2000
- 4000
- full document

Author-count scaling:

- 5
- 8
- 10
- 12
- 16
- 20 authors, subject to corpus availability

Both use the existing strong classical baselines:

- char 3-5 gram + LinearSVC
- word 1-2 gram + LinearSVC

## Recommended first run

Start with the neural models:

```powershell
python src/neural_models.py --root . --texts texts_strict `
  --epochs 8 --repeats 1 --batch-size 8
```

Then run the full 3-seed neural experiment:

```powershell
python src/neural_models.py --root . --texts texts_strict `
  --epochs 12 --repeats 3 --batch-size 8
```

Then transformer:

```powershell
python src/transformer_malayalam.py --root . `
  --model google/muril-base-cased `
  --epochs 3 --repeats 1 `
  --batch-size 2 --grad-accum 8
```

Then ablations:

```powershell
python src/ablations_malayalam.py --root . `
  --texts texts_strict --repeats 3 --author-repeats 5
```

Or everything:

```powershell
python src/run_research_suite.py --root . --stage all
```

## Outputs

```text
results/
└── malayalam/
    ├── neural/
    │   ├── neural_results.csv
    │   └── neural_summary.csv
    ├── transformer/
    │   └── transformer_results.csv
    └── ablations/
        ├── ablation_results.csv
        └── ablation_summary.csv
```

## Interpretation

Do not select the "winner" solely because it has the largest F1.

The experiments are intended to answer:

1. Do learned neural representations outperform the existing TF-IDF
   baselines?
2. Does a pretrained Indic transformer add anything beyond shallow
   lexical/subword representations?
3. How much text is necessary for reliable attribution?
4. How does performance change as the candidate-author set grows?
5. Do the same conclusions hold for fiction and essays?

A negative result is useful. If the CNN/BiLSTM/Transformer does not beat
the word 1-2 gram SVM, that is itself an important result for this corpus:
a much larger learned model is not automatically a better representation
of authorial style in this setting.
