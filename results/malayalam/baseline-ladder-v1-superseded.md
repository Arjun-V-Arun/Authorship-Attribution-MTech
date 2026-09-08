# Baseline ladder

Text version: `corpus/texts_strict`. Macro-F1, stratified k-fold CV.

| Model | fiction | essays | Hypothesis |
|---|---|---|---|
| FLOOR  stratified random | 0.100 | 0.051 | what guessing gets |
| L1  Burrows' Delta (300 MFW) | 0.784 | 0.633 | does English-style function-word stylometry transfer? |
| L1  char 3-5 gram + SVM | 0.914 | 0.905 | the reference; every earlier number came from this |
| L1  char 2-4 gram + SVM | 0.924 | 0.906 | shorter n-grams: more morphology, less lexis |
| L1  char 4-6 gram + SVM | 0.921 | 0.907 | longer n-grams: closer to whole words |
| L1  word unigram + SVM | 0.905 | 0.928 | lexical choice alone: which words, order ignored |
| L1  word bigram + SVM | 0.875 | 0.824 | phrasing alone: adjacent word pairs, no unigrams |
| L1  word 1-2 gram + SVM | 0.912 | 0.935 | both together: does the bigram add anything over unigrams? |
| L1  stylometric + LogReg | 0.794 | 0.781 | can ~30 readable features do it? |
| L2  char 3-5 + LogReg | 0.875 | 0.906 | is the SVM or the features doing the work? |
| L2  char 3-5 + ComplementNB | 0.449 | 0.397 | generative comparison; Complement not Multinomial, which cannot weight classes and collapses onto the large ones |
| L2  char 3-5 + NearestCentroid | 0.751 | 0.850 | distance-based, the modern analogue of Delta |
| _chance_ | _0.083_ | _0.062_ | |

| Corpus | Works | Authors |
|---|---|---|
| fiction | 114 | 12 |
| essays | 599 | 16 |
