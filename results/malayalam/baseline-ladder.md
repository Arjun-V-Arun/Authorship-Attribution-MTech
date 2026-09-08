# Baseline ladder

Text version: `corpus/texts_strict`. Macro-F1, stratified k-fold CV.

| Model | fiction | essays | Hypothesis |
|---|---|---|---|
| FLOOR  stratified random | 0.107 | 0.069 | what guessing gets |
| L1  Burrows' Delta (300 MFW) | 0.752 | 0.656 | does English-style function-word stylometry transfer? |
| L1  char 3-5 gram + SVM | 0.888 | 0.919 | the reference; every earlier number came from this |
| L1  char 2-4 gram + SVM | 0.902 | 0.915 | shorter n-grams: more morphology, less lexis |
| L1  char 4-6 gram + SVM | 0.898 | 0.938 | longer n-grams: closer to whole words |
| L1  word unigram + SVM | 0.925 | 0.939 | lexical choice alone: which words, order ignored |
| L1  word bigram + SVM | 0.883 | 0.831 | phrasing alone: adjacent word pairs, no unigrams |
| L1  word 1-2 gram + SVM | 0.928 | 0.953 | both together: does the bigram add anything over unigrams? |
| L1  stylometric + LogReg | 0.808 | 0.784 | can ~30 readable features do it? |
| L2  char 3-5 + LogReg | 0.867 | 0.911 | is the SVM or the features doing the work? |
| L2  char 3-5 + ComplementNB | 0.419 | 0.389 | generative comparison; Complement not Multinomial, which cannot weight classes and collapses onto the large ones |
| L2  char 3-5 + NearestCentroid | 0.751 | 0.842 | distance-based, the modern analogue of Delta |
| _chance_ | _0.083_ | _0.062_ | |

| Corpus | Works | Authors |
|---|---|---|
| fiction | 114 | 12 |
| essays | 599 | 16 |
