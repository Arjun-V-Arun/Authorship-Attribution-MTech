# Diagnostic comparison

Macro-F1. `raw` = as downloaded, `clean` = repeated paragraphs (author bios, site furniture) removed, `strict` = also removed paragraphs naming the author.

| Condition | fiction raw | fiction clean | fiction strict | essays raw | essays clean | essays strict |
|---|---|---|---|---|---|---|
| Reference: char 3-5 gram | 0.963 | 0.893 | 0.893 | 0.978 | 0.945 | 0.945 |
| Balanced works/author | 0.974 | 0.958 | 0.958 | 0.949 | 0.883 | 0.875 |
| Author-unique tokens masked | 0.927 | 0.851 | 0.851 | 0.978 | 0.918 | 0.915 |
| Rare tokens masked | 0.962 | 0.947 | 0.947 | 0.978 | 0.951 | 0.954 |
| Top-100 tokens only | 0.780 | 0.762 | 0.762 | 0.904 | 0.802 | 0.773 |
| Top-300 tokens only | 0.867 | 0.851 | 0.843 | 0.953 | 0.896 | 0.891 |
| Top-1000 tokens only | 0.914 | 0.869 | 0.872 | 0.990 | 0.959 | 0.943 |
| char, spans boundaries | 0.963 | 0.887 | 0.888 | 0.966 | 0.930 | 0.930 |
| char, word order destroyed | 0.963 | 0.882 | 0.882 | 0.966 | 0.928 | 0.911 |
| char_wb, within words only | 0.963 | 0.893 | 0.893 | 0.978 | 0.945 | 0.945 |
| First 100 words | 0.397 | 0.389 | 0.407 | 0.584 | 0.579 | 0.578 |
| First 250 words | 0.656 | 0.656 | 0.674 | 0.762 | 0.751 | 0.749 |
| First 500 words | 0.767 | 0.770 | 0.756 | 0.813 | 0.797 | 0.804 |
| First 1000 words | 0.916 | 0.884 | 0.889 | 0.918 | 0.854 | 0.877 |
| First 2000 words | 0.957 | 0.950 | 0.950 | 0.978 | 0.948 | 0.933 |
| **COMBINED (honest)** | **0.752** | **0.752** | **0.752** | **0.776** | **0.788** | **0.776** |
| _chance_ | _0.083_ | _0.083_ | _0.083_ | _0.059_ | _0.059_ | _0.059_ |
| _permutation_ | _0.073_ | _0.075_ | _0.071_ | _0.056_ | _0.056_ | _0.053_ |

## Effect of cleaning

| Corpus | Condition | raw | clean | strict | raw to strict |
|---|---|---|---|---|---|
| fiction | Reference: char 3-5 gram | 0.963 | 0.893 | 0.893 | -0.070 |
| fiction | Top-100 tokens only | 0.780 | 0.762 | 0.762 | -0.018 |
| fiction | First 500 words | 0.767 | 0.770 | 0.756 | -0.011 |
| fiction | COMBINED (honest) | 0.752 | 0.752 | 0.752 | +0.000 |
| essays | Reference: char 3-5 gram | 0.978 | 0.945 | 0.945 | -0.033 |
| essays | Top-100 tokens only | 0.904 | 0.802 | 0.773 | -0.131 |
| essays | First 500 words | 0.813 | 0.797 | 0.804 | -0.009 |
| essays | COMBINED (honest) | 0.776 | 0.788 | 0.776 | +0.000 |

## Corpus sizes

| Version | Works | Authors | Masked token mass |
|---|---|---|---|
| fiction raw | 114 | 12 | 39.5% |
| fiction clean | 114 | 12 | 39.4% |
| fiction strict | 114 | 12 | 39.4% |
| essays raw | 610 | 17 | 39.7% |
| essays clean | 609 | 17 | 39.9% |
| essays strict | 607 | 17 | 40.0% |
