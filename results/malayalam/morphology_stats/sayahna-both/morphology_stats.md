# Vocabulary and morphology statistics

Text version: `corpus/texts_strict`. Corpus statistics only — no models.

## Coverage: share of running tokens in the k most frequent word types

| k | fiction | essays |
|---|---|---|
| top 10 | 7.4% | 5.4% |
| top 50 | 14.0% | 10.4% |
| top 100 | 18.1% | 13.4% |
| top 300 | 26.0% | 19.5% |
| top 1,000 | 36.6% | 28.0% |
| top 3,000 | 46.8% | 37.1% |
| top 10,000 | 58.8% | 48.7% |

**`top 300` is the number that matters for Burrows' Delta**, which assumes the most frequent word forms cover most of a text. In an analytic language they largely do. Under heavy inflection each lemma splits across many forms, so the same 300 slots reach far less text.

## Vocabulary growth and word shape

| Statistic | fiction | essays |
|---|---|---|
| Tokens | 435,664 | 657,952 |
| Types | 152,290 | 257,430 |
| Type-token ratio | 0.3496 | 0.3913 |
| Hapax share of types | 77.3% | 77.4% |
| Heaps' beta | 0.865 | 0.890 |
| Mean word length (graphemes) | 4.15 | 4.63 |
| Median word length (graphemes) | 4 | 4 |

## How to read this

A low `top 300` coverage together with a high Heaps' beta and a high hapax share is the signature of heavy inflection: the vocabulary keeps growing because word forms, not lemmas, are being counted.

That single fact predicts both results in the baseline ladder:

- **Delta underperforms.** Its 300 most-frequent-word features reach only a small share of the text, so it discards most of the evidence.
- **Word features beat character features.** Each inflected form carries morphological information that character n-grams have to reconstruct piecemeal across the same span.

Run this on a fusional Indo-Aryan corpus (your Hindi set) with the same command. If Hindi shows higher top-300 coverage and a lower beta, the explanation holds across language families and the claim stops being about Malayalam alone.

