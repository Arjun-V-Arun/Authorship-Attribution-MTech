# Vocabulary and morphology statistics

Text version: `corpus/texts`. Corpus statistics only — no models.

## Coverage: share of running tokens in the k most frequent word types

| k | fiction | essays |
|---|---|---|
| top 10 | 7.3% | 5.3% |
| top 50 | 13.8% | 10.3% |
| top 100 | 17.9% | 13.4% |
| top 300 | 25.7% | 19.8% |
| top 1,000 | 36.3% | 30.0% |
| top 3,000 | 46.6% | 39.6% |
| top 10,000 | 59.0% | 51.0% |

**`top 300` is the number that matters for Burrows' Delta**, which assumes the most frequent word forms cover most of a text. In an analytic language they largely do. Under heavy inflection each lemma splits across many forms, so the same 300 slots reach far less text.

## Vocabulary growth and word shape

| Statistic | fiction | essays |
|---|---|---|
| Tokens | 450,039 | 708,139 |
| Types | 154,568 | 261,350 |
| Type-token ratio | 0.3435 | 0.3691 |
| Hapax share of types | 77.0% | 77.3% |
| Heaps' beta | 0.861 | 0.883 |
| Mean word length (graphemes) | 4.15 | 4.62 |
| Median word length (graphemes) | 4 | 4 |

## How to read this

A low `top 300` coverage together with a high Heaps' beta and a high hapax share is the signature of heavy inflection: the vocabulary keeps growing because word forms, not lemmas, are being counted.

That single fact predicts both results in the baseline ladder:

- **Delta underperforms.** Its 300 most-frequent-word features reach only a small share of the text, so it discards most of the evidence.
- **Word features beat character features.** Each inflected form carries morphological information that character n-grams have to reconstruct piecemeal across the same span.

Run this on a fusional Indo-Aryan corpus (your Hindi set) with the same command. If Hindi shows higher top-300 coverage and a lower beta, the explanation holds across language families and the claim stops being about Malayalam alone.

