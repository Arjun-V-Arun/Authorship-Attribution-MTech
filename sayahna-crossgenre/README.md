# sayahna-crossgenre/ — merged corpus for the cross-genre experiment

Generated output of [`src/merge_and_crossgenre.py`](../src/merge_and_crossgenre.py)
— deduplicates and merges [`sayahna-fiction/`](../sayahna-fiction/) and
[`sayahna-essays/`](../sayahna-essays/), then checks which authors have
enough works in both genres to train on one and test on the other.

**Status: blocked.** After merging and deduplicating (728 → 673 works), only
one author (Vallathol Vasudevamenon) has >=4 works in both genres — not
enough to run the experiment. See
[`docs/project-summary.md`](../docs/project-summary.md) and
[`docs/assessment-and-roadmap.md`](../docs/assessment-and-roadmap.md#c-cross-topic-experiment--the-domain-shift-result-you-can-actually-run)
for the cross-*topic* replacement experiment that doesn't have this problem.

| Path | What it is |
|---|---|
| `metadata.csv` | The combined, deduplicated corpus metadata. |
| `texts/` | One text file per merged work. |
| `crossgenre.txt` | The experiment's output report (author overlap counts and, once unblocked, the within/cross-genre scores). |

Regenerate with `python src/merge_and_crossgenre.py --texts texts_strict --min-per-genre 4`.
