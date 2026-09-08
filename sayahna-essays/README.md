# sayahna-essays/ — Malayalam essay corpus

599-610 works (varies slightly by cleaning level, see
[`results/malayalam/comparison.md`](../results/malayalam/comparison.md)),
16-17 authors, crawled from the Sayahna Foundation digital library
(books.sayahna.org) via [`src/sayahna_crawler.py`](../src/sayahna_crawler.py).
Part of the Malayalam benchmark — see
[`docs/project-summary.md`](../docs/project-summary.md).

## `corpus/`

| Path | What it is |
|---|---|
| `xml/<id>.xml` | Raw TEI source, cached (re-crawling is free/idempotent). |
| `pages/` | Cached raw HTML of the Sayahna index/author-collection pages the crawler parsed. |
| `texts/<id>.txt` | The work only — author bio and site furniture excluded at parse time. **Raw** cleaning level. |
| `texts_clean/<id>.txt` | `texts/` with repeated boilerplate paragraphs (site furniture, stray bio fragments) stripped by `src/find_boilerplate_v2.py --strip`. |
| `texts_strict/<id>.txt` | `texts_clean/` with paragraphs that name the author also stripped (`--strip --strip-named`). This is the text version most experiments should use. |
| `metadata.csv` | One row per work, every TEI header field. |
| `authors.csv` | One row per author, with their biography once. |
| `prelim/` | Diagnostic and baseline-ladder output for this corpus specifically (see [`src/README.md`](../src/README.md) for what produces each file). |

## Reproducing

```
cd sayahna-essays
python ../src/sayahna_crawler.py --indexes article
python ../src/find_boilerplate_v2.py --strip --strip-named
cd ..
python src/baseline_ladder-v2.py --root . --texts texts_strict --corpora sayahna-essays
```

See [`run_diagnostics.ps1`](../run_diagnostics.ps1) at the repo root for the
full unattended diagnostic sequence across both corpora.
