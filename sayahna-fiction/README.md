# sayahna-fiction/ — Malayalam fiction corpus

114 works, 12 authors, crawled from the Sayahna Foundation digital library
(books.sayahna.org) via [`src/sayahna_crawler.py`](../src/sayahna_crawler.py).
Part of the Malayalam benchmark — see
[`docs/project-summary.md`](../docs/project-summary.md).

## `corpus/`

| Path | What it is |
|---|---|
| `xml/<id>.xml` | Raw TEI source, cached (re-crawling is free/idempotent). |
| `texts/<id>.txt` | The work only — author bio and site furniture excluded at parse time. **Raw** cleaning level. |
| `texts_clean/<id>.txt` | `texts/` with repeated boilerplate paragraphs (site furniture, stray bio fragments) stripped by `src/find_boilerplate_v2.py --strip`. |
| `texts_strict/<id>.txt` | `texts_clean/` with paragraphs that name the author also stripped (`--strip --strip-named`). This is the text version most experiments should use. |
| `metadata.csv` | One row per work, every TEI header field. |
| `prelim/` | Diagnostic and baseline-ladder output for this corpus specifically (see [`src/README.md`](../src/README.md) for what produces each file). |

Note: unlike `sayahna-essays/`, this corpus has no `pages/`, `bios/`, or
`authors.csv` — it predates those additions to the crawler. Not regenerated
here since doing so is a corpus-versioning decision, not a cleanup one; see
the note in the root cleanup report.

## Reproducing

```
cd sayahna-fiction
python ../src/sayahna_crawler.py --indexes shortstory
python ../src/find_boilerplate_v2.py --strip --strip-named
cd ..
python src/baseline_ladder-v2.py --root . --texts texts_strict --corpora sayahna-fiction
```

See [`run_diagnostics.ps1`](../run_diagnostics.ps1) at the repo root for the
full unattended diagnostic sequence across both corpora.
