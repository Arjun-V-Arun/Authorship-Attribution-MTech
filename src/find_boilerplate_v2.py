#!/usr/bin/env python3
"""
find_boilerplate.py — find text that repeats across an author's works.

WHY THIS EXISTS
Sayahna attaches an author note to its works. If that note sits inside
<text><body>, the extractor picks it up, and every work by an author then
carries the same paragraph naming that author, their birth year, their
awards and their home town. A classifier reading that is not doing
authorship attribution; it is reading a label that was left in the data.

The diagnostic feature lists made this visible: authors' own names,
"(2019)", "ലഭിച്ചിട്ടുണ്ടു്" (has received), "ജീവചരിത്രം" (biography).

WHAT IT DOES
    1. splits every work into paragraphs
    2. finds paragraphs that repeat across an author's works (a bio blurb)
    3. finds paragraphs that repeat across authors (site boilerplate)
    4. flags paragraphs containing the author's own name
    5. with --strip, writes cleaned texts to corpus/texts_clean/

Usage:
    python find_boilerplate.py                  # report only, changes nothing
    python find_boilerplate.py --strip          # write corpus/texts_clean/
    python find_boilerplate.py --min-share 0.4  # tune the repeat threshold

After --strip, re-run the baselines pointing TEXTS at corpus/texts_clean.
"""

import argparse
import os
import re
import sys
import glob
import shutil
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

TEXTS = "corpus/texts"
CLEAN = "corpus/texts_clean"
STRICT = "corpus/texts_strict"
REPORT = os.path.join("corpus", "prelim", "boilerplate-report.txt")

OUT = []


def w(s=""):
    print(s)
    OUT.append(s)


def norm(p):
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", p)).strip()


MIN_NAME_LEN = 4      # shorter fragments match inside ordinary words


def name_variants(author, author_ml=""):
    """
    Name tokens distinctive enough to search for. Short fragments are dropped:
    a 2-character Malayalam string occurs inside dozens of unrelated words and
    would flag every paragraph in the corpus.
    """
    out = []
    for src in (str(author), str(author_ml)):
        for p in re.split(r"[\s.,]+", src):
            p = p.strip(" .,\u200c\u200d")
            if len(p) >= MIN_NAME_LEN:
                out.append(p)
    return sorted(set(out), key=len, reverse=True)


def load():
    files = sorted(glob.glob("corpus/metadata*.csv"))
    if not files:
        sys.exit("no corpus/metadata*.csv here")
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files],
                   ignore_index=True).fillna("")
    keep = []
    for _, r in df.iterrows():
        p = os.path.join(TEXTS, str(r.work_id) + ".txt")
        if os.path.exists(p):
            keep.append((r.work_id, r.author, r.get("author_ml", ""),
                         open(p, encoding="utf-8").read()))
    return pd.DataFrame(keep, columns=["work_id", "author", "author_ml", "text"])


def main(a):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    df = load()
    w("=" * 76)
    w("BOILERPLATE AND AUTHOR-BIOGRAPHY DETECTION")
    w("=" * 76)
    w(f"{len(df)} works, {df.author.nunique()} authors")
    w("")

    paras = {}                      # work_id -> [normalised paragraphs]
    for _, r in df.iterrows():
        paras[r.work_id] = [norm(p) for p in r.text.split("\n\n") if norm(p)]

    # ---- paragraphs repeating across the SAME author's works -> bio blurb
    w("-" * 76)
    w("1. REPEATED WITHIN AN AUTHOR  (this is where a bio note shows up)")
    w("-" * 76)
    doomed = set()
    per_author_hits = {}
    for author, grp in df.groupby("author"):
        ids = list(grp.work_id)
        if len(ids) < 2:
            continue
        c = Counter()
        for i in ids:
            c.update(set(paras[i]))
        hits = [(p, n) for p, n in c.items()
                if n >= max(2, a.min_share * len(ids)) and len(p) > 40]
        if not hits:
            continue
        hits.sort(key=lambda x: -x[1])
        per_author_hits[author] = hits
        w(f"\n  {author}   ({len(ids)} works)")
        for p, n in hits[:a.show]:
            doomed.add(p)
            w(f"    in {n}/{len(ids)} works, {len(p.split())} words:")
            w(f"      {p[:150]}{'…' if len(p) > 150 else ''}")
        if len(hits) > a.show:
            for p, _ in hits[a.show:]:
                doomed.add(p)
            w(f"    ... and {len(hits)-a.show} more repeated paragraphs")
    if not per_author_hits:
        w("  none found — no author-level repeated paragraphs")

    # ---- paragraphs repeating ACROSS authors -> site furniture
    w("")
    w("-" * 76)
    w("2. REPEATED ACROSS AUTHORS  (site furniture, licence notes)")
    w("-" * 76)
    owners = defaultdict(set)
    for _, r in df.iterrows():
        for p in set(paras[r.work_id]):
            owners[p].add(r.author)
    cross = [(p, len(v)) for p, v in owners.items() if len(v) >= 2 and len(p) > 40]
    cross.sort(key=lambda x: -x[1])
    if cross:
        for p, n in cross[:a.show]:
            doomed.add(p)
            w(f"  in {n} authors' works: {p[:130]}{'…' if len(p) > 130 else ''}")
        if len(cross) > a.show:
            w(f"  ... and {len(cross)-a.show} more")
    else:
        w("  none found")

    # ---- paragraphs naming their own author
    w("")
    w("-" * 76)
    w("3. PARAGRAPHS CONTAINING THE AUTHOR'S OWN NAME")
    w("-" * 76)
    named = 0
    by_author = Counter()
    named_paras = defaultdict(set)          # work_id -> paragraphs naming the author
    for _, r in df.iterrows():
        vs = name_variants(r.author, r.author_ml)
        hit = False
        for p in paras[r.work_id]:
            if any(v and v in p for v in vs):
                named_paras[r.work_id].add(p)
                if not hit:
                    named += 1
                    by_author[r.author] += 1
                    if named <= a.show:
                        w(f"  [{r.author}] {p[:130]}{'…' if len(p) > 130 else ''}")
                hit = True
    w(f"\n  {named} of {len(df)} works contain a paragraph naming their own author")
    if by_author:
        w("  worst offenders: " + ", ".join(
            f"{k} ({v})" for k, v in by_author.most_common(6)))
    w("  These are NOT removed by --strip, because each one differs (a different")
    w("  subject work is named each time) so the repeated-paragraph filter misses")
    w("  them. Use --strip-named to remove them into corpus/texts_strict/.")
    w("  In fiction a character may share the author's surname, so treat the")
    w("  strict version as an upper bound on cleaning, not as the ground truth.")

    # ---- impact
    w("")
    w("-" * 76)
    w("4. IMPACT IF STRIPPED")
    w("-" * 76)
    tot_before = sum(len(" ".join(v).split()) for v in paras.values())
    removed_words = 0
    affected = 0
    for wid, ps in paras.items():
        kept = [p for p in ps if p not in doomed]
        if len(kept) != len(ps):
            affected += 1
        removed_words += len(" ".join(p for p in ps if p in doomed).split())
    w(f"  paragraphs flagged      : {len(doomed):,}")
    w(f"  works affected          : {affected} of {len(df)}")
    w(f"  words removed           : {removed_words:,} of {tot_before:,} "
      f"({removed_words/max(1,tot_before):.1%})")

    strict_removed = sum(len(v) for v in named_paras.values())
    w(f"  additionally naming the author  : {strict_removed:,} paragraph instances "
      f"(--strip-named only)")

    def write_dir(path, extra):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        empties = []
        for wid, ps in paras.items():
            drop = doomed | (named_paras.get(wid, set()) if extra else set())
            kept = [p for p in ps if p not in drop]
            if not kept:
                empties.append(wid)
            open(os.path.join(path, str(wid) + ".txt"), "w",
                 encoding="utf-8").write("\n\n".join(kept))
        w(f"  written to {path}/")
        if empties:
            w(f"  !! {len(empties)} works are now EMPTY: {', '.join(map(str, empties[:8]))}")
        return empties

    if a.strip or a.strip_named:
        w("")
        if a.strip or a.strip_named:
            write_dir(CLEAN, False)
        if a.strip_named:
            write_dir(STRICT, True)
        w("")
        w("  Now run the three-way comparison:")
        w(f"    python low_baselines.py --texts {TEXTS}")
        w(f"    python low_baselines.py --texts {CLEAN}")
        if a.strip_named:
            w(f"    python low_baselines.py --texts {STRICT}")
        w("  The differences between those runs are the result.")
    else:
        w("")
        w("  report only — nothing written. Use --strip / --strip-named to apply.")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\nwritten: {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", action="store_true", help="write corpus/texts_clean/")
    ap.add_argument("--strip-named", action="store_true",
                    help="also write corpus/texts_strict/ with author-naming paragraphs removed")
    ap.add_argument("--min-share", type=float, default=0.4,
                    help="paragraph must appear in this share of an author's works")
    ap.add_argument("--show", type=int, default=6, help="examples to print per section")
    main(ap.parse_args())
