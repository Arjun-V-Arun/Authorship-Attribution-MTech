#!/usr/bin/env python3
"""
collect_results.py — turn the six low-baselines files into one table.

After run_diagnostics.ps1 you have, in each corpus folder:
    corpus/prelim/low-baselines.txt          (raw text)
    corpus/prelim/low-baselines-clean.txt    (repeated paragraphs stripped)
    corpus/prelim/low-baselines-strict.txt   (+ author-naming paragraphs stripped)

This reads all of them and writes a single comparison, so you are not
transcribing numbers by hand into the paper.

Usage:
    python collect_results.py
    python collect_results.py --root C:\\Users\\CSE\\Documents\\Authorship-Attribution-MTech

Writes, next to the corpora:
    comparison.md    markdown table, paste straight into notes or a slide
    comparison.csv   same numbers, for the LaTeX table
"""

import argparse
import csv
import os
import re
import sys

# label in the file  ->  short row name for the table, in display order
ROWS = [
    ("char 3-5 gram, full text",            "Reference: char 3-5 gram"),
    ("char 3-5 gram, capped",               "Balanced works/author"),
    ("single-author tokens masked",         "Author-unique tokens masked"),
    ("rare tokens",                         "Rare tokens masked"),
    ("top-100 tokens only",                 "Top-100 tokens only"),
    ("top-300 tokens only",                 "Top-300 tokens only"),
    ("top-1000 tokens only",                "Top-1000 tokens only"),
    ("char (spans boundaries), intact",     "char, spans boundaries"),
    ("char (spans boundaries), shuffled",   "char, word order destroyed"),
    ("char_wb (within words only), intact", "char_wb, within words only"),
    ("first 100 words only",                "First 100 words"),
    ("first 250 words only",                "First 250 words"),
    ("first 500 words only",                "First 500 words"),
    ("first 1000 words only",               "First 1000 words"),
    ("first 2000 words only",               "First 2000 words"),
    ("capped 8/author + masked",            "COMBINED (honest)"),
]

VERSIONS = [("low-baselines.txt", "raw"),
            ("low-baselines-clean.txt", "clean"),
            ("low-baselines-strict.txt", "strict")]

SCORE = re.compile(r"macro-F1=([0-9.]+)")
PERM = re.compile(r"permutation mean macro-F1 = ([0-9.]+)")
CHANCE = re.compile(r"chance level[^=\u2248]*[\u2248=]\s*([0-9.]+)")
CORPUS = re.compile(r"corpus:\s*(\d+)\s*works,\s*(\d+)\s*authors")
MASSLINE = re.compile(r"those types are ([0-9.]+)% of all running text")


def parse(path):
    """-> {row_label: macro_f1}, plus a few scalars."""
    if not os.path.exists(path):
        return None
    out, meta = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = CORPUS.search(line)
            if m:
                meta["works"], meta["authors"] = int(m.group(1)), int(m.group(2))
            m = PERM.search(line)
            if m:
                meta["permutation"] = float(m.group(1))
            m = CHANCE.search(line)
            if m and "chance" not in meta:
                meta["chance"] = float(m.group(1))
            m = MASSLINE.search(line)
            if m:
                meta["masked_mass"] = float(m.group(1))

            sc = SCORE.search(line)
            if not sc:
                continue
            for key, label in ROWS:
                if key in line:
                    out.setdefault(label, float(sc.group(1)))
                    break
    return out, meta


def main(a):
    corpora = []
    for name in (a.corpora or ["sayahna-fiction", "sayahna-essays"]):
        d = os.path.join(a.root, name, "corpus", "prelim")
        if os.path.isdir(d):
            corpora.append((name, d))
        else:
            print(f"  (skipping {name}: {d} not found)")
    if not corpora:
        sys.exit("no corpus/prelim folders found — check --root")

    # collect
    data, meta, cols = {}, {}, []
    for cname, d in corpora:
        for fname, ver in VERSIONS:
            p = os.path.join(d, fname)
            got = parse(p)
            if got is None:
                continue
            scores, mt = got
            key = (cname.replace("sayahna-", ""), ver)
            data[key] = scores
            meta[key] = mt
            cols.append(key)

    if not cols:
        sys.exit("no low-baselines*.txt files found — run the diagnostics first")

    print(f"found {len(cols)} result files\n")

    # ---- markdown
    md = ["# Diagnostic comparison", ""]
    md.append("Macro-F1. `raw` = as downloaded, `clean` = repeated paragraphs "
              "(author bios, site furniture) removed, `strict` = also removed "
              "paragraphs naming the author.")
    md.append("")
    head = "| Condition | " + " | ".join(f"{c} {v}" for c, v in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    md += [head, sep]

    for _, label in ROWS:
        cells = []
        for k in cols:
            v = data[k].get(label)
            cells.append(f"{v:.3f}" if v is not None else "—")
        if all(c == "—" for c in cells):
            continue
        bold = "**" if label == "COMBINED (honest)" else ""
        md.append(f"| {bold}{label}{bold} | " + " | ".join(
            f"{bold}{c}{bold}" for c in cells) + " |")

    md.append("| _chance_ | " + " | ".join(
        f"_{meta[k].get('chance', float('nan')):.3f}_" for k in cols) + " |")
    md.append("| _permutation_ | " + " | ".join(
        f"_{meta[k].get('permutation', float('nan')):.3f}_" for k in cols) + " |")

    # ---- the deltas that matter
    md += ["", "## Effect of cleaning", ""]
    md.append("| Corpus | Condition | raw | clean | strict | raw to strict |")
    md.append("|---|---|---|---|---|---|")
    def fmt(x):
        return f"{x:.3f}" if x is not None else "—"

    for cname, _ in corpora:
        c = cname.replace("sayahna-", "")
        for label in ["Reference: char 3-5 gram", "Top-100 tokens only",
                      "First 500 words", "COMBINED (honest)"]:
            r = data.get((c, "raw"), {}).get(label)
            cl = data.get((c, "clean"), {}).get(label)
            st = data.get((c, "strict"), {}).get(label)
            if r is None:
                continue
            delta = f"{st - r:+.3f}" if st is not None else "—"
            md.append(f"| {c} | {label} | {fmt(r)} | {fmt(cl)} | {fmt(st)} | {delta} |")

    md += ["", "## Corpus sizes", ""]
    md.append("| Version | Works | Authors | Masked token mass |")
    md.append("|---|---|---|---|")
    for k in cols:
        m = meta[k]
        md.append(f"| {k[0]} {k[1]} | {m.get('works','—')} | {m.get('authors','—')} | "
                  f"{m.get('masked_mass','—')}% |")

    out_md = os.path.join(a.root, "comparison.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- csv
    out_csv = os.path.join(a.root, "comparison.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition"] + [f"{c}_{v}" for c, v in cols])
        for _, label in ROWS:
            row = [label] + [data[k].get(label, "") for k in cols]
            if any(x != "" for x in row[1:]):
                wr.writerow(row)
        wr.writerow(["chance"] + [meta[k].get("chance", "") for k in cols])

    print("\n".join(md))
    print(f"\nwritten: {out_md}")
    print(f"written: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="folder containing sayahna-fiction and sayahna-essays")
    ap.add_argument("--corpora", nargs="*",
                    help="corpus folder names (default: both sayahna-* folders)")
    main(ap.parse_args())
