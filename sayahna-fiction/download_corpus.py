#!/usr/bin/env python3
"""
download_corpus.py — download the whole Sayahna corpus listed in sayahna-fiction.txt.

One work = one document. Nothing is filtered, split, or excluded: short
stories, novels, essays, poems and drama all come down. Sorting happens later.

    https://books.sayahna.org/html/<slug>.html
 -> https://books.sayahna.org/xml/<slug>.xml          (directory AND extension change)

Usage:
    pip install requests
    python download_corpus.py                       # reads sayahna-fiction.txt
    python download_corpus.py mylist.txt
    python download_corpus.py --test sample.xml     # parse one local file, no network

Output:
    corpus/xml/<slug>.xml        cached TEI, so re-runs cost nothing
    corpus/texts/<slug>.txt      the work's text, plain UTF-8
    corpus/metadata.csv          one row per work, every field the TEI carries
"""

import csv
import os
import re
import sys
import time
import hashlib
import unicodedata
import xml.etree.ElementTree as ET

import requests

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

IN_FILE = "sayahna-fiction.txt"
OUT = "corpus"
XMLDIR = os.path.join(OUT, "xml")
TEXTDIR = os.path.join(OUT, "texts")
META = os.path.join(OUT, "metadata.csv")

SLEEP = 1.2
TIMEOUT = 60
UA = "MalayalamAA-research-collector/1.0 (M.Tech corpus study)"

# Your unified author names. The Malayalam string in column 1 of the input
# file is the key; whatever the TEI header says is kept separately as
# author_name_tei so you can see where the two disagree.
AUTHOR_NAMES = {
    "ഹരികുമാർ ഇ": "E. Harikumar",
    "സുകുമാരൻ കെ": "K. Sukumaran",
    "സാബു ഹരിഹരൻ": "Sabu Hariharan",
    "സച്ചിദാനന്ദൻ": "K. Satchidanandan",
    "സന്തോഷ് കുമാർ സി": "C. Santhosh Kumar",
    "സക്കറിയ": "Paul Zacharia",
    "വേണുഗോപൻ നായർ എസ് വി": "S. V. Venugopan Nair",
    "വള്ളത്തോൾ വാസുദേവ മേനോൻ ബിഎ": "Vallathol Vasudevamenon B. A.",
    "ടി എ രാജലക്ഷ്മി": "T. A. Rajalakshmi",
    "വി കെ കെ രമേഷ്": "V. K. K. Ramesh",
    "ബിനീഷ് പിലാശ്ശേരി": "Bineesh Pilasseri",
    "ബാബുരാജ് കെ ടി": "K. T. Baburaj",
    "അയ്മനം ജോൺ": "Aymanam John",
}

MAL = re.compile(r"[\u0D00-\u0D7F]")
ATOMIC_CHILLU = re.compile(r"[\u0D7A-\u0D7F]")
ZWJ_CHILLU = re.compile(r"[\u0D15-\u0D39]\u0D4D\u200D")

COLUMNS = [
    # identity
    "work_id", "author", "author_ml", "author_name_tei",
    "title_ml", "title_en", "section_label",
    # what kind of text this is
    "work_type", "keywords", "domain", "derivation", "factuality",
    "constitution", "language",
    # provenance
    "source_url", "html_url", "publisher", "pub_place",
    "digital_edition_date", "digital_edition_year",
    "source_pub_year", "source_publisher", "source_pages",
    "typeset_by", "edited_by", "encoded_by", "sponsor", "funder",
    "licence", "licence_url", "setting_place", "setting_time",
    # measurements
    "word_count", "char_count", "para_count",
    "chillu_atomic", "chillu_zwj", "chillu_encoding", "unicode_form",
    "sha256",
]

S = requests.Session()
S.headers.update({"User-Agent": UA})


# ------------------------------------------------------------------ helpers

def slug_of(url):
    return re.sub(r"\.(xml|html?|pdf)$", "", url.rstrip("/").split("/")[-1], flags=re.I)


def to_xml_url(url):
    """/html/foo.html -> /xml/foo.xml   (both the directory and the extension)."""
    u = re.sub(r"/(?:ml/)?(?:html|pdf)/", "/xml/", url)
    return re.sub(r"\.(?:html?|pdf)$", ".xml", u)


def txt(el):
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def mal_ratio(s):
    t = re.sub(r"\s", "", s)
    return len(MAL.findall(t)) / len(t) if t else 0.0


# ------------------------------------------------------------- TEI metadata

def parse_metadata(root):
    """Pull every useful field out of the teiHeader."""
    h = root.find(f"{TEI}teiHeader")
    m = {c: "" for c in COLUMNS}
    if h is None:
        return m

    fd = h.find(f"{TEI}fileDesc")
    ts = fd.find(f"{TEI}titleStmt") if fd is not None else None

    if ts is not None:
        for t in ts.iter(f"{TEI}title"):
            lang = t.get(XML_LANG)
            if lang == "ml" and not m["title_ml"]:
                m["title_ml"] = txt(t)
            elif lang == "en" and not m["title_en"]:
                m["title_en"] = txt(t)
        m["author_name_tei"] = txt(ts.find(f"{TEI}author"))
        m["sponsor"] = txt(ts.find(f"{TEI}sponsor"))
        m["funder"] = txt(ts.find(f"{TEI}funder"))
        for rs in ts.findall(f"{TEI}respStmt"):
            resp = txt(rs.find(f"{TEI}resp")).lower()
            who = txt(rs.find(f"{TEI}name"))
            if "typeset" in resp:
                m["typeset_by"] = who
            elif "edit" in resp:
                m["edited_by"] = who
            elif "encod" in resp:
                m["encoded_by"] = who

    ps = fd.find(f"{TEI}publicationStmt") if fd is not None else None
    if ps is not None:
        m["publisher"] = txt(ps.find(f"{TEI}publisher"))
        m["pub_place"] = txt(ps.find(f"{TEI}pubPlace"))
        d = ps.find(f"{TEI}date")
        if d is not None:
            m["digital_edition_date"] = txt(d)
            m["digital_edition_year"] = (d.get("when") or "")[:4] or \
                (re.search(r"\b(19|20)\d\d\b", txt(d)) or [""])[0]
        av = ps.find(f"{TEI}availability")
        if av is not None:
            m["licence"] = txt(av)
            ref = av.find(f".//{TEI}ref")
            if ref is not None:
                m["licence_url"] = ref.get("target", "")

    # the printed edition this was digitised from
    sd = fd.find(f"{TEI}sourceDesc") if fd is not None else None
    if sd is not None:
        bf = sd.find(f"{TEI}biblFull")
        scope = bf if bf is not None else sd
        sp = scope.find(f"{TEI}publicationStmt")
        if sp is not None:
            m["source_publisher"] = txt(sp.find(f"{TEI}publisher"))
            sdate = txt(sp.find(f"{TEI}date"))
            y = re.search(r"\b(1[6-9]\d\d|20[0-2]\d)\b", sdate)
            m["source_pub_year"] = y.group(0) if y else ""
        meas = scope.find(f".//{TEI}measure")
        if meas is not None:
            m["source_pages"] = meas.get("quantity", "") or txt(meas)

    pd_ = h.find(f"{TEI}profileDesc")
    if pd_ is not None:
        lang = pd_.find(f".//{TEI}language")
        if lang is not None:
            m["language"] = lang.get("ident", "") or txt(lang)
        terms = [txt(t) for t in pd_.iter(f"{TEI}term")]
        m["keywords"] = "; ".join(terms)

        td = pd_.find(f".//{TEI}textDesc")
        if td is not None:
            m["work_type"] = (td.get("n") or "").strip().lower()
            for tag, col in [("domain", "domain"), ("derivation", "derivation"),
                             ("factuality", "factuality"),
                             ("constitution", "constitution")]:
                el = td.find(f"{TEI}{tag}")
                if el is not None:
                    m[col] = el.get("type", "")
        st = pd_.find(f".//{TEI}setting")
        if st is not None:
            m["setting_place"] = txt(st.find(f"{TEI}name")).strip(" ,")
            m["setting_time"] = txt(st.find(f"{TEI}time"))

    # fall back to the first keyword if textDesc had no @n
    if not m["work_type"] and m["keywords"]:
        m["work_type"] = m["keywords"].split(";")[0].strip().lower()

    return m


def extract_text(root):
    """All paragraph text in <text><body>. <front> and <back> are skipped."""
    body = root.find(f"{TEI}text/{TEI}body")
    if body is None:
        return "", 0
    paras, prev = [], None
    for p in body.iter(f"{TEI}p"):
        t = re.sub(r"\s+", " ", "".join(p.itertext())).strip()
        if len(t) < 5 or mal_ratio(t) < 0.30:
            continue
        if t != prev:
            paras.append(t)
        prev = t
    return "\n\n".join(paras), len(paras)


# --------------------------------------------------------------- downloading

def fetch(url, cache):
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        return open(cache, encoding="utf-8").read(), True
    r = S.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return None, False
    r.encoding = "utf-8"
    open(cache, "w", encoding="utf-8").write(r.text)
    time.sleep(SLEEP)
    return r.text, False


def read_list(path):
    rows, seen = [], set()
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 3:
            print(f"  line {n}: malformed, skipped")
            continue
        author_ml, label, url = parts[0], parts[1], parts[2]
        if url in seen:
            print(f"  duplicate in list, skipped: {slug_of(url)}")
            continue
        seen.add(url)
        rows.append((author_ml, label, url))
    return rows


def main(list_file):
    for d in (XMLDIR, TEXTDIR):
        os.makedirs(d, exist_ok=True)

    works = read_list(list_file)
    print(f"\n{len(works)} works to fetch\n")

    rows, failed, cached = [], [], 0
    for i, (author_ml, label, html_url) in enumerate(works, 1):
        slug = slug_of(html_url)
        xml_url = to_xml_url(html_url)
        try:
            raw, from_cache = fetch(xml_url, os.path.join(XMLDIR, slug + ".xml"))
        except Exception as e:
            failed.append((slug, str(e)[:60]))
            print(f"[{i:>3}/{len(works)}] {slug:<34} ERROR {str(e)[:40]}")
            continue
        if not raw:
            failed.append((slug, "404 — no XML at that URL"))
            print(f"[{i:>3}/{len(works)}] {slug:<34} 404")
            continue
        cached += from_cache

        try:
            root = ET.fromstring(raw.encode("utf-8"))
        except ET.ParseError as e:
            failed.append((slug, f"bad XML: {e}"))
            print(f"[{i:>3}/{len(works)}] {slug:<34} BAD XML")
            continue

        m = parse_metadata(root)
        text, nparas = extract_text(root)
        if not text.strip():
            failed.append((slug, "empty body"))
            print(f"[{i:>3}/{len(works)}] {slug:<34} EMPTY")
            continue

        open(os.path.join(TEXTDIR, slug + ".txt"), "w", encoding="utf-8").write(text)

        atomic = len(ATOMIC_CHILLU.findall(text))
        zwj = len(ZWJ_CHILLU.findall(text))
        m.update({
            "work_id": slug,
            "author": AUTHOR_NAMES.get(author_ml, author_ml),
            "author_ml": author_ml,
            "section_label": label,
            "source_url": xml_url,
            "html_url": html_url,
            "word_count": len(text.split()),
            "char_count": len(text),
            "para_count": nparas,
            "chillu_atomic": atomic,
            "chillu_zwj": zwj,
            "chillu_encoding": ("mixed" if atomic and zwj else
                                "atomic" if atomic else "zwj" if zwj else "none"),
            "unicode_form": "NFC" if unicodedata.is_normalized("NFC", text) else "other",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
        rows.append(m)
        print(f"[{i:>3}/{len(works)}] {slug:<34} {m['author']:<28} "
              f"{m['work_type']:<12} {m['word_count']:>7,} words"
              f"{'  (cached)' if from_cache else ''}")

    with open(META, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- summary
    print("\n" + "=" * 70)
    print(f"{len(rows)} works downloaded   ({cached} served from cache)")
    print(f"total words: {sum(int(r['word_count']) for r in rows):,}")
    print(f"metadata:    {META}")
    print(f"texts:       {TEXTDIR}/")

    by_author = {}
    for r in rows:
        a = by_author.setdefault(r["author"], [0, 0])
        a[0] += 1
        a[1] += int(r["word_count"])
    print("\nper author:")
    for a, (n, w) in sorted(by_author.items(), key=lambda x: -x[1][0]):
        print(f"  {a:<32} {n:>3} works   {w:>9,} words")

    types = {}
    for r in rows:
        types[r["work_type"] or "(none)"] = types.get(r["work_type"] or "(none)", 0) + 1
    print("\nwork types (from TEI, not from the list's labels):")
    for t, n in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<24} {n:>3}")

    if failed:
        print(f"\n{len(failed)} failed:")
        for s, why in failed:
            print(f"  {s:<34} {why}")

    print("\nnext:  python corpus_overview.py")


def test_local(path):
    root = ET.parse(path).getroot()
    m = parse_metadata(root)
    text, nparas = extract_text(root)
    print("---- metadata ----")
    for k in COLUMNS:
        if m.get(k):
            v = str(m[k])
            print(f"  {k:<22} {v[:88]}{'…' if len(v) > 88 else ''}")
    print(f"\n---- text ----\n  paragraphs {nparas}, words {len(text.split()):,}")
    print("\n  first 250 chars:")
    print("  " + text[:250].replace("\n", " "))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        test_local(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else IN_FILE)
