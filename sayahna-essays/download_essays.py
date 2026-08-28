#!/usr/bin/env python3
"""
download_essays.py — fully automated crawl of the Sayahna essay collections.

    1. GET https://books.sayahna.org/sfn-article.html
    2. find the <div class="lsection"> whose <div class="lsechead"> is സമാഹാരം
       -> these are the authors with a whole collection of their own
    3. for each author page (e.g. .../html/sfn-karassery.html) find the
       <div class="lsection"> whose <div class="lsechead"> is ലേഖനങ്ങൾ
    4. from every <p class="tocindent"> take the title and the /xml/ link
    5. download and parse each TEI XML

Nothing is hard-coded: add a collection to the site and the next run picks it up.

Usage:
    pip install requests beautifulsoup4
    python download_essays.py
    python download_essays.py --min-works 5        # only authors with >=5 essays
    python download_essays.py --all-sections       # not just ലേഖനങ്ങൾ
    python download_essays.py --list-only          # show what it would fetch
    python download_essays.py --limit 20           # stop after 20 essays (a test run)

Output (same schema as the fiction corpus, so the overview script works on both):
    corpus/xml/<slug>.xml        cached TEI
    corpus/texts/<slug>.txt      essay text
    corpus/metadata.csv          one row per essay
    corpus/authors.csv           one row per author, with the site biography
"""

import argparse
import csv
import os
import re
import sys
import time
import hashlib
import unicodedata
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

INDEX = "https://books.sayahna.org/sfn-article.html"
COLLECTION_HEAD = "സമാഹാരം"      # section of the index listing author pages
ESSAY_HEAD = "ലേഖനങ്ങൾ"          # section of an author page listing essays

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

OUT = "corpus"
XMLDIR = os.path.join(OUT, "xml")
TEXTDIR = os.path.join(OUT, "texts")
HTMLDIR = os.path.join(OUT, "pages")
META = os.path.join(OUT, "metadata.csv")
AUTHORS = os.path.join(OUT, "authors.csv")

SLEEP = 1.0
TIMEOUT = 60
UA = "MalayalamAA-research-collector/1.0 (M.Tech corpus study)"

# Optional: force an English name for an author. Key is the Malayalam name as
# it appears in the index. Leave empty to use whatever the TEI headers say.
AUTHOR_NAMES = {
    # "കാരശ്ശേരി എം എൻ": "M. N. Karassery",
}

MAL = re.compile(r"[\u0D00-\u0D7F]")
ATOMIC_CHILLU = re.compile(r"[\u0D7A-\u0D7F]")
ZWJ_CHILLU = re.compile(r"[\u0D15-\u0D39]\u0D4D\u200D")

COLUMNS = [
    "work_id", "author", "author_ml", "author_name_tei", "author_page",
    "title_ml", "title_en", "toc_title", "section_label",
    "work_type", "keywords", "domain", "derivation", "factuality",
    "constitution", "language",
    "source_url", "html_url", "publisher", "pub_place",
    "digital_edition_date", "digital_edition_year",
    "source_pub_year", "source_publisher", "source_pages",
    "typeset_by", "edited_by", "encoded_by", "sponsor", "funder",
    "licence", "licence_url", "setting_place", "setting_time",
    "word_count", "char_count", "para_count",
    "chillu_atomic", "chillu_zwj", "chillu_encoding", "unicode_form",
    "sha256",
]

S = requests.Session()
S.headers.update({"User-Agent": UA})


# ------------------------------------------------------------------ helpers

def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("\u200b", "").replace("\u200d", "")).strip()


def slug_of(url):
    return re.sub(r"\.(xml|html?|pdf)$", "", url.rstrip("/").split("/")[-1], flags=re.I)


def mal_ratio(s):
    t = re.sub(r"\s", "", s)
    return len(MAL.findall(t)) / len(t) if t else 0.0


def get(url, cache):
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        return open(cache, encoding="utf-8").read(), True
    r = S.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return None, False
    r.encoding = "utf-8"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "w", encoding="utf-8").write(r.text)
    time.sleep(SLEEP)
    return r.text, False


def find_section(soup, head_text):
    """The <div class='lsection'> whose <div class='lsechead'> matches."""
    want = norm(head_text)
    for sec in soup.find_all("div", class_="lsection"):
        head = sec.find("div", class_="lsechead")
        if head and norm(head.get_text()) == want:
            return sec
    return None


def all_sections(soup):
    out = []
    for sec in soup.find_all("div", class_="lsection"):
        head = sec.find("div", class_="lsechead")
        out.append((norm(head.get_text()) if head else "(unnamed)", sec))
    return out


# ------------------------------------------------------------- step 1 and 2

def author_pages(index_html):
    """Author-collection pages listed under സമാഹാരം on the index."""
    soup = BeautifulSoup(index_html, "html.parser")
    sec = find_section(soup, COLLECTION_HEAD)
    if sec is None:
        print(f"  !! no '{COLLECTION_HEAD}' section found; falling back to any sfn- link")
        sec = soup
    seen, out = set(), []
    for a in sec.find_all("a", href=True):
        href = a["href"]
        if "/html/sfn-" not in href or not href.endswith(".html"):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append((norm(a.get_text()), href))
    return out


def essays_on_page(page_html, take_all=False):
    """
    (toc_title, xml_url) for every entry in the ലേഖനങ്ങൾ section,
    plus the author's display name, biography, and the section names present.
    """
    soup = BeautifulSoup(page_html, "html.parser")

    display = ""
    for tag in ("h1", "h2", "h3"):
        el = soup.find(tag)
        if el and norm(el.get_text()):
            display = norm(el.get_text())
            break

    bio_parts = []
    for p in soup.find_all("p"):
        if "tocindent" in (p.get("class") or []):
            continue
        t = norm(p.get_text())
        if len(t) > 80 and mal_ratio(t) > 0.5:
            bio_parts.append(t)
        if len(" ".join(bio_parts)) > 1500:
            break
    bio = " ".join(bio_parts)[:2000]

    names = [n for n, _ in all_sections(soup)]

    if take_all:
        blocks = [s for _, s in all_sections(soup)] or [soup]
    else:
        sec = find_section(soup, ESSAY_HEAD)
        blocks = [sec] if sec is not None else [soup]

    items, seen = [], set()
    for block in blocks:
        for p in block.find_all("p", class_="tocindent"):
            link = None
            for a in p.find_all("a", href=True):
                if re.search(r"/xml/[^/]+\.xml$", a["href"]):
                    link = a["href"]
                    break
            if not link or link in seen:
                continue
            seen.add(link)
            clone = BeautifulSoup(str(p), "html.parser")
            for sp in clone.find_all("span", class_="toclink"):
                sp.decompose()
            title = norm(clone.get_text()).lstrip("⦾").strip(" —-–\u2014")
            items.append((title, link))
    return display, bio, names, items


# ---------------------------------------------------------- TEI (unchanged)

def txt(el):
    return "" if el is None else re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_metadata(root):
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
            y = re.search(r"\b(19|20)\d\d\b", (d.get("when") or "") + " " + txt(d))
            m["digital_edition_year"] = y.group(0) if y else ""
        av = ps.find(f"{TEI}availability")
        if av is not None:
            m["licence"] = txt(av)[:200]
            ref = av.find(f".//{TEI}ref")
            if ref is not None:
                m["licence_url"] = ref.get("target", "")
    sd = fd.find(f"{TEI}sourceDesc") if fd is not None else None
    if sd is not None:
        bf = sd.find(f"{TEI}biblFull")
        scope = bf if bf is not None else sd
        sp = scope.find(f"{TEI}publicationStmt")
        if sp is not None:
            m["source_publisher"] = txt(sp.find(f"{TEI}publisher"))
            y = re.search(r"\b(1[6-9]\d\d|20[0-2]\d)\b", txt(sp.find(f"{TEI}date")))
            m["source_pub_year"] = y.group(0) if y else ""
        meas = scope.find(f".//{TEI}measure")
        if meas is not None:
            m["source_pages"] = meas.get("quantity", "") or txt(meas)
    pdsc = h.find(f"{TEI}profileDesc")
    if pdsc is not None:
        lang = pdsc.find(f".//{TEI}language")
        if lang is not None:
            m["language"] = lang.get("ident", "") or txt(lang)
        m["keywords"] = "; ".join(txt(t) for t in pdsc.iter(f"{TEI}term"))
        td = pdsc.find(f".//{TEI}textDesc")
        if td is not None:
            m["work_type"] = (td.get("n") or "").strip().lower()
            for tag, col in [("domain", "domain"), ("derivation", "derivation"),
                             ("factuality", "factuality"), ("constitution", "constitution")]:
                el = td.find(f"{TEI}{tag}")
                if el is not None:
                    m[col] = el.get("type", "")
        st = pdsc.find(f".//{TEI}setting")
        if st is not None:
            m["setting_place"] = txt(st.find(f"{TEI}name")).strip(" ,")
            m["setting_time"] = txt(st.find(f"{TEI}time"))
    if not m["work_type"] and m["keywords"]:
        m["work_type"] = m["keywords"].split(";")[0].strip().lower()
    return m


def extract_text(root):
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


# ----------------------------------------------------------------------- main

def main(a):
    for d in (XMLDIR, TEXTDIR, HTMLDIR):
        os.makedirs(d, exist_ok=True)

    print(f"index: {INDEX}")
    idx, _ = get(INDEX, os.path.join(HTMLDIR, "sfn-article.html"))
    if not idx:
        sys.exit("could not fetch the index page")

    pages = author_pages(idx)
    print(f"{len(pages)} author collections under '{COLLECTION_HEAD}'\n")

    plan, author_rows = [], []
    for name_ml, url in pages:
        html, cached = get(url, os.path.join(HTMLDIR, slug_of(url) + ".html"))
        if not html:
            print(f"  !! {slug_of(url)}: unreachable")
            continue
        display, bio, sections, items = essays_on_page(html, a.all_sections)
        if len(items) < a.min_works:
            print(f"  -- {display or name_ml:<34} {len(items):>4} works — below "
                  f"--min-works {a.min_works}, skipped")
            continue
        print(f"  {display or name_ml:<34} {len(items):>4} works   "
              f"sections: {', '.join(sections) or '(none)'}")
        author_rows.append({"author_ml": name_ml, "display_name_ml": display,
                            "author_page": url, "n_listed": len(items),
                            "sections": " | ".join(sections), "biography": bio})
        for title, xml_url in items:
            plan.append((name_ml, url, title, xml_url))

    print(f"\n{len(plan)} essays from {len(author_rows)} authors")
    if a.limit:
        plan = plan[:a.limit]
        print(f"--limit {a.limit}: fetching only the first {len(plan)}")
    if a.list_only:
        for name_ml, _, title, xml_url in plan[:60]:
            print(f"  {name_ml:<26} {title[:44]:<46} {xml_url}")
        if len(plan) > 60:
            print(f"  ... and {len(plan)-60} more")
        return
    mins = len(plan) * SLEEP / 60
    print(f"about {mins:.0f} min if nothing is cached; safe to re-run\n")

    rows, failed, cached_n = [], [], 0
    for i, (name_ml, page, title, xml_url) in enumerate(plan, 1):
        slug = slug_of(xml_url)
        try:
            raw, from_cache = get(xml_url, os.path.join(XMLDIR, slug + ".xml"))
        except Exception as e:
            failed.append((slug, str(e)[:60]))
            continue
        if not raw:
            failed.append((slug, "404"))
            print(f"[{i:>4}/{len(plan)}] {slug:<38} 404")
            continue
        cached_n += from_cache
        try:
            root = ET.fromstring(raw.encode("utf-8"))
        except ET.ParseError as e:
            failed.append((slug, f"bad XML: {e}"))
            continue

        m = parse_metadata(root)
        text, nparas = extract_text(root)
        if not text.strip():
            failed.append((slug, "empty body"))
            continue

        open(os.path.join(TEXTDIR, slug + ".txt"), "w", encoding="utf-8").write(text)
        atomic = len(ATOMIC_CHILLU.findall(text))
        zwj = len(ZWJ_CHILLU.findall(text))
        m.update({
            "work_id": slug,
            "author": AUTHOR_NAMES.get(name_ml, "") or m["author_name_tei"] or name_ml,
            "author_ml": name_ml,
            "author_page": page,
            "toc_title": title,
            "section_label": "Essay",
            "source_url": xml_url,
            "html_url": re.sub(r"/xml/(.+)\.xml$", r"/html/\1.html", xml_url),
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
        if i % 25 == 0 or i == len(plan):
            print(f"[{i:>4}/{len(plan)}] {slug:<38} {m['word_count']:>6,} words")

    # one canonical English name per author: the commonest TEI spelling
    best = {}
    for r in rows:
        d = best.setdefault(r["author_ml"], {})
        n = r["author_name_tei"]
        if n:
            d[n] = d.get(n, 0) + 1
    canon = {k: (AUTHOR_NAMES.get(k) or (max(v, key=v.get) if v else k))
             for k, v in best.items()}
    for r in rows:
        r["author"] = canon.get(r["author_ml"], r["author"])

    with open(META, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for ar in author_rows:
        ar["author"] = canon.get(ar["author_ml"], "")
    if author_rows:
        with open(AUTHORS, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["author", "author_ml", "display_name_ml",
                                              "author_page", "n_listed", "sections",
                                              "biography"], extrasaction="ignore")
            w.writeheader()
            w.writerows(author_rows)

    print("\n" + "=" * 74)
    print(f"{len(rows)} essays downloaded   ({cached_n} from cache)")
    print(f"total words: {sum(int(r['word_count']) for r in rows):,}")
    print(f"metadata:    {META}")
    print(f"authors:     {AUTHORS}")

    agg = {}
    for r in rows:
        s = agg.setdefault(r["author"], [0, 0])
        s[0] += 1
        s[1] += int(r["word_count"])
    print("\nper author:")
    for k, (n, wd) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"  {k:<38} {n:>4} essays   {wd:>10,} words")

    types = {}
    for r in rows:
        types[r["work_type"] or "(none)"] = types.get(r["work_type"] or "(none)", 0) + 1
    print("\nwork types (from TEI):")
    for t, n in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<28} {n:>4}")

    if failed:
        print(f"\n{len(failed)} failed:")
        for s, why in failed[:25]:
            print(f"  {s:<38} {why}")
        if len(failed) > 25:
            print(f"  ... and {len(failed)-25} more")

    print("\nnext:  python corpus_overview.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-works", type=int, default=2,
                    help="skip authors with fewer than this many essays")
    ap.add_argument("--all-sections", action="store_true",
                    help="take every lsection on an author page, not just ലേഖനങ്ങൾ")
    ap.add_argument("--list-only", action="store_true",
                    help="show what would be fetched, download nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N essays (for a quick test run)")
    main(ap.parse_args())
