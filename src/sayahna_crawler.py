#!/usr/bin/env python3
"""
sayahna_crawler.py — build a Malayalam corpus from sayahna.org, with the
work text, the author biography and the metadata kept strictly apart.

WHY THE SEPARATION MATTERS
Sayahna appends an "about the author" section to the body of each work. It is
a <div type="lsection"> carrying the author's name as its heading, a portrait,
and often a list of their books. If that text is left in the training data,
a classifier reads the author's own name and birth year instead of their
prose. This script splits it out structurally rather than by guesswork.

    <body>
      <div type="lchapter">
        <div type="lsection">                      -> THE WORK        (texts/)
        <div type="lsection"> head=<author name>   -> BIOGRAPHY       (bios/)
            <figure type="gra">                       (portrait)
            <div type="subsection"> head=കൃതികൾ       (list of works)
    <back> Colophon                                -> METADATA        (metadata.csv)

SOURCES CRAWLED
    https://sayahna.org/sfn-article.html      articles and essays
    https://sayahna.org/sfn-shortstory.html   fiction: stories, novels, plays
    https://sayahna.org/sfn-poetry.html       poetry

Each index lists works two ways: directly, as "Author: Title" with pdf/xml/html
links, and indirectly, as a link to an author-collection page which lists more
works. Both are followed.

Usage:
    pip install requests beautifulsoup4
    python sayahna_crawler.py                       # crawl everything
    python sayahna_crawler.py --list-only           # show what it would fetch
    python sayahna_crawler.py --indexes shortstory  # one index only
    python sayahna_crawler.py --limit 20            # short test run
    python sayahna_crawler.py --check work.xml      # split one local file

Output:
    corpus/xml/<id>.xml        raw TEI, cached (re-runs are free)
    corpus/texts/<id>.txt      THE WORK ONLY — this is the training data
    corpus/bios/<id>.txt       the author biography that shipped with it
    corpus/metadata.csv        one row per work, every TEI header field
    corpus/authors.csv         one row per author, with their biography once
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
from collections import defaultdict

import requests
from bs4 import BeautifulSoup

T = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

INDEXES = {
    "article":    "https://sayahna.org/sfn-article.html",
    "shortstory": "https://sayahna.org/sfn-shortstory.html",
    "poetry":     "https://sayahna.org/sfn-poetry.html",
}

OUT = "corpus"
XMLD = os.path.join(OUT, "xml")
TEXTD = os.path.join(OUT, "texts")
BIOD = os.path.join(OUT, "bios")
PAGED = os.path.join(OUT, "pages")
META = os.path.join(OUT, "metadata.csv")
AUTHORS = os.path.join(OUT, "authors.csv")

SLEEP = 1.0
TIMEOUT = 60
UA = "MalayalamAA-corpus-builder/2.0 (M.Tech research; contact: your-email@example.com)"

MAL = re.compile(r"[\u0D00-\u0D7F]")
ATOMIC_CHILLU = re.compile(r"[\u0D7A-\u0D7F]")
ZWJ_CHILLU = re.compile(r"[\u0D15-\u0D39]\u0D4D\u200D")

# a <div type="subsection"> with one of these headings belongs to the biography
BIO_SUBSECTION_HEADS = [
    "കൃതികൾ", "പുസ്തകങ്ങൾ", "രചനകൾ", "പ്രധാന കൃതികൾ", "ഗ്രന്ഥങ്ങൾ",
    "അവാർഡുകൾ", "പുരസ്കാരങ്ങൾ", "ജീവചരിത്രം",
]

# field labels that only ever appear in an author note
BIO_LABELS = [
    "പിതാവു്", "പിതാവ്", "മാതാവു്", "മാതാവ്", "ഭാര്യ", "ഭർത്താവു്", "പത്നി",
    "മക്കൾ", "ജനനം", "മരണം", "വിദ്യാഭ്യാസം", "തൊഴിൽ", "സ്വദേശം",
    "മുഴുവൻ പേരു്", "ജീവചരിത്രക്കുറിപ്പു്", "ലഘുജീവചരിത്രം",
]

# an index entry marked this way is a translation, not original Malayalam
TRANSLATION_MARKS = ["വിവ:", "വിവർത്തനം", "പരിഭാഷ", "മൊഴിമാറ്റം",
                     "(trans.)", "(trans)", "translated"]

COLUMNS = [
    "work_id", "author", "author_index", "author_tei", "author_ml",
    "title_ml", "title_en", "index_title",
    "index_page", "collection_page", "work_type", "keywords",
    "domain", "derivation", "factuality", "constitution", "language",
    "is_translation", "translator",
    "xml_url", "html_url", "pdf_url",
    "publisher", "pub_place", "digital_edition_date", "digital_edition_year",
    "source_pub_year", "source_publisher", "source_pages",
    "typeset_by", "edited_by", "encoded_by", "sponsor", "funder",
    "licence", "licence_url", "setting_place", "setting_time",
    "n_chapters", "n_sections",
    "text_words", "text_chars", "text_paras",
    "bio_words", "bio_found", "bio_rule",
    "chillu_atomic", "chillu_zwj", "chillu_encoding", "unicode_form",
    "text_sha256",
]

S = requests.Session()
S.headers.update({"User-Agent": UA})


# ------------------------------------------------------------------ helpers

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def txt_of(el):
    return "" if el is None else norm("".join(el.itertext()))


def mal_ratio(s):
    t = re.sub(r"\s", "", s)
    return len(MAL.findall(t)) / len(t) if t else 0.0


def slug_of(url):
    return re.sub(r"\.(xml|html?|pdf)$", "",
                  url.rstrip("/").split("/")[-1], flags=re.I)


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


def split_author_title(s):
    """
    'കാരശ്ശേരി എം എൻ: പതിനാലാം രാവു്'  ->  (author, title)
    Colons inside brackets are ignored, so '(വിവ: രവിവർമ്മ)' does not confuse it.
    """
    depth, cut = 0, -1
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            cut = i
            break
    if cut < 0:
        return "", norm(s)
    return norm(s[:cut]), norm(s[cut + 1:])


def translation_info(s):
    low = s.lower()
    if not any(m in s or m in low for m in TRANSLATION_MARKS):
        return "no", ""
    m = re.search(r"\(\s*(?:വിവ|പരിഭാഷ|വിവർത്തനം)\s*[:.]\s*([^)]+)\)", s)
    return "yes", norm(m.group(1)) if m else ""


# ------------------------------------------------------------ index crawling

def parse_index(html, page_url):
    """
    Returns (direct_works, collection_pages).
      direct_works    : dicts with xml/html/pdf urls and the 'Author: Title' text
      collection_pages: (author_name, url) for sfn-*.html author pages
    """
    soup = BeautifulSoup(html, "html.parser")
    works, pages, seen_x, seen_p = [], [], set(), set()

    for p in soup.find_all("p", class_="tocindent"):
        links = p.find_all("a", href=True)
        xml_url = next((a["href"] for a in links
                        if re.search(r"/xml/[^/]+\.xml$", a["href"])), None)

        if xml_url:
            if xml_url in seen_x:
                continue
            seen_x.add(xml_url)
            clone = BeautifulSoup(str(p), "html.parser")
            for sp in clone.find_all("span", class_="toclink"):
                sp.decompose()
            label = norm(clone.get_text()).lstrip("⦾").strip(" —-–")
            works.append({
                "label": label,
                "xml_url": xml_url,
                "html_url": next((a["href"] for a in links
                                  if "/html/" in a["href"]), ""),
                "pdf_url": next((a["href"] for a in links
                                 if a["href"].endswith(".pdf")), ""),
                "index_page": page_url,
                "collection_page": "",
            })
            continue

        for a in links:
            href = a["href"]
            if re.search(r"/html/sfn-[^/]+\.html$", href) and "-cover" not in href:
                if href in seen_p:
                    continue
                seen_p.add(href)
                pages.append((norm(a.get_text()), href))

    # author-collection links can also sit outside p.tocindent
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/html/sfn-[^/]+\.html$", href) and "-cover" not in href:
            if href not in seen_p:
                seen_p.add(href)
                pages.append((norm(a.get_text()), href))

    return works, pages


def parse_collection(html, page_url, author_hint):
    """Works listed on an author-collection page."""
    soup = BeautifulSoup(html, "html.parser")
    display = author_hint
    for tag in ("h1", "h2", "h3"):
        el = soup.find(tag)
        if el and norm(el.get_text()):
            display = norm(el.get_text())
            break

    out, seen = [], set()
    for p in soup.find_all("p", class_="tocindent"):
        links = p.find_all("a", href=True)
        xml_url = next((a["href"] for a in links
                        if re.search(r"/xml/[^/]+\.xml$", a["href"])), None)
        if not xml_url or xml_url in seen:
            continue
        seen.add(xml_url)
        clone = BeautifulSoup(str(p), "html.parser")
        for sp in clone.find_all("span", class_="toclink"):
            sp.decompose()
        label = norm(clone.get_text()).lstrip("⦾").strip(" —-–")
        out.append({
            "label": (label if ":" in label else f"{display}: {label}"),
            "xml_url": xml_url,
            "html_url": next((a["href"] for a in links if "/html/" in a["href"]), ""),
            "pdf_url": next((a["href"] for a in links if a["href"].endswith(".pdf")), ""),
            "index_page": "",
            "collection_page": page_url,
        })
    return display, out


# --------------------------------------------------------------- TEI header

def parse_header(root):
    h = root.find(f"{T}teiHeader")
    m = {c: "" for c in COLUMNS}
    if h is None:
        return m
    fd = h.find(f"{T}fileDesc")
    ts = fd.find(f"{T}titleStmt") if fd is not None else None
    if ts is not None:
        for t in ts.iter(f"{T}title"):
            lang = t.get(XML_LANG)
            if lang == "ml" and not m["title_ml"]:
                m["title_ml"] = txt_of(t)
            elif lang == "en" and not m["title_en"]:
                m["title_en"] = txt_of(t)
        m["author_tei"] = txt_of(ts.find(f"{T}author"))
        m["sponsor"] = txt_of(ts.find(f"{T}sponsor"))
        m["funder"] = txt_of(ts.find(f"{T}funder"))
        for rs in ts.findall(f"{T}respStmt"):
            resp = txt_of(rs.find(f"{T}resp")).lower()
            who = txt_of(rs.find(f"{T}name"))
            if "typeset" in resp:
                m["typeset_by"] = who
            elif "edit" in resp:
                m["edited_by"] = who
            elif "encod" in resp:
                m["encoded_by"] = who
            elif "translat" in resp:
                m["translator"] = who
                m["is_translation"] = "yes"
    ps = fd.find(f"{T}publicationStmt") if fd is not None else None
    if ps is not None:
        m["publisher"] = txt_of(ps.find(f"{T}publisher"))
        m["pub_place"] = txt_of(ps.find(f"{T}pubPlace"))
        d = ps.find(f"{T}date")
        if d is not None:
            m["digital_edition_date"] = txt_of(d)
            y = re.search(r"\b(19|20)\d\d\b", (d.get("when") or "") + " " + txt_of(d))
            m["digital_edition_year"] = y.group(0) if y else ""
        av = ps.find(f"{T}availability")
        if av is not None:
            a = txt_of(av)
            m["licence"] = ("CC-BY-NC-SA-4.0" if "NonCommercial" in a
                            else "CC-BY-SA-4.0" if "ShareAlike" in a else a[:80])
            ref = av.find(f".//{T}ref")
            if ref is not None:
                m["licence_url"] = ref.get("target", "")
    sd = fd.find(f"{T}sourceDesc") if fd is not None else None
    if sd is not None:
        bf = sd.find(f"{T}biblFull")
        scope = bf if bf is not None else sd
        sp = scope.find(f"{T}publicationStmt")
        if sp is not None:
            m["source_publisher"] = txt_of(sp.find(f"{T}publisher"))
            y = re.search(r"\b(1[6-9]\d\d|20[0-2]\d)\b", txt_of(sp.find(f"{T}date")))
            m["source_pub_year"] = y.group(0) if y else ""
        meas = scope.find(f".//{T}measure")
        if meas is not None:
            m["source_pages"] = meas.get("quantity", "") or txt_of(meas)
    pd = h.find(f"{T}profileDesc")
    if pd is not None:
        lang = pd.find(f".//{T}language")
        if lang is not None:
            m["language"] = lang.get("ident", "") or txt_of(lang)
        terms = [txt_of(t) for t in pd.iter(f"{T}term")]
        m["keywords"] = "; ".join(terms)
        ml = [t for t in terms if mal_ratio(t) > 0.5]
        m["author_ml"] = ml[0] if ml else ""
        td = pd.find(f".//{T}textDesc")
        if td is not None:
            m["work_type"] = (td.get("n") or "").strip().lower()
            for tag in ("domain", "derivation", "factuality", "constitution"):
                el = td.find(f"{T}{tag}")
                if el is not None:
                    m[tag] = el.get("type", "")
        st = pd.find(f".//{T}setting")
        if st is not None:
            m["setting_place"] = txt_of(st.find(f"{T}name")).strip(" ,")
            m["setting_time"] = txt_of(st.find(f"{T}time"))
    if not m["work_type"] and m["keywords"]:
        m["work_type"] = m["keywords"].split(";")[0].strip().lower()
    return m


# ------------------------------------------------ separating text from bio

def name_tokens(*names):
    out = set()
    for n in names:
        for p in re.split(r"[\s.,]+", str(n)):
            p = p.strip(" .,\u200c\u200d")
            if len(p) >= 4:
                out.add(p)
    return out


def is_bio_section(div, names, is_last):
    """
    Decide whether an <div type='lsection'> is the author note.
    Returns the rule that fired, or "" if it is part of the work.
    """
    head = txt_of(div.find(f"{T}head"))
    body_txt = " ".join(txt_of(p) for p in div.iter(f"{T}p"))

    # a portrait only ever appears in the author note
    for fig in div.iter(f"{T}figure"):
        if fig.get("type") == "gra" or fig.get("rend") == "fleft":
            return "portrait"

    # heading is the author's own name
    if head and names:
        hits = sum(1 for tok in names if tok in head)
        if hits and len(head.split()) <= 8:
            return "head=author-name"

    # a sub-section listing the author's books
    for sub in div.iter(f"{T}div"):
        if txt_of(sub.find(f"{T}head")) in BIO_SUBSECTION_HEADS:
            return "works-list"

    # trailing short block full of biographical field labels
    if is_last and len(body_txt.split()) < 250:
        if sum(1 for lab in BIO_LABELS if lab in body_txt) >= 2:
            return "bio-labels"

    return ""


def paragraphs(el):
    out, prev = [], None
    for p in el.iter(f"{T}p"):
        t = txt_of(p)
        if len(t) < 5 or mal_ratio(t) < 0.30:
            continue
        if t != prev:
            out.append(t)
        prev = t
    return out


def split_body(root, names):
    """
    -> (work_text, bio_text, n_chapters, n_sections, rule)
    <front> and <back> are never read: cover art and colophon are metadata.
    """
    body = root.find(f"{T}text/{T}body")
    if body is None:
        return "", "", 0, 0, ""

    chapters = [d for d in body.iter(f"{T}div")
                if (d.get("type") or "").lower() in ("lchapter", "chapter")]
    if not chapters:
        chapters = [body]

    work_paras, bio_paras, rules, nsec = [], [], [], 0
    for ch in chapters:
        secs = [d for d in ch.findall(f"{T}div")
                if (d.get("type") or "").lower() in ("lsection", "section")]
        if not secs:
            work_paras += paragraphs(ch)
            continue
        nsec += len(secs)
        for i, sec in enumerate(secs):
            rule = is_bio_section(sec, names, i == len(secs) - 1)
            if rule:
                bio_paras += paragraphs(sec)
                rules.append(rule)
            else:
                work_paras += paragraphs(sec)

    return ("\n\n".join(work_paras).strip(),
            "\n\n".join(bio_paras).strip(),
            len(chapters), nsec, ";".join(sorted(set(rules))))


# ------------------------------------------------------------------- main

def collect_plan(a):
    plan, seen = [], set()
    coll_names = {}
    for key in a.indexes:
        url = INDEXES[key]
        print(f"\nindex: {url}")
        html, _ = get(url, os.path.join(PAGED, f"{key}.html"))
        if not html:
            print("  !! unreachable")
            continue
        works, pages = parse_index(html, url)
        print(f"  {len(works)} direct works, {len(pages)} author pages")
        for w in works:
            if w["xml_url"] not in seen:
                seen.add(w["xml_url"])
                w["index_key"] = key
                plan.append(w)
        for name, purl in pages:
            phtml, _ = get(purl, os.path.join(PAGED, slug_of(purl) + ".html"))
            if not phtml:
                continue
            display, pworks = parse_collection(phtml, purl, name)
            coll_names[purl] = display
            new = 0
            for w in pworks:
                if w["xml_url"] not in seen:
                    seen.add(w["xml_url"])
                    w["index_key"] = key
                    plan.append(w)
                    new += 1
            print(f"    {display:<34} {len(pworks):>4} listed, {new:>4} new")
    return plan


def main(a):
    for d in (XMLD, TEXTD, BIOD, PAGED):
        os.makedirs(d, exist_ok=True)

    plan = collect_plan(a)
    print(f"\n{len(plan)} unique works found across {len(a.indexes)} index page(s)")
    if a.limit:
        plan = plan[:a.limit]
        print(f"--limit {a.limit}")
    if a.list_only:
        for w in plan[:80]:
            print(f"  [{w['index_key']:<10}] {w['label'][:64]}")
        if len(plan) > 80:
            print(f"  ... and {len(plan)-80} more")
        return
    print(f"about {len(plan)*SLEEP/60:.0f} min if nothing is cached\n")

    rows, bios, failed = [], defaultdict(list), []
    for i, wk in enumerate(plan, 1):
        wid = slug_of(wk["xml_url"])
        try:
            raw, cached = get(wk["xml_url"], os.path.join(XMLD, wid + ".xml"))
        except Exception as e:
            failed.append((wid, str(e)[:50]))
            continue
        if not raw:
            failed.append((wid, "404"))
            continue
        try:
            root = ET.fromstring(raw.encode("utf-8"))
        except ET.ParseError as e:
            failed.append((wid, f"bad XML: {e}"))
            continue

        m = parse_header(root)
        idx_author, idx_title = split_author_title(wk["label"])
        is_tr, translator = translation_info(wk["label"])
        if m["is_translation"] != "yes":
            m["is_translation"] = is_tr
        if translator and not m["translator"]:
            m["translator"] = translator

        author = m["author_tei"] or idx_author or "UNKNOWN"
        names = name_tokens(author, idx_author, m["author_ml"])
        text, bio, nch, nsec, rule = split_body(root, names)

        if len(text.split()) < 100:
            failed.append((wid, f"only {len(text.split())} words of work text"))
            continue

        open(os.path.join(TEXTD, wid + ".txt"), "w", encoding="utf-8").write(text)
        if bio:
            open(os.path.join(BIOD, wid + ".txt"), "w", encoding="utf-8").write(bio)
            bios[author].append(bio)

        atomic = len(ATOMIC_CHILLU.findall(text))
        zwj = len(ZWJ_CHILLU.findall(text))
        m.update({
            "work_id": wid, "author": author,
            "author_index": idx_author, "index_title": idx_title,
            "index_page": wk["index_page"], "collection_page": wk["collection_page"],
            "xml_url": wk["xml_url"], "html_url": wk["html_url"], "pdf_url": wk["pdf_url"],
            "n_chapters": nch, "n_sections": nsec,
            "text_words": len(text.split()), "text_chars": len(text),
            "text_paras": len(text.split("\n\n")),
            "bio_words": len(bio.split()), "bio_found": "yes" if bio else "no",
            "bio_rule": rule,
            "chillu_atomic": atomic, "chillu_zwj": zwj,
            "chillu_encoding": ("mixed" if atomic and zwj else "atomic" if atomic
                                else "zwj" if zwj else "none"),
            "unicode_form": "NFC" if unicodedata.is_normalized("NFC", text) else "other",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
        rows.append(m)
        if i % 25 == 0 or i == len(plan):
            print(f"[{i:>4}/{len(plan)}] {wid:<34} {m['text_words']:>6} words "
                  f"| bio {m['bio_words']:>4} ({rule or 'none'})")

    with open(META, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    with open(AUTHORS, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["author", "n_works", "total_words", "biography"])
        agg = defaultdict(lambda: [0, 0])
        for r in rows:
            agg[r["author"]][0] += 1
            agg[r["author"]][1] += int(r["text_words"])
        for au, (n, wd) in sorted(agg.items(), key=lambda x: -x[1][0]):
            longest = max(bios.get(au, [""]), key=len)
            wr.writerow([au, n, wd, longest])

    # ---- summary
    print("\n" + "=" * 76)
    print(f"{len(rows)} works | {sum(int(r['text_words']) for r in rows):,} words of TEXT")
    nb = sum(1 for r in rows if r["bio_found"] == "yes")
    bw = sum(int(r["bio_words"]) for r in rows)
    print(f"biography separated from {nb}/{len(rows)} works ({bw:,} words held out)")
    print(f"  texts    -> {TEXTD}/     <- train on these")
    print(f"  bios     -> {BIOD}/      <- never train on these")
    print(f"  metadata -> {META}")
    print(f"  authors  -> {AUTHORS}")

    byrule = defaultdict(int)
    for r in rows:
        byrule[r["bio_rule"] or "(none found)"] += 1
    print("\nbio detected by rule:")
    for k, v in sorted(byrule.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v:>5}")

    tr = sum(1 for r in rows if r["is_translation"] == "yes")
    print(f"\ntranslations flagged: {tr}")
    types = defaultdict(int)
    for r in rows:
        types[r["work_type"] or "(none)"] += 1
    print("work types:")
    for k, v in sorted(types.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k:<24} {v:>5}")

    agg = defaultdict(int)
    for r in rows:
        agg[r["author"]] += 1
    multi = {k: v for k, v in agg.items() if v >= 5}
    print(f"\nauthors: {len(agg)} total, {len(multi)} with >=5 works")
    for k, v in sorted(multi.items(), key=lambda x: -x[1])[:25]:
        print(f"  {k:<44} {v:>4}")

    if failed:
        print(f"\n{len(failed)} failed:")
        for w, why in failed[:20]:
            print(f"  {w:<34} {why}")


def check(path):
    root = ET.parse(path).getroot()
    m = parse_header(root)
    names = name_tokens(m["author_tei"], m["author_ml"])
    text, bio, nch, nsec, rule = split_body(root, names)
    print(f"author      : {m['author_tei']}   (ml: {m['author_ml']})")
    print(f"title       : {m['title_ml']} / {m['title_en']}")
    print(f"work_type   : {m['work_type']}    licence: {m['licence']}")
    print(f"structure   : {nch} chapter(s), {nsec} section(s)")
    print(f"\nWORK TEXT   : {len(text.split()):,} words")
    print("   " + text[:200].replace("\n", " ") + " …")
    print(f"\nBIOGRAPHY   : {len(bio.split()):,} words   (rule: {rule or 'not found'})")
    print("   " + (bio[:200].replace("\n", " ") + " …" if bio else "(none)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes", nargs="*", default=list(INDEXES),
                    choices=list(INDEXES))
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", metavar="FILE.xml")
    a = ap.parse_args()
    if a.check:
        check(a.check)
    else:
        main(a)
