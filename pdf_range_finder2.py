#!/usr/bin/env python3
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import fitz  # PyMuPDF

# -----------------------------
# Configurable heuristics
# -----------------------------
CHAPTER_PATTERNS = [
    re.compile(r"^\s*(chapter|chap\.)\s+(?P<num>[ivxlcdm]+|\d+)\b[:\-\.]?\s*(?P<title>[^\n\r]{0,120})$", re.I),
    re.compile(r"^\s*(unit|section)\s+(?P<num>[ivxlcdm]+|\d+)\b[:\-\.]?\s*(?P<title>[^\n\r]{0,120})$", re.I),
    re.compile(r"^\s*(?P<num>\d{1,3})\s+(?P<title>[A-Z][^\n\r]{0,120})$"),
]
TOP_OF_PAGE_FRACTION = 0.30  # consider text within top 30% of the page only

# -----------------------------
# Learning vs. non-learning filters
# -----------------------------
NON_LEARNING_WORDS = [
    r"\bcover\b", r"\btitle\s*page\b", r"\bcopyright\b",
    r"\babout\s+the\s+author(s)?\b", r"\babout\s+this\s+book\b",
    r"\bdedication\b", r"\bpreface\b", r"\bforeword\b",
    r"\backnowledg?e?ments?\b", r"\btable\s+of\s+contents\b", r"\bcontents\b",
    r"\bcredits\b", r"\bpublisher'?s?\s+note\b", r"\bpublication\s+data\b",
    r"\bind(?:ex|ices)\b", r"\breferences\b", r"\bbibliograph(?:y|ies)\b",
    r"\bcolophon\b", r"\bglossary\b", r"\bnotes?\b$",
]
NON_LEARNING_SINGLE_WORDS = {"index", "references", "bibliography", "glossary", "contents"}
ALWAYS_LEARNING_WORDS = [r"\bintroduction\b", r"\boverview\b", r"\bfundamentals\b", r"\bbackground\b", r"\bcase\s+stud(?:y|ies)\b", r"\bmethods?\b"]
NON_LEARNING_RE = re.compile("|".join(NON_LEARNING_WORDS), re.I)
ALWAYS_LEARNING_RE = re.compile("|".join(ALWAYS_LEARNING_WORDS), re.I)

def is_learning_topic(title: str) -> bool:
    if not title:
        return False
    t = title.strip().lower()
    if ALWAYS_LEARNING_RE.search(t):
        return True
    t_nopunct = re.sub(r"[^\w\s]", " ", t)
    t_nopunct = re.sub(r"\s+", " ", t_nopunct).strip()
    if NON_LEARNING_RE.search(t):
        return False
    if t_nopunct in NON_LEARNING_SINGLE_WORDS:
        return False
    if len(t_nopunct) <= 3:
        return False
    return True

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Heading:
    page_index: int  # 0-based
    raw_text: str
    title: str
    number: Optional[str] = None
    confidence: float = 0.5

# -----------------------------
# Utilities
# -----------------------------
def normalize_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\s:\-–—\.]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# -----------------------------
# Extraction via outline (ToC)
# -----------------------------
def outline_chapters(doc: fitz.Document) -> List[Heading]:
    toc = doc.get_toc(simple=True)  # [level, title, page1-based]
    headings: List[Heading] = []
    if not toc:
        return headings

    for level, title, page1 in toc:
        page_index = max(0, int(page1) - 1)
        tnorm = normalize_title(title)
        looks_like_chapter = any(p.search(tnorm) for p in CHAPTER_PATTERNS) or re.search(r"\bchapter\b", tnorm, re.I)

        if level == 1 or looks_like_chapter:
            m = None
            for pat in CHAPTER_PATTERNS:
                m = pat.search(tnorm)
                if m:
                    break
            cleaned = tnorm
            number = None
            if m:
                number = m.groupdict().get("num")
                title_part = m.groupdict().get("title") or ""
                cleaned = normalize_title(title_part or tnorm)

            if is_learning_topic(cleaned or tnorm):
                headings.append(Heading(page_index=page_index, raw_text=tnorm, title=cleaned or tnorm, number=number, confidence=0.9))

    # Deduplicate by page
    unique = {}
    for h in headings:
        unique.setdefault(h.page_index, h)
    return sorted(unique.values(), key=lambda h: h.page_index)

# -----------------------------
# Extraction via page text heuristics
# -----------------------------
def text_headings(doc: fitz.Document) -> List[Heading]:
    headings: List[Heading] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        top_cutoff = page_height * TOP_OF_PAGE_FRACTION

        lines: List[Tuple[float, str, float]] = []  # (y, text, max_font_size)
        for b in page_dict.get("blocks", []):
            for l in b.get("lines", []):
                y_positions = [s["bbox"][1] for s in l.get("spans", [])]
                font_sizes = [s.get("size", 0) for s in l.get("spans", [])]
                if not y_positions or not font_sizes:
                    continue
                y = min(y_positions)
                if y > top_cutoff:
                    continue
                text = "".join(s.get("text", "") for s in l.get("spans", []))
                if not text.strip():
                    continue
                max_font = max(font_sizes)
                lines.append((y, text.strip(), max_font))

        lines.sort(key=lambda t: t[0])
        if lines:
            sizes = [fs for _, _, fs in lines]
            sizes_sorted = sorted(sizes)
            q75 = sizes_sorted[int(0.75 * (len(sizes_sorted) - 1))]
        else:
            q75 = 0

        for _, text, max_font in lines[:15]:
            tnorm = normalize_title(text)
            matched = False
            for pat in CHAPTER_PATTERNS:
                m = pat.match(tnorm)
                if m:
                    title = normalize_title(m.groupdict().get("title") or tnorm)
                    if is_learning_topic(title):
                        number = m.groupdict().get("num")
                        conf = 0.65 + (0.15 if max_font >= q75 else 0.0)
                        headings.append(Heading(page_index=pno, raw_text=tnorm, title=title, number=number, confidence=conf))
                    matched = True
                    break
            if matched:
                continue
            if re.match(r"^[A-Z0-9][A-Z0-9 \-:]{5,}$", tnorm) and max_font >= q75:
                title = tnorm.title()
                if is_learning_topic(title):
                    headings.append(Heading(page_index=pno, raw_text=tnorm, title=title, number=None, confidence=0.6))

    # Deduplicate nearby pages (e.g., running heads)
    dedup: Dict[int, Heading] = {}
    for h in headings:
        if h.page_index not in dedup or h.confidence > dedup[h.page_index].confidence:
            dedup[h.page_index] = h
    return sorted(dedup.values(), key=lambda h: h.page_index)

# -----------------------------
# Range consolidation & output
# -----------------------------
def consolidate_ranges(starts: List[Heading], page_count: int, min_gap: int) -> List[Tuple[Heading, int, int]]:
    if not starts:
        return []
    filtered: List[Heading] = []
    for h in sorted(starts, key=lambda h: h.page_index):
        if not filtered or (h.page_index - filtered[-1].page_index) >= min_gap:
            filtered.append(h)
        elif h.confidence > filtered[-1].confidence:
            filtered[-1] = h

    ranges: List[Tuple[Heading, int, int]] = []
    for i, h in enumerate(filtered):
        start_p = h.page_index
        end_p = (filtered[i + 1].page_index - 1) if i + 1 < len(filtered) else (page_count - 1)
        ranges.append((h, start_p, end_p))
    return ranges

def filter_learning_ranges(ranges: List[Tuple[Heading, int, int]]) -> List[Tuple[Heading, int, int]]:
    return [rng for rng in ranges if is_learning_topic(rng[0].title)]

def build_output(ranges: List[Tuple[Heading, int, int]], one_index: bool) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    i = 1
    for (h, s, e) in ranges:
        start = s + (1 if one_index else 0)
        end = e + (1 if one_index else 0)
        topic = h.title.lower() if h.title else ""
        out[f"Chapter_{i}"] = {"topic": topic, "start": start, "end": end}
        i += 1
    return out

# -----------------------------
# Public function
# -----------------------------
def generate_chapters_json(
    pdf_path: str | Path,
    *,
    one_index: bool = False,
    min_gap: int = 2,
    save_debug: bool = False
) -> Dict[str, Dict[str, object]]:
    """
    Extract *learning* chapter ranges from a PDF and return:
        { "Chapter_1": {"topic": "...", "start": 32, "end": 111}, ... }

    Parameters
    ----------
    pdf_path : str | Path
        Path to the textbook PDF.
    one_index : bool
        If True, page numbers are 1-based in the output. Default False (0-based).
    min_gap : int
        Minimum pages between detected chapter starts. Default 2.
    save_debug : bool
        If True, writes a .headings.tsv with detected headings next to the PDF.

    Returns
    -------
    Dict[str, Dict[str, object]]
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        by_outline = outline_chapters(doc)
        if by_outline:
            ranges = consolidate_ranges(by_outline, doc.page_count, min_gap=min_gap)
            starts_used = by_outline
        else:
            by_text = text_headings(doc)
            ranges = consolidate_ranges(by_text, doc.page_count, min_gap=min_gap)
            starts_used = by_text
    finally:
        doc.close()

    learning_ranges = filter_learning_ranges(ranges)
    result = build_output(learning_ranges, one_index=one_index)

    if save_debug:
        debug_path = pdf_path.with_suffix(".headings.tsv")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("page_index\tconfidence\tnumber\traw_text\ttitle\tis_learning\n")
            for h in starts_used:
                f.write(f"{h.page_index}\t{h.confidence:.2f}\t{h.number or ''}\t{h.raw_text}\t{h.title}\t{is_learning_topic(h.title)}\n")

    return result

# Optional: keep a tiny CLI runner if you still want it
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract learning chapter ranges from a PDF and print JSON.")
    ap.add_argument("pdf", type=Path, help="Path to the input PDF")
    ap.add_argument("--out", type=Path, default=None, help="Save JSON to path")
    ap.add_argument("--one-index", action="store_true", help="Use 1-based page numbers in output")
    ap.add_argument("--min-gap", type=int, default=2, help="Minimum pages between chapter starts")
    ap.add_argument("--save-debug", action="store_true", help="Write a .headings.tsv next to the PDF")
    args = ap.parse_args()

    data = generate_chapters_json(args.pdf, one_index=args.one_index, min_gap=args.min_gap, save_debug=args.save_debug)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
