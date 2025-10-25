#!/usr/bin/env python3
"""
PDF Chapter Range Extractor (learning-topics only)

Given a textbook PDF, this script finds the start and end page of each *learning* chapter
and outputs a JSON mapping like:
{
  "Chapter_1": {"topic": "computer networks and the internet", "start": 32, "end": 111},
  ...
}

Changes vs. original:
  - Added is_learning_topic() to filter out non-educational/front/back matter
    like cover, title page, acknowledgments, ToC, references, index, etc.
  - Applied filtering uniformly to both outline- and heuristic-based headings.
  - Renumbers remaining chapters sequentially after filtering.

Requirements:
  pip install pymupdf

Usage:
  python chapter_finder.py input.pdf --out chapters.json --zero-index

Options:
  --one-index          -> make page numbers 1-based in the output
  --zero-index         -> make page numbers 0-based in the output (default)
  --min-gap 2          -> minimum number of pages between chapter starts
  --save-debug         -> save a debug TSV of detected headings per page
"""

from __future__ import annotations
import argparse
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
    # e.g., "Chapter 1", "CHAPTER 3 - Databases", "Chap. 2: Trees"
    re.compile(r"^\s*(chapter|chap\.)\s+(?P<num>[ivxlcdm]+|\d+)\b[:\-\.]?\s*(?P<title>[^\n\r]{0,120})$", re.I),
    # e.g., "Unit 5 Title", "SECTION 3 — Graphs"
    re.compile(r"^\s*(unit|section)\s+(?P<num>[ivxlcdm]+|\d+)\b[:\-\.]?\s*(?P<title>[^\n\r]{0,120})$", re.I),
    # e.g., "1 Introduction" (bare number + title)
    re.compile(r"^\s*(?P<num>\d{1,3})\s+(?P<title>[A-Z][^\n\r]{0,120})$"),
]

TOP_OF_PAGE_FRACTION = 0.30  # consider text within top 30% of the page only

# -----------------------------
# Filters: learning vs. non-learning topics
# -----------------------------
# Tokens that strongly indicate non-learning front/back matter
# (tuned for technical textbooks; customize as needed)
NON_LEARNING_WORDS = [
    r"\bcover\b",
    r"\btitle\s*page\b",
    r"\bcopyright\b",
    r"\babout\s+the\s+author(s)?\b",
    r"\babout\s+this\s+book\b",
    r"\bdedication\b",
    r"\bpreface\b",
    r"\bforeword\b",
    r"\backnowledg?e?ments?\b",
    r"\btable\s+of\s+contents\b",
    r"\bcontents\b",
    r"\bcredits\b",
    r"\bpublisher'?s?\s+note\b",    
    r"\bpublication\s+data\b",
    r"\bind(?:ex|ices)\b",
    r"\breferences\b",
    r"\bbibliograph(?:y|ies)\b",
    r"\bcolophon\b",
    r"\bglossary\b",
    r"\bnotes?\b$",
]

# Some short/ambiguous single-word titles that are often *not* learning in textbooks
NON_LEARNING_SINGLE_WORDS = {"index", "references", "bibliography", "glossary", "contents"}

# Titles to ALWAYS ALLOW even if they might be flagged by other logic
ALWAYS_LEARNING_WORDS = [
    r"\bintroduction\b",      # often a real chapter in CS/math texts
    r"\boverview\b",
    r"\bfundamentals\b",
    r"\bbackground\b",
    r"\bcase\s+stud(?:y|ies)\b",
    r"\bmethods?\b",
]

NON_LEARNING_RE = re.compile("|".join(NON_LEARNING_WORDS), re.I)
ALWAYS_LEARNING_RE = re.compile("|".join(ALWAYS_LEARNING_WORDS), re.I)

def is_learning_topic(title: str) -> bool:
    """
    Returns True if the given title looks like a learning chapter and False if it
    likely refers to front/back matter.

    Heuristics:
      - If it matches ALWAYS_LEARNING -> True
      - If it matches NON_LEARNING_RE -> False
      - If it's extremely short or in a set of known non-learning single words -> False
      - Otherwise, True
    """
    if not title:
        return False
    t = title.strip().lower()

    # Hard allows first (e.g., "Introduction" should survive even if ToC nearby)
    if ALWAYS_LEARNING_RE.search(t):
        return True

    # Remove punctuation noise before basic checks
    t_nopunct = re.sub(r"[^\w\s]", " ", t)
    t_nopunct = re.sub(r"\s+", " ", t_nopunct).strip()

    # Obvious non-learning?
    if NON_LEARNING_RE.search(t):
        return False
    if t_nopunct in NON_LEARNING_SINGLE_WORDS:
        return False

    # Very short titles (e.g., "Index", "Notes") are often non-learning
    if len(t_nopunct) <= 3:
        return False

    # Default allow
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
    s = re.sub(r"^[\s:\-–—\.]+", "", s)  # leading punctuation
    s = re.sub(r"\s+", " ", s)          # collapse whitespace
    return s

# -----------------------------
# Extraction via outline (ToC)
# -----------------------------
def outline_chapters(doc: fitz.Document) -> List[Heading]:
    toc = doc.get_toc(simple=True)  # [level, title, page_num(1-based)]
    headings: List[Heading] = []
    if not toc:
        return headings

    for level, title, page1 in toc:
        page_index = max(0, int(page1) - 1)
        tnorm = normalize_title(title)
        looks_like_chapter = any(p.search(tnorm) for p in CHAPTER_PATTERNS) or re.search(r"\bchapter\b", tnorm, re.I)

        # Take top-level entries or anything that looks like a chapter
        if level == 1 or looks_like_chapter:
            # Try to extract a nicer "title"
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

            # Filter non-learning before adding
            if is_learning_topic(cleaned or tnorm):
                headings.append(
                    Heading(page_index=page_index, raw_text=tnorm, title=cleaned or tnorm, number=number, confidence=0.9)
                )

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
            # Bonus heuristic: single-line ALL-CAPS 2+ words near top, large font
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
# Range consolidation & output building
# -----------------------------
def consolidate_ranges(starts: List[Heading], page_count: int, min_gap: int) -> List[Tuple[Heading, int, int]]:
    if not starts:
        return []
    # Sort and de-noise: keep starts that are at least min_gap pages apart
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
    """Keep only ranges whose heading titles are learning topics."""
    return [rng for rng in ranges if is_learning_topic(rng[0].title)]

def build_output(ranges: List[Tuple[Heading, int, int]], one_index: bool) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    # Renumber after filtering
    i = 1
    for (h, s, e) in ranges:
        start = s + (1 if one_index else 0)
        end = e + (1 if one_index else 0)
        topic = h.title.lower() if h.title else ""
        out[f"Chapter_{i}"] = {"topic": topic, "start": start, "end": end}
        i += 1
    return out

# -----------------------------
# Orchestration
# -----------------------------
def detect_chapters(pdf_path: Path, min_gap: int = 2) -> Tuple[List[Tuple[Heading, int, int]], List[Heading]]:
    doc = fitz.open(pdf_path)
    try:
        by_outline = outline_chapters(doc)
        if by_outline:
            ranges = consolidate_ranges(by_outline, doc.page_count, min_gap=min_gap)
            return ranges, by_outline
        # Fallback to heuristics
        by_text = text_headings(doc)
        ranges = consolidate_ranges(by_text, doc.page_count, min_gap=min_gap)
        return ranges, by_text
    finally:
        doc.close()

def main():
    ap = argparse.ArgumentParser(description="Extract learning chapter start/end page ranges from a PDF textbook.")
    ap.add_argument("pdf", type=Path, help="Path to the input PDF")
    ap.add_argument("--out", type=Path, default=None, help="Path to save JSON output (optional)")
    idx = ap.add_mutually_exclusive_group()
    idx.add_argument("--one-index", action="store_true", help="Use 1-based page numbers in output")
    idx.add_argument("--zero-index", action="store_true", help="Use 0-based page numbers in output (default)")
    ap.add_argument("--min-gap", type=int, default=2, help="Minimum pages between chapter starts (default: 2)")
    ap.add_argument("--save-debug", action="store_true", help="Save a TSV of detected headings for inspection")
    args = ap.parse_args()

    one_index = args.one_index and not args.zero_index

    ranges, starts_used = detect_chapters(args.pdf, min_gap=args.min_gap)

    # Keep only learning-topic ranges and renumber
    learning_ranges = filter_learning_ranges(ranges)
    result = build_output(learning_ranges, one_index=one_index)

    if args.save_debug:
        debug_path = (args.out.with_suffix(".tsv") if args.out else args.pdf.with_suffix(".headings.tsv"))
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("page_index\tconfidence\tnumber\traw_text\ttitle\tis_learning\n")
            for h in starts_used:
                f.write(f"{h.page_index}\t{h.confidence:.2f}\t{h.number or ''}\t{h.raw_text}\t{h.title}\t{is_learning_topic(h.title)}\n")
        print(f"Saved debug headings: {debug_path}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
