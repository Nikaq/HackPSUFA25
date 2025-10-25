#!/usr/bin/env python3
"""
End-to-end study pipeline:

1) Run pdf_range_finder to build chapters JSON from a PDF.
2) Run ChatGPT router on that JSON + a question to pick relevant chapter(s).
3) Run pdf_text_extractor to dump selected chapters' text to .txt files.

Usage:
  python study_pipeline.py --pdf ComputerNetworking.pdf \
      --question "How does TCP handle congestion control?" \
      --chapters-out chapters.json \
      --top-k 2
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path

# --- Import your modules (files must be in the same folder) ---
import pdf_range_finder as prf        # step 1 (build chapters JSON)
from chatGPT_router import ChatGPTLiteralRouter  # step 2 (choose chapters)
import pdf_text_extractor as pte      # step 3 (extract text files)

# If your chatGPT_router.py stores the API key inline, you don't need it here.
# Otherwise, you can pass a key string into ChatGPTLiteralRouter(api_key="...").

def build_chapters_json(pdf_path: Path, out_json: Path, one_index: bool = False, min_gap: int = 2) -> dict:
    # Use the same internal functions your script exposes
    # 1) detect chapters
    ranges, _starts_used = prf.detect_chapters(pdf_path, min_gap=min_gap)
    # 2) keep only learning-topic ranges
    learning_ranges = prf.filter_learning_ranges(ranges)
    # 3) build numbered JSON
    result = prf.build_output(learning_ranges, one_index=one_index)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def choose_chapters(question: str, chapters_map: dict, top_k: int = 3, model: str | None = None) -> dict:
    router = ChatGPTLiteralRouter(api_key=None if False else "xxxxxxxxx",  # not used if the class has a key embedded
                                  model=model or "gpt-4o-mini-2024-07-18")
    return router.route(
        question=question,
        chapters=chapters_map,
        top_k=top_k,
        extra_context=None,   # optional: pass blurbs/summaries here
    )

def extract_texts(pdf_path: Path, chapters_map: dict, selected: dict):
    # pdf_text_extractor expects a JSON string and a list of chapter keys
    chapters_json_str = json.dumps(chapters_map, ensure_ascii=False)
    selected_keys = list(selected.keys())
    if not selected_keys:
        print("No matching chapters; nothing to extract.")
        return
    pte.extract_selected_chapters(str(pdf_path), chapters_json_str, selected_keys)

def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline: PDF → chapters.json → selected.json → .txts")
    ap.add_argument("--pdf", required=True, type=Path, help="Input textbook PDF")
    ap.add_argument("--question", required=True, type=str, help="Student question")
    ap.add_argument("--chapters-out", type=Path, default=Path("chapters.json"), help="Where to write the full chapters JSON")
    ap.add_argument("--selected-out", type=Path, default=Path("selected_chapters.json"), help="Where to write the selected subset JSON")
    ap.add_argument("--top-k", type=int, default=3, help="Max chapters to return from the router")
    ap.add_argument("--one-index", action="store_true", help="Store pages 1-indexed in chapters.json (default is 0-indexed)")
    ap.add_argument("--min-gap", type=int, default=2, help="Minimum pages between chapter starts (range finder)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI model name for the router")
    args = ap.parse_args()

    # 1) Build chapters JSON (learning topics only)
    print("Step 1/3: Building chapters.json ...")
    chapters_map = build_chapters_json(args.pdf, args.chapters_out, one_index=args.one_index, min_gap=args.min_gap)

    # 2) Use ChatGPT to choose chapter(s)
    print("Step 2/3: Selecting chapters with ChatGPT ...")
    selected = choose_chapters(args.question, chapters_map, top_k=args.top_k, model=args.model)
    args.selected_out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected chapters: {list(selected.keys()) or '[]'}")

    # 3) Extract the selected chapters to .txt files
    print("Step 3/3: Extracting selected chapters to .txt files ...")
    extract_texts(args.pdf, chapters_map, selected)

    print("Done.")

if __name__ == "__main__":
    main()
