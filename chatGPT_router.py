#!/usr/bin/env python3
"""
chatgpt_literal_router_keyed.py  (Chat Completions JSON-mode version)

Route a student question to relevant textbook chapters using ChatGPT ONLY.
No embeddings. The model receives literal text (chapters JSON + optional context)
and returns ONLY the matching subset JSON.

Run:
  python chatGPT_router.py --chapters chapters.json --question "How does TCP handle congestion control?"
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------
# PASTE YOUR API KEY HERE ↓↓↓
OPENAI_API_KEY = "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"
# ---------------------------------------------------------------------

# Choose a model that supports JSON mode on Chat Completions:
MODEL = "gpt-4o-mini-2024-07-18"   # also works: "gpt-4o-2024-08-06"

SYSTEM_INSTRUCTIONS = """You route student questions to relevant textbook chapters.
You are given:
  • A chapters JSON mapping of Chapter_* -> {topic,start,end}
  • Optional additional literal context text
  • A student question
Return ONLY a JSON object that is a *subset* of the chapters mapping, containing
ONLY the most relevant chapters for answering the question.
Do NOT invent chapters or change any fields. Do NOT include explanations or extra keys.
Output JSON only.
"""

def _allowed_keys_prompt(keys: List[str]) -> str:
    # Provide a compact allowlist to the model to reduce mistakes.
    # (JSON Schema enforcement would be nicer, but JSON mode is sufficient here.)
    return (
        "Only use these chapter keys (if relevant): "
        + ", ".join(keys)
        + ". Do not add any keys that are not in this list."
    )

class ChatGPTLiteralRouter:
    def __init__(self, api_key: str, model: str = MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def route(
        self,
        question: str,
        chapters: Dict[str, Dict[str, Any]],
        *,
        top_k: int = 3,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        allowed_keys = list(chapters.keys())
        allowlist = _allowed_keys_prompt(allowed_keys)

        context_block = f"\nAdditional context text:\n{extra_context}\n" if extra_context else ""

        user_prompt = f"""
Question: {question}

You may return up to {top_k} chapters. If nothing clearly fits, return an empty object {{}}.
Prioritize high-precision matches. If multiple chapters are plausible, choose the best {top_k}.
{allowlist}

Chapters JSON (copy matching entries EXACTLY as-is; do not alter values):
{json.dumps(chapters, ensure_ascii=False)}

{context_block}
""".strip()

        # --- Chat Completions with JSON mode (supported in openai==2.6.1) ---
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},  # force JSON-only output
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = resp.choices[0].message.content
        result = json.loads(content) if content else {}
        if not isinstance(result, dict):
            raise ValueError("Model did not return a JSON object.")
        return result


# --- Example CLI usage ---
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Route a question to chapters using ChatGPT (literal text only).")
    parser.add_argument("--chapters", type=str, required=True, help="Path to chapters.json (Chapter_* -> {topic,start,end})")
    parser.add_argument("--question", type=str, required=True, help="Student question")
    parser.add_argument("--top-k", type=int, default=3, help="Max number of chapters to return")
    parser.add_argument("--extra-context", type=str, default=None, help="Optional path to a text file with extra literal context")
    parser.add_argument("--model", type=str, default=MODEL, help="OpenAI model")
    args = parser.parse_args()

    with open(args.chapters, "r", encoding="utf-8") as f:
        chapters_map = json.load(f)

    extra_context = None
    if args.extra_context:
        try:
            with open(args.extra_context, "r", encoding="utf-8") as ef:
                extra_context = ef.read()
        except Exception as e:
            print(f"Warning: could not read extra context file: {e}", file=sys.stderr)

    router = ChatGPTLiteralRouter(api_key=OPENAI_API_KEY, model=args.model)
    filtered = router.route(args.question, chapters_map, top_k=args.top_k, extra_context=extra_context)

    print(json.dumps(filtered, ensure_ascii=False, indent=2))
