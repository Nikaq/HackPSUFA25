from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
import os

# Choose a model that supports JSON mode on Chat Completions
_DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

_SYSTEM_INSTRUCTIONS = """You route student questions to relevant textbook chapters.
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
    return (
        "Only use these chapter keys (if relevant): "
        + ", ".join(keys)
        + ". Do not add any keys that are not in this list."
    )

def route_chapters(
    question: str,
    chapters_list: List[Dict[str, Any]],
    *,
    top_k: int = 3,
    extra_context: Optional[str] = None,
    model: str = _DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Route a question to relevant chapters using Chat Completions JSON mode.

    Parameters
    ----------
    question : str
        The student's question.
    chapters_list : List[Dict[str, Any]]
        A list of JSON-like chapter objects. Each item should represent a chapter's data
        (e.g., {"topic": "...", "start": 12, "end": 34}). If an item contains a "key"
        field (e.g., "Chapter_1"), that will be used as the chapter's key; otherwise
        keys will be auto-generated as "Chapter_1", "Chapter_2", ...
    top_k : int, optional
        Maximum number of chapters to return (default 3).
    extra_context : str, optional
        Additional literal context to include for routing (default None).
    model : str, optional
        OpenAI model name that supports JSON mode (default: gpt-4o-mini-2024-07-18).
    api_key : str, optional
        OpenAI API key. If omitted, will use the OPENAI_API_KEY environment variable.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A subset mapping {chapter_key: chapter_data} selected by the model.
    """
    if not isinstance(chapters_list, list):
        raise TypeError("chapters_list must be a list of JSON objects (dicts).")

    # Normalize list -> mapping with stable keys that the model can reference.
    chapters_map: Dict[str, Dict[str, Any]] = {}
    auto_idx = 1
    for item in chapters_list:
        if not isinstance(item, dict):
            raise TypeError("Each chapter must be a dict/JSON object.")
        key = item.get("key")
        if not isinstance(key, str) or not key.strip():
            key = f"Chapter_{auto_idx}"
            auto_idx += 1
        # Store a copy WITHOUT the "key" field (to match the original mapping schema)
        value = {k: v for k, v in item.items() if k != "key"}
        chapters_map[key] = value

    # If nothing to route, exit early
    if not chapters_map:
        return {}

    allowlist = _allowed_keys_prompt(list(chapters_map.keys()))
    context_block = f"\nAdditional context text:\n{extra_context}\n" if extra_context else ""

    user_prompt = f"""
Question: {question}

You may return up to {top_k} chapters. If nothing clearly fits, return an empty object {{}}.
Prioritize high-precision matches. If multiple chapters are plausible, choose the best {top_k}.
{allowlist}

Chapters JSON (copy matching entries EXACTLY as-is; do not alter values):
{json.dumps(chapters_map, ensure_ascii=False)}

{context_block}
""".strip()

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},  # force JSON-only output
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = resp.choices[0].message.content
    result = json.loads(content) if content else {}

    if not isinstance(result, dict):
        raise ValueError("Model did not return a JSON object.")

    # Safety: ensure the model only returned allowed keys with exact values.
    # (If the model somehow altered values, we overwrite them with originals.)
    filtered: Dict[str, Dict[str, Any]] = {}
    for k in result.keys():
        if k in chapters_map:
            filtered[k] = chapters_map[k]

    # If the model returned more than top_k, truncate deterministically by key order
    if len(filtered) > top_k:
        filtered = {k: filtered[k] for k in list(filtered.keys())[:top_k]}

    return filtered


if __name__ == "__main__":
    
    import json

    # Your chapters JSON string
    chapters_json = '''
    {
    "Chapter_1": {"topic": "cover", "start": 0, "end": 1},
    "Chapter_2": {"topic": "title page", "start": 2, "end": 3},
    "Chapter_3": {"topic": "about the authors", "start": 4, "end": 5},
    "Chapter_4": {"topic": "dedication", "start": 6, "end": 7},
    "Chapter_5": {"topic": "preface", "start": 8, "end": 18},
    "Chapter_6": {"topic": "acknowledgments for the global edition", "start": 19, "end": 21},
    "Chapter_7": {"topic": "table of contents", "start": 22, "end": 31},
    "Chapter_8": {"topic": "computer networks and the internet", "start": 32, "end": 111},
    "Chapter_9": {"topic": "application layer", "start": 112, "end": 211},
    "Chapter_10": {"topic": "transport layer", "start": 212, "end": 333},
    "Chapter_11": {"topic": "the network layer: data plane", "start": 334, "end": 407},
    "Chapter_12": {"topic": "the network layer: control plane", "start": 408, "end": 479},
    "Chapter_13": {"topic": "the link layer and lans", "start": 480, "end": 561},
    "Chapter_14": {"topic": "wireless and mobile networks", "start": 562, "end": 637},
    "Chapter_15": {"topic": "security in computer networks", "start": 638, "end": 721},
    "Chapter_16": {"topic": "references", "start": 722, "end": 761},
    "Chapter_17": {"topic": "index", "start": 762, "end": 796}
    }
    '''

    # Parse the JSON string into a Python dict
    chapters_dict = json.loads(chapters_json)

    # Convert it into a list of dicts (each item has a "key" so route_chapters() can identify chapters)
    chapters_list = [{"key": key, **value} for key, value in chapters_dict.items()]

    # Now you can call the function (assuming you imported it)
    result = route_chapters(
        question="How does TCP handle congestion control?",
        chapters_list=chapters_list,
        top_k=2,
        api_key="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",
        model="gpt-4o-2024-08-06"
    )

    print(json.dumps(result, indent=2))
