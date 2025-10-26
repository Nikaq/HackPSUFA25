import os
import sys
import uuid
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure we can import chatGPT_router2.py if it's alongside this file
sys.path.append(str(Path(__file__).parent))

# ---- your extractor for selected chapters (writes .txt files) ----
from pdf_text_extractor import extract_selected_chapters

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json

# ---- PDF libraries (pypdf for simple text extraction; fitz optional elsewhere) ----
import fitz  # PyMuPDF (used for page-range reads if needed)
from pypdf import PdfReader

# ---- OpenAI ----
from openai import OpenAI

# ---- Router imported from chatGPT_router2.py ----
from chatGPT_router_2 import route_chapters

# ================================================================
#                       CONFIG / GLOBALS
# ================================================================
BASE_DIR = Path.cwd().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://avnadmin:AVNS_jsTSmdD8sgf0CoR9UaW@pg-27f0ba51-syamsulbakhri-27a8.g.aivencloud.com:16774/defaultdb?sslmode=require",
)
pool = ConnectionPool(conninfo=DB_DSN, min_size=1, max_size=5, timeout=10)

# OpenAI
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "API_KEY_HERE").strip()
OPENAI_OK = bool(OPENAI_API_KEY)
OPENAI_MODEL_NEW = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")            # used for QA answers
OPENAI_JSON_MODEL = os.getenv("OPENAI_JSON_MODEL", "gpt-4o-mini-2024-07-18") # used for JSON extraction
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_OK else None

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["null", "http://127.0.0.1:8000", "http://localhost:8000"]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

print("BASE_DIR:", BASE_DIR)
print("UPLOAD_DIR:", UPLOAD_DIR)
print("OPENAI_OK:", OPENAI_OK)

# ================================================================
#               CHAPTER GENERATION (PDF via your function)
# ================================================================
try:
    # Your generator returns dict: {Chapter_i: {topic,start,end}}
    from pdf_range_finder2 import generate_chapters_json
except Exception:
    def generate_chapters_json(pdf_path: str | Path, *, one_index: bool = False, min_gap: int = 2, save_debug: bool = False):
        # Fallback no-op if your module isn't importable
        return {}

# ================================================================
#                          HELPERS
# ================================================================
def read_pdf_text(pdf_path: Path, max_chars: int = 60000) -> str:
    """
    Simple text extraction using pypdf only.
    If the PDF is scanned / image-only, this will likely return empty text.
    """
    if not pdf_path.exists():
        return ""
    parts = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
            if sum(len(x) for x in parts) > max_chars:
                break
    print(pdf_path)
    os.remove(pdf_path)
    return "\n\n".join(parts)[:max_chars]

def normalize_titles_list(items):
    out, seen = [], set()
    for it in items or []:
        t = (str(it) if it else "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t[:200])
    return out[:100]

# ------------------- OpenAI syllabus → contents (JSON mode) -------------------
def _normalize_titles_list(items):
    out, seen = [], set()
    for it in items or []:
        t = (str(it) if it else "").strip()
        if not t:
            continue
        lt = t.lower()
        if lt in seen:
            continue
        seen.add(lt)
        out.append(t[:200])
    return out[:100]

def _chunk(text: str, max_chars: int = 12000) -> List[str]:
    text = text or ""
    chunks = []
    i = 0
    n = len(text)
    while i < n and len(chunks) < 8:  # hard cap
        chunks.append(text[i:i+max_chars])
        i += max_chars
    return chunks

def _extract_titles_from_chunk(chunk_text: str) -> List[str]:
    """
    Calls OpenAI in JSON mode and returns a list[str] for one chunk.
    """
    if not client or not OPENAI_OK or not chunk_text.strip():
        return []

    system_prompt = (
        "Extract a clean, ordered list of course contents (chapters/modules/units) "
        "from the provided syllabus text. Ignore grading policies, schedules, policies, "
        "and instructor bios. Keep each item concise."
    )
    user_prompt = f"""Syllabus text:
---
{chunk_text}
---
Return JSON ONLY in this schema:
{{ "contents": ["Title 1", "Title 2", "Title 3"] }}"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_JSON_MODEL,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        titles = data.get("contents", [])
        return _normalize_titles_list(titles)
    except Exception:
        traceback.print_exc()
        return []

def ai_extract_contents_as_titles(text: str) -> list[str]:
    """
    Robust extractor:
    - Splits long syllabi into chunks
    - Calls OpenAI JSON mode per chunk
    - De-dupes/merges results
    - Falls back to naive line-pick if OpenAI unavailable or empty
    """
    if not text.strip():
        return []

    if not client or not OPENAI_OK:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return _normalize_titles_list(lines[:30])

    merged: List[str] = []
    for ch in _chunk(text, max_chars=12000):
        titles = _extract_titles_from_chunk(ch)
        for t in titles:
            if t and all(t.lower() != m.lower() for m in merged):
                merged.append(t)
        if len(merged) >= 100:
            break

    if not merged:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        merged = _normalize_titles_list(lines[:30])

    return merged

# ================================================================
#                   PAGE & PATH HELPERS
# ================================================================
def _resolve_pdf_path(source_url: str) -> Path:
    """
    Turn source_url into an absolute file path.
    - If it starts with '/uploads/', we look inside UPLOAD_DIR.
    - Otherwise treat as absolute path.
    """
    if not source_url:
        return Path("")
    if source_url.startswith("/uploads/"):
        return UPLOAD_DIR / source_url.split("/uploads/", 1)[1]
    return Path(source_url)

# --- sanitize for filenames used by pdf_text_extractor-generated files ---
_SAN_RE = re.compile(r'[^A-Za-z0-9_]')
def _sanitize_filename(s: str) -> str:
    return _SAN_RE.sub('_', s or "")

def _collect_extracted_texts(selected: Dict[str, Dict[str, Any]]) -> str:
    """
    Reads back the .txt files created by extract_selected_chapters and concatenates.
    Filenames: f"{ChapterKey}_{sanitize(topic)}.txt"
    """
    parts: List[str] = []
    for chapter_key, info in selected.items():
        topic = info.get("topic", "")
        fname = f"{chapter_key}_{_sanitize_filename(topic)}.txt"
        p = Path(fname)
        if p.exists():
            try:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
                os.remove(p)
            except Exception:
                continue
    return "\n\n".join(parts).strip()

# ================================================================
#           OPENAI ANSWER using your EXACT TEMPLATE
# ================================================================
def fetch_formatted_history(book_id: int, limit: int = 5) -> str:
    """
    Returns a single string like:
      AI: content
      User: content
      ...
    using the last `limit` messages for this book, in chronological order.
    """
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT role, content
            FROM hack_books_history
            WHERE book_id = %s
            ORDER BY "timestamp" DESC, id DESC
            LIMIT %s
            """,
            (int(book_id), int(limit))
        )
        rows: List[dict] = cur.fetchall()

    rows.reverse()  # chronological

    def label(role: str) -> str:
        return "AI" if (role or "").lower() == "bot" else "User"

    lines = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        oneline = " ".join(content.split())
        lines.append(f"{label(r.get('role'))}: {oneline}")

    return "\n".join(lines) if lines else ""

def ai_answer_with_template(question: str, textbook_content: str, memory: str) -> str:
    """
    Uses OpenAI to answer with the exact template:
    Chat History: <memory>
    Question: <question>
    Textbook Content:
    <textbook_content>
    """
    if not OPENAI_OK or not client:
        return ""
    prompt = f"""
Chat History:
{memory}

Question: {question}

Textbook Content:
{textbook_content}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NEW,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful tutor. Answer only using the provided Textbook Content. "
                        "If the Textbook Content does not contain enough information, say so explicitly. "
                        "Cite page numbers when you can infer them from the provided content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        traceback.print_exc()
        return ""

# ================================================================
#        PROGRESS HELPERS (keyword overlap + basic proportional math)
# ================================================================
_STOPWORDS = {
    "the","a","an","and","or","of","to","for","in","on","at","by","with","from",
    "is","are","was","were","be","been","it","this","that","these","those",
    "as","about","into","over","under","between","within","without","not",
    "how","what","why","when","where","which","who","whom","whose","you","your",
    "i","we","they","he","she","them","us","our","their"
}
_WORD_RE = re.compile(r"[a-z0-9]+")

def _keywords(text: str) -> set[str]:
    """Lowercase alphanum tokens minus common stopwords; length >= 2."""
    if not text:
        return set()
    toks = _WORD_RE.findall(text.lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) >= 2}

def _allocate_and_update_progress(
    cur,
    *,
    book_id: int,
    question: str,
    routed: Dict[str, Dict[str, Any]],
    chapters_json_list: List[Dict[str, Any]],
) -> None:
    """
    Update hack_books.course_contents using only basic math:

    1) covered_percent = sum of chapter 'percent' for the chapters we just taught.
    2) Build a keyword signal from selected chapter topics + the user's question.
    3) Score each course content title by keyword overlap with the signal.
    4) Distribute 'covered_percent' proportionally to titles with score > 0.
    5) Cap each 'progress' at 100 and round to 2 decimals.

    Only runs if there *was* a bot reply (caller ensures this).
    """
    if not routed:
        return

    selected_keys = list(routed.keys())
    if not selected_keys:
        return

    # Map chapter key -> percent
    percent_map: Dict[str, float] = {}
    for ch in chapters_json_list or []:
        k = ch.get("chapter")
        if k:
            try:
                percent_map[k] = float(ch.get("percent", 0) or 0.0)
            except Exception:
                percent_map[k] = 0.0

    covered_percent = sum(percent_map.get(k, 0.0) for k in selected_keys)
    if covered_percent <= 0:
        return

    # Load current contents
    cur.execute("SELECT course_contents FROM hack_books WHERE id = %s", (int(book_id),))
    row = cur.fetchone()
    contents: List[Dict[str, Any]] = (row.get("course_contents") or []) if row else []
    if not contents:
        return

    # Build the keyword signal from selected chapter topics + the question
    chap_topics = " ".join([routed[k].get("topic", "") for k in selected_keys if k in routed])
    signal = _keywords(chap_topics) | _keywords(question)

    # Score each content item
    scores: List[int] = []
    for item in contents:
        title = (item.get("Content") or "").strip()
        score = len(_keywords(title) & signal)
        scores.append(score)

    total_score = sum(scores)
    if total_score <= 0:
        # Nothing matched; skip updating to avoid miscrediting unrelated content
        return

    # Distribute covered_percent proportionally by score
    for i, item in enumerate(contents):
        sc = scores[i]
        if sc <= 0:
            continue
        add = covered_percent * (sc / total_score)
        try:
            current = float(item.get("progress", 0) or 0.0)
        except Exception:
            current = 0.0
        item["progress"] = round(min(100.0, current + add), 2)

    # Write back
    cur.execute(
        "UPDATE hack_books SET course_contents = %s WHERE id = %s",
        (Json(contents), int(book_id))
    )

# ================================================================
#                            ROUTES
# ================================================================
@app.get("/api/books/all")
def get_all():
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM hack_books ORDER BY id')
        return jsonify(cur.fetchall())

@app.post("/api/upload")
def upload_file():
    """Handles file uploads for syllabus/textbook."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "missing file"}), 400

    kind = (request.form.get("kind") or "file").strip().lower()
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "empty filename"}), 400

    ext = Path(f.filename).suffix
    fname = f"{kind}-{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / fname
    f.save(dest)
    print("Saved upload:", dest)

    return jsonify({"ok": True, "url": f"/uploads/{fname}", "kind": kind})

@app.get("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname, as_attachment=False)

@app.post("/api/contents/extract")
def extract_contents():
    """
    Extract course contents (titles only) from syllabus file using:
    - read_pdf_text (pypdf) or plain text (for .txt)
    - OpenAI JSON mode for robust list extraction
    """
    data = request.get_json(silent=True) or {}
    syllabus_url = (data.get("syllabus_url") or "").strip()
    if not syllabus_url.startswith("/uploads/"):
        return jsonify({"ok": False, "error": "Invalid syllabus URL"}), 400

    file_path = UPLOAD_DIR / syllabus_url.split("/uploads/", 1)[1]
    if not file_path.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404

    if file_path.suffix.lower() == ".pdf":
        text = read_pdf_text(file_path)  # pypdf-only
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

    titles = ai_extract_contents_as_titles(text)
    if not titles and not text.strip():
        return jsonify({"ok": True, "contents": [], "hint": "Syllabus seems to have no extractable text."})
    return jsonify({"ok": True, "contents": titles})

# ===================== INSERT on Save Changes =====================
@app.post("/api/courses/save")
def create_course():
    """
    Body JSON:
    {
      "title": "CMPSC 443",
      "isbn": "9781598295931",
      "textbook_url": "/uploads/textbook.pdf",   # optional
      "syllabus_url": "/uploads/syllabus.pdf",   # optional
      "contents": ["Intro", ...]  # front-end preview; backend stores as [{Content,progress}]
    }

    INSERTS:
      name            = title
      isbn            = isbn
      source_url      = textbook_url (fallback to syllabus_url, else "")
      course_contents = [{Content, progress}]  (JSON array)
      chapters_json   = list of chapter dicts
    """
    data = request.get_json(silent=True) or {}
    title        = (data.get("title") or "").strip()
    isbn         = (data.get("isbn") or "").strip()
    textbook_url = (data.get("textbook_url") or "").strip()
    syllabus_url = (data.get("syllabus_url") or "").strip()
    contents     = data.get("contents")  # list[str] from UI

    if not title:
        return jsonify({"ok": False, "error": "title required"}), 400

    # Store contents as [{Content, progress:0}]
    def _to_content_objs(items):
        titles = normalize_titles_list(items)
        return [{"Content": t, "progress": 0} for t in titles]

    content_objs = _to_content_objs(contents)
    source_url = textbook_url or syllabus_url or ""

    # Build chapters_json (list) if textbook exists
    chapters_list = []
    abs_pdf = _resolve_pdf_path(source_url)
    if abs_pdf and abs_pdf.exists() and abs_pdf.suffix.lower() == ".pdf":
        # IMPORTANT: 0-based pages to match pypdf PdfReader indexing used in pdf_text_extractor
        chapters_dict = generate_chapters_json(abs_pdf, one_index=False, min_gap=2, save_debug=False)
        # Convert dict -> list with your desired structure (title,start,end,percent)
        total_pages = sum(max(0, ch["end"] - ch["start"] + 1) for ch in chapters_dict.values()) or 1
        for key, value in chapters_dict.items():
            start, end = value.get("start", 0), value.get("end", 0)
            pages = max(0, end - start + 1)
            percent = round((pages / total_pages) * 100, 2)
            chapters_list.append({
                "chapter": key,
                "title": value.get("topic", ""),
                "start": start,   # 0-based
                "end": end,       # 0-based
                "percent": percent
            })

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            INSERT INTO hack_books (name, isbn, source_url, chapters_json, course_contents)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, name, isbn, source_url, chapters_json, course_contents
        """, (title, isbn or None, source_url, Json(chapters_list), Json(content_objs)))
        row = cur.fetchone()
        conn.commit()

    return jsonify({"ok": True, "course": row})

# ---------- Save a message & auto-reply ----------
@app.post("/api/history")
def add_history():
    """
    Body JSON:
    {
      "book_id": 123,               # <-- hack_books.id
      "role": "user" | "bot",
      "content": "How does TCP slow start work?",
      "textbook_content": "...OPTIONAL manual paste to override routing..."
    }
    """
    data = request.get_json(silent=True) or {}
    book_id = data.get("book_id")
    role    = (data.get("role") or "user").strip().lower()
    content = (data.get("content") or "").strip()
    manual_textbook_content = None

    if not isinstance(book_id, (int, float)) or int(book_id) <= 0:
        return jsonify({"ok": False, "error": "valid numeric book_id required"}), 400
    if not content:
        return jsonify({"ok": False, "error": "content required"}), 400
    if role not in ("user", "bot"):
        role = "user"

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Save the user (or bot) message and return full row (incl. timestamp)
        cur.execute(
            """
            INSERT INTO hack_books_history (book_id, role, content)
            VALUES (%s, %s, %s)
            RETURNING id, book_id, role, content, "timestamp"
            """,
            (int(book_id), role, content)
        )
        saved_user_or_bot = cur.fetchone()

        # Only auto-reply when a user sends a message
        bot_msg = None
        routed = {}  # keep routed visible after answering
        chapters_json_list = []
        source_url = ""

        if role == "user":
            # Load chapters_json + source_url
            cur.execute(
                "SELECT chapters_json, source_url FROM hack_books WHERE id = %s",
                (int(book_id),)
            )
            row = cur.fetchone()
            if row:
                chapters_json_list = row.get("chapters_json") or []
                source_url = row.get("source_url") or ""

            # If caller provided manual content, use it
            if manual_textbook_content:
                textbook_content = manual_textbook_content
            else:
                # Prepare router input (list -> expected by router)
                chapters_list_for_router = [
                    {
                        "key": ch.get("chapter") or f"Chapter_{i+1}",
                        "topic": ch.get("title", ""),
                        "start": ch.get("start"),
                        "end": ch.get("end"),
                    }
                    for i, ch in enumerate(chapters_json_list)
                    if ch and ch.get("start") is not None and ch.get("end") is not None
                ]

                # 1) Route to the most relevant chapters using chatGPT_router2.py
                routed = route_chapters(
                    question=content,
                    chapters_list=chapters_list_for_router,
                    top_k=3,
                    api_key=OPENAI_API_KEY,
                    model="gpt-4o-2024-08-06"
                )  # dict of {Chapter_X: {topic,start,end}}
                print(chapters_json_list)
                print(routed)

                # 2) Convert list (stored) -> dict mapping (what extractor expects)
                chapters_map_for_extractor: Dict[str, Dict[str, Any]] = {}
                for ch in chapters_json_list:
                    key = ch.get("chapter")
                    if not key:
                        continue
                    chapters_map_for_extractor[key] = {
                        "topic": ch.get("title", ""),
                        "start": int(ch.get("start", 0)),  # already 0-based from save()
                        "end": int(ch.get("end", 0)),
                    }
                chapters_json_str = json.dumps(chapters_map_for_extractor, ensure_ascii=False)

                # 3) Build selected chapter keys (order as in 'routed')
                selected_keys = list(routed.keys())  # e.g., ["Chapter_12", "Chapter_9"]

                # 4) Resolve PDF and call your extractor (writes .txt files),
                #    then read those files back
                extracted = ""
                pdf_path = _resolve_pdf_path(source_url)
                if pdf_path and pdf_path.exists() and selected_keys:
                    try:
                        extract_selected_chapters(str(pdf_path), chapters_json_str, selected_keys)
                        header_lines = [f"{k}: {routed[k].get('topic','')} (pages {routed[k].get('start')}-{routed[k].get('end')})"
                                        for k in selected_keys if k in routed]
                        header_text = "Selected Chapters:\n" + ("\n".join(header_lines) if header_lines else "None")
                        extracted_text = _collect_extracted_texts({k: routed[k] for k in selected_keys if k in routed})
                        textbook_content = f"{header_text}\n\n{extracted_text}".strip()
                    except Exception:
                        traceback.print_exc()
                        textbook_content = "Selected chapters could not be extracted."
                else:
                    textbook_content = "No matching chapters or missing PDF."

            # Answer with your strict template
            ai_text = ai_answer_with_template(content, textbook_content, fetch_formatted_history(book_id))
            if ai_text:
                cur.execute(
                    """
                    INSERT INTO hack_books_history (book_id, role, content)
                    VALUES (%s, %s, %s)
                    RETURNING id, book_id, role, content, "timestamp"
                    """,
                    (int(book_id), "bot", ai_text)
                )
                saved_bot = cur.fetchone()
                bot_msg = saved_bot  # contains DB-generated unique id

                # >>> ONLY update progress when the LLM responded <<<
                try:
                    _allocate_and_update_progress(
                        cur,
                        book_id=int(book_id),
                        question=content,
                        routed=routed,
                        chapters_json_list=chapters_json_list
                    )
                except Exception:
                    traceback.print_exc()

        conn.commit()

    # If user sent a message: saved_user_or_bot is the user row, else it's a bot row
    return jsonify({
        "ok": True,
        "saved": saved_user_or_bot,  # includes id + timestamp
        "bot": bot_msg               # includes id + timestamp, or null if no auto reply
    })

# ---------- Load messages for a course ----------
@app.get("/api/history")
def get_history():
    """
    Query: /api/history?book_id=123&limit=200
    Returns rows sorted by timestamp then id.
    """
    try:
        book_id = int(request.args.get("book_id", "0"))
    except ValueError:
        return jsonify({"ok": False, "error": "book_id must be numeric"}), 400
    if book_id <= 0:
        return jsonify({"ok": False, "error": "book_id required"}), 400

    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 500))

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, book_id, role, content, "timestamp"
            FROM hack_books_history
            WHERE book_id = %s
            ORDER BY "timestamp" ASC, id ASC
            LIMIT %s
            """,
            (book_id, limit)
        )
        rows = cur.fetchall()

    # ISO timestamps for the frontend
    for r in rows:
        ts = r.get("timestamp")
        if hasattr(ts, "isoformat"):
            r["timestamp"] = ts.isoformat()
        elif ts is not None:
            r["timestamp"] = str(ts)

    return jsonify({"ok": True, "rows": rows})

# ================================================================
#                            RUN
# ================================================================
if __name__ == "__main__":
    # Ensure DB connectivity at startup (optional)
    try:
        with pool.connection() as _c:
            pass
    except Exception as e:
        print("WARNING: could not connect to DB at startup:", e)

    app.run(host="127.0.0.1", port=5001, debug=True)
