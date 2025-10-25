import os
import uuid
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg.errors import UniqueViolation

# ---------- Optional (AI extraction) ----------
OPENAI_OK = True
try:
    from openai import OpenAI
    from pypdf import PdfReader
except Exception:
    OPENAI_OK = False
    OpenAI = None
    PdfReader = None

# ===================== Config =====================
app = Flask(__name__)

# Allow local HTTP and file:// (null origin) during dev
CORS(
    app,
    resources={r"/api/*": {"origins": ["null", "http://127.0.0.1:8000", "http://localhost:8000"]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://avnadmin:AVNS_jsTSmdD8sgf0CoR9UaW@pg-27f0ba51-syamsulbakhri-27a8.g.aivencloud.com:16774/defaultdb?sslmode=require"
)
pool = ConnectionPool(conninfo=DB_DSN, min_size=1, max_size=5, timeout=10)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OPENAI_MODEL_NEW = "gpt-4.1-mini"

# Hardcoded API key (local dev)
HARDCODED_OPENAI_KEY = "API_KEY_HERE"
client = None
if OPENAI_OK:
    client = OpenAI(api_key=HARDCODED_OPENAI_KEY)


# ===================== Helpers =====================
def read_pdf_text(pdf_path: Path, max_chars: int = 60000) -> str:
    """Extract plain text from a PDF."""
    if not pdf_path.exists() or PdfReader is None:
        return ""
    parts = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
            if sum(len(x) for x in parts) > max_chars:
                break
    return "\n\n".join(parts)[:max_chars]


def normalize_titles_list(items):
    out, seen = [], set()
    for it in items:
        t = (str(it) if it else "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t[:200])
    return out[:100]


def ai_extract_contents_as_titles(text: str) -> list[str]:
    """Extracts course contents using OpenAI."""
    if not text.strip() or not client:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return normalize_titles_list(lines[:20])

    system_prompt = "Extract a clean ordered list of course contents (chapters/modules)."
    user_prompt = f"""Syllabus text:
---
{text}
---
Return JSON ONLY: {{ "contents": ["Title 1", "Title 2"] }}"""

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NEW,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception:
        return []

    import json as _json
    try:
        data = _json.loads(raw)
        titles = data.get("contents", [])
        return normalize_titles_list(titles)
    except Exception:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return normalize_titles_list(lines[:30])


# ===================== Routes =====================

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

    return jsonify({"ok": True, "url": f"/uploads/{fname}", "kind": kind})


@app.get("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname, as_attachment=False)


@app.post("/api/contents/extract")
def extract_contents():
    """Extract course contents from syllabus file."""
    data = request.get_json(silent=True) or {}
    syllabus_url = (data.get("syllabus_url") or "").strip()
    if not syllabus_url.startswith("/uploads/"):
        return jsonify({"ok": False, "error": "Invalid syllabus URL"}), 400

    file_path = UPLOAD_DIR / syllabus_url.split("/uploads/", 1)[1]
    if not file_path.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404

    text = read_pdf_text(file_path) if file_path.suffix.lower() == ".pdf" else file_path.read_text(errors="ignore")
    titles = ai_extract_contents_as_titles(text)
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
      "contents": ["Intro", "Simultaneous Games", "Sequential Games"]  # <- auto-filled titles
    }

    INSERTS:
      name          = title
      isbn          = isbn
      source_url    = textbook_url (fallback to syllabus_url, else "")
      course_contents = contents (JSON array of strings)
      chapters_json   = []   (empty list; pages handled later)
    """
    from psycopg.types.json import Json
    from psycopg.rows import dict_row

    data = request.get_json(silent=True) or {}
    title        = (data.get("title") or "").strip()
    isbn         = (data.get("isbn") or "").strip()
    textbook_url = (data.get("textbook_url") or "").strip()
    syllabus_url = (data.get("syllabus_url") or "").strip()
    contents     = data.get("contents")  # list of strings (may be [] or None)

    if not title:
        return jsonify({"ok": False, "error": "title required"}), 400

    # normalize contents (list[str]) -> de-dup, trim
    def _norm(items):
        if not isinstance(items, list):
            return []
        out, seen = [], set()
        for it in items:
            t = (str(it) if it else "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t[:200])
        return out[:100]

    titles = _norm(contents)

    source_url = textbook_url or syllabus_url or ""

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            INSERT INTO hack_books (name, isbn, source_url, chapters_json, course_contents)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, name, isbn, source_url, chapters_json, course_contents
        """, (title, isbn or None, source_url, Json([]), Json(titles)))
        row = cur.fetchone()
        conn.commit()

    return jsonify({"ok": True, "course": row})

# --- add near the top ---
from psycopg.rows import dict_row

# ---------- Save a message ----------
@app.post("/api/history")
def add_history():
    """
    Body JSON:
    {
      "book_id": 123,               # <-- hack_books.id (numeric)
      "role": "user" | "bot",
      "content": "Hello world"
    }
    """
    data = request.get_json(silent=True) or {}
    book_id = data.get("book_id")
    role    = (data.get("role") or "user").strip().lower()
    content = (data.get("content") or "").strip()

    if not isinstance(book_id, (int, float)) or int(book_id) <= 0:
        return jsonify({"ok": False, "error": "valid numeric book_id required"}), 400
    if not content:
        return jsonify({"ok": False, "error": "content required"}), 400
    if role not in ("user", "bot"):
        role = "user"

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hack_books_history (book_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (int(book_id), role, content)
        )
        conn.commit()

    return jsonify({"ok": True})

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




# ===================== Run =====================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
