# -*- coding: utf-8 -*-
"""
app.py — واجهة ويب محلية للمحادثة مع مستنداتك (بدون أي اعتماديات إضافية).
الاستخدام: python3 app.py  ثم افتح http://localhost:8765
"""
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import time

import chromadb
import ollama

import ingest
import query  # للوصول الديناميكي إلى الإعدادات القابلة للتغيير وقت التشغيل (النموذج، حد الصلة…)
from query import (CHAT_PROMPT, FALLBACK_PROMPT, GENERAL_PROMPT, GENERAL_STYLES, ATTACH_PROMPT,
                   NOT_FOUND_MSG, COLLECTION, DB_DIR, DEPTHS,
                   LLM_MODEL, EMBED_MODEL, MAX_DISTANCE,
                   route, select_hits, iter_tokens, _chat,
                   greeting_reply, match_score, build_answer_messages)

FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"

# ===== سجل التدقيق (audit log): من فعل ماذا ومتى — لأغراض الحوكمة والمساءلة =====
# ملاحظة: يحتوي نصوص الأسئلة، لذا يُستبعد من المستودع ويُعامَل كبيانات تشغيلية.
AUDIT_FILE = Path(__file__).parent / "audit.jsonl"
_audit_lock = threading.Lock()


def audit_log(event, **fields):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event}
    rec.update(fields)
    try:
        with _audit_lock, open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # التدقيق لا يجب أن يُعطّل الخدمة


def _g(part, key):
    """قراءة حقل من رد Ollama (يدعم dict وكائنات pydantic)."""
    try:
        return part[key] if isinstance(part, dict) else getattr(part, key, None)
    except Exception:
        return None

# تحذير مبكر وواضح إذا كانت بيئة بايثون ناقصة (بدل خطأ غامض عند الرفع)
_missing = []
for _mod, _pkg in (("pypdf", "pypdf"), ("docx", "python-docx")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print(f"⚠️ مكتبات ناقصة في بيئة بايثون الحالية: {', '.join(_missing)}")
    print("   شغّل التطبيق عبر:  bash run_app.sh  (يختار البيئة الصحيحة ويثبّت الناقص تلقائيًا)")

BASE_DIR = Path(__file__).parent
UI_FILE = BASE_DIR / "ui.html"
PORT = 8765

try:
    _client = chromadb.PersistentClient(path=str(DB_DIR))
    # get_or_create: التطبيق يعمل حتى على قاعدة فارغة (ارفع ملفاتك من الواجهة مباشرة)
    COL = _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    # مجموعة مؤقتة للمستندات المُرفقة للجلسة (تُفرَّغ عند كل تشغيل للخادم)
    ATTACH_COLLECTION = "attached_session"
    try:
        _client.delete_collection(ATTACH_COLLECTION)
    except Exception:
        pass
    COL_TMP = _client.get_or_create_collection(ATTACH_COLLECTION, metadata={"hnsw:space": "cosine"})
except Exception as e:
    sys.exit(f"تعذّر فتح قاعدة البيانات: {e}")

# ملفات مُرفقة للجلسة الحالية (مؤقتة، غير دائمة). الصغيرة تُحقن نصًا كاملًا،
# والكبيرة (> الحد) تُفهرَس في COL_TMP ويُسترجع منها.
ATTACH_INLINE_CHARS = int(os.environ.get("RAG_ATTACH_INLINE_CHARS", "12000"))
ATTACHED = {"inline": [], "temp": [], "names": []}
ATTACH_DIR = BASE_DIR / ".attach_tmp"

# لوحة الإدارة: كلمة مرور بسيطة (اضبطها عبر متغيّر البيئة RAG_ADMIN_PASSWORD)
ADMIN_PASS = os.environ.get("RAG_ADMIN_PASSWORD", "admin")
ADMIN_FILE = BASE_DIR / "admin.html"
DEFAULT_DEPTH = "balanced"  # المستوى الافتراضي (قابل للتغيير من لوحة الإدارة ضمن الجلسة)

# ===== صلاحيات الوصول: مجموعات بيانات ↔ مجموعات مستخدمين ↔ مستخدمون =====
ACCESS_FILE = BASE_DIR / "access.json"
ALL_DATA = "كل البيانات"          # مجموعة بيانات ثابتة تعني كل الملفات (وصول كامل)
USER_SESSIONS = {}                 # token -> username (جلسات المستخدمين في الذاكرة)
ADMIN_SESSIONS = set()             # توكنات جلسات الأدمن المؤقتة (تُفرَّغ عند إعادة التشغيل)

# ===== تجزئة كلمات المرور (مكتبة قياسية فقط — لا اعتمادية خارجية، يعمل دون إنترنت) =====
_PBKDF2_ROUNDS = 200_000
_HASH_PREFIX = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    """يُرجع تجزئة كلمة المرور بصيغة  pbkdf2_sha256$rounds$salt$hash (آمنة للتخزين)."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return f"{_HASH_PREFIX}${_PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """تحقّق بزمن ثابت من مطابقة كلمة المرور للتجزئة المخزّنة."""
    try:
        prefix, rounds, salt, digest = stored.split("$", 3)
        if prefix != _HASH_PREFIX:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def _is_hashed(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_HASH_PREFIX + "$")


def load_access():
    try:
        d = json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d.setdefault("data_groups", {})   # {اسم: [أسماء الملفات]}
    d.setdefault("user_groups", {})   # {اسم: {"data_groups": [أسماء مجموعات البيانات]}}
    d.setdefault("users", {})         # {اسم_المستخدم: {"password": .., "group": اسم_مجموعة_مستخدمين}}
    # ترحيل تلقائي: أي كلمة مرور مخزّنة نصاً صريحاً تُجزَّأ وتُحفَظ (لمرة واحدة، دون تغيير كلمة المرور نفسها)
    migrated = False
    for _uname, _u in d["users"].items():
        _pw = _u.get("password", "")
        if _pw and not _is_hashed(_pw):
            _u["password"] = hash_password(_pw)
            migrated = True
    if migrated:
        save_access(d)
        print("🔐 تم ترحيل كلمات المرور في access.json إلى صيغة مجزّأة (pbkdf2).")
    return d


def save_access(d):
    ACCESS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def user_allowed_sources(username):
    """None = وصول كامل (كل البيانات). set = أسماء الملفات المسموحة. set فارغة = لا وصول.

    ملاحظة دمج Azure: هذه الدالة تُقرِّر الصلاحيات انطلاقاً من مجموعة المستخدم.
    عند التكامل مع Entra ID تبقى كما هي — فقط مرِّر اسم المستخدم/المجموعة القادم
    من مطالبات توكن Azure بدل قراءته من access.json المحلي.
    """
    acc = load_access()
    u = acc["users"].get(username)
    if not u:
        return set()
    grp = acc["user_groups"].get(u.get("group", ""), {})
    dgs = grp.get("data_groups", []) if isinstance(grp, dict) else []
    if ALL_DATA in dgs:
        return None
    allowed = set()
    for dg in dgs:
        allowed.update(acc["data_groups"].get(dg, []))
    return allowed


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):  # إسكات سجلّ الطلبات
        pass

    def _headers(self, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._headers("text/html; charset=utf-8")
            self.wfile.write(UI_FILE.read_text(encoding="utf-8").encode("utf-8"))
        elif url.path == "/ask":
            params = parse_qs(url.query)
            q = params.get("q", [""])[0].strip()
            depth = params.get("depth", [DEFAULT_DEPTH])[0]
            if depth not in DEPTHS:
                depth = DEFAULT_DEPTH
            # مصدر الإجابة: "docs" (بحث في المستندات - افتراضي) أو "general" (معرفة عامة)
            source = params.get("source", ["docs"])[0]
            # هوية المستخدم لتطبيق صلاحيات الوصول
            token = params.get("token", [""])[0]
            username = USER_SESSIONS.get(token)
            self._headers("text/event-stream; charset=utf-8")
            if not username:
                self.wfile.write(sse("err", {"msg": "انتهت الجلسة — سجّل الدخول من جديد."}))
                return
            if not q:
                self.wfile.write(sse("err", {"msg": "سؤال فارغ"}))
                return
            allowed = user_allowed_sources(username)
            audit_log("ask", user=username, q=q, depth=depth, source=source)
            try:
                self._answer(q, depth, source, allowed=allowed)
            except (BrokenPipeError, ConnectionResetError):
                pass  # المستخدم أغلق الصفحة
            except Exception as e:
                try:
                    self.wfile.write(sse("err", {"msg": str(e)}))
                except Exception:
                    pass
        elif url.path == "/info":
            self._headers("application/json; charset=utf-8")
            self.wfile.write(json.dumps(self._system_info(), ensure_ascii=False).encode("utf-8"))
        elif url.path == "/admin":
            self._headers("text/html; charset=utf-8")
            self.wfile.write(ADMIN_FILE.read_text(encoding="utf-8").encode("utf-8"))
        elif url.path == "/admin/api/ask":
            # منصة تجربة النموذج (SSE) — التحقق عبر توكن جلسة مؤقت (EventSource بلا ترويسات)
            # التوكن يُصدَر من /admin/login، فلا تظهر كلمة المرور في الرابط أو السجلات.
            params = parse_qs(url.query)
            self._headers("text/event-stream; charset=utf-8")
            if params.get("token", [""])[0] not in ADMIN_SESSIONS:
                self.wfile.write(sse("err", {"msg": "غير مصرّح"}))
                return
            q = params.get("q", [""])[0].strip()
            depth = params.get("depth", [DEFAULT_DEPTH])[0]
            if depth not in DEPTHS:
                depth = DEFAULT_DEPTH
            source = params.get("source", ["docs"])[0]
            dataset = params.get("dataset", [ALL_DATA])[0]
            if not q:
                self.wfile.write(sse("err", {"msg": "سؤال فارغ"}))
                return
            allowed = None if dataset == ALL_DATA else set(load_access()["data_groups"].get(dataset, []))
            try:
                self._answer(q, depth, source, allowed=allowed)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.wfile.write(sse("err", {"msg": str(e)}))
                except Exception:
                    pass
        elif url.path.startswith("/admin/api/"):
            self._admin_get(url.path, parse_qs(url.query))
        else:
            self._headers("text/plain", 404)

    def _system_info(self):
        per = {}
        try:
            for m in COL.get(include=["metadatas"])["metadatas"]:
                per[m["source"]] = per.get(m["source"], 0) + 1
        except Exception:
            pass
        models = []
        try:
            resp = ollama.list()
            items = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])
            for m in items or []:
                name = (m.get("model") if isinstance(m, dict)
                        else getattr(m, "model", None) or getattr(m, "name", None))
                if name:
                    models.append(str(name))
        except Exception:
            pass
        return {
            "llm": LLM_MODEL, "embed": EMBED_MODEL, "dims": 1024,
            "chunk_words": ingest.CHUNK_WORDS, "overlap_words": ingest.OVERLAP_WORDS,
            "max_distance": MAX_DISTANCE,
            "files": [{"name": k, "chunks": v} for k, v in sorted(per.items())],
            "total_chunks": sum(per.values()),
            "ollama_models": models,
            "depths": {k: {"k": v["k"], "verify": bool(v.get("verify"))}
                       for k, v in DEPTHS.items()},
        }

    # ============================ لوحة الإدارة ============================
    def _json(self, obj, code=200):
        self._headers("application/json; charset=utf-8", code)
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _admin_ok(self):
        return hmac.compare_digest(self.headers.get("X-Admin-Pass", ""), ADMIN_PASS)

    def _user_ok(self):
        return USER_SESSIONS.get(self.headers.get("X-User-Token", "")) is not None

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def _ollama_models(self):
        out = []
        try:
            resp = ollama.list()
            items = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])
            for m in items or []:
                name = (m.get("model") if isinstance(m, dict)
                        else getattr(m, "model", None) or getattr(m, "name", None))
                if name:
                    out.append(str(name))
        except Exception:
            pass
        return out

    def _files_detail(self):
        """قائمة الملفات المفهرسة مع عدد المقاطع ونطاق الصفحات."""
        files = {}
        try:
            got = COL.get(include=["metadatas"])
            for m in got["metadatas"]:
                src = m.get("source", "?")
                f = files.setdefault(src, {"name": src, "chunks": 0, "pages": set(), "headings": set()})
                f["chunks"] += 1
                if m.get("pages"):
                    f["pages"].add(str(m["pages"]))
                if m.get("heading"):
                    f["headings"].add(m["heading"])
        except Exception:
            pass
        return [{"name": f["name"], "chunks": f["chunks"],
                 "pages_count": len(f["pages"]), "headings_count": len(f["headings"])}
                for f in sorted(files.values(), key=lambda x: x["name"])]

    def _admin_get(self, path, params):
        if not self._admin_ok():
            self._json({"error": "غير مصرّح — كلمة مرور خاطئة"}, 401)
            return
        if path == "/admin/api/overview":
            per = {}
            try:
                for m in COL.get(include=["metadatas"])["metadatas"]:
                    per[m["source"]] = per.get(m["source"], 0) + 1
            except Exception:
                pass
            fb = 0
            try:
                fb = sum(1 for _ in open(FEEDBACK_FILE, encoding="utf-8"))
            except Exception:
                pass
            self._json({
                "llm": query.LLM_MODEL, "embed": EMBED_MODEL, "dims": 1024,
                "chunk_words": ingest.CHUNK_WORDS, "overlap_words": ingest.OVERLAP_WORDS,
                "max_distance": query.MAX_DISTANCE, "default_depth": DEFAULT_DEPTH,
                "ocr_enabled": ingest.OCR_ENABLED, "ocr_model": ingest.OCR_MODEL,
                "files_count": len(per), "total_chunks": sum(per.values()),
                "feedback_count": fb, "ollama_models": self._ollama_models(),
                "attached_session": list(ATTACHED["names"]),
                "depths": list(DEPTHS.keys()),
            })
        elif path == "/admin/api/files":
            self._json({"files": self._files_detail()})
        elif path == "/admin/api/chunks":
            src = params.get("source", [""])[0]
            rows = []
            try:
                got = COL.get(where={"source": src}, include=["documents", "metadatas"])
                for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"]):
                    rows.append({"id": cid, "chunk": m.get("chunk", 0),
                                 "pages": m.get("pages", ""), "heading": m.get("heading", ""),
                                 "text": doc})
                rows.sort(key=lambda r: r["chunk"])
            except Exception as e:
                self._json({"error": str(e)}, 500)
                return
            self._json({"source": src, "chunks": rows})
        elif path == "/admin/api/feedback":
            entries = []
            try:
                for line in open(FEEDBACK_FILE, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            except FileNotFoundError:
                pass
            except Exception as e:
                self._json({"error": str(e)}, 500)
                return
            self._json({"entries": entries[-200:]})
        elif path == "/admin/api/access":
            acc = load_access()
            self._json({
                "all_data": ALL_DATA,
                "data_groups": acc["data_groups"],
                "user_groups": acc["user_groups"],
                "users": {u: {"group": v.get("group", "")} for u, v in acc["users"].items()},
                "all_files": [f["name"] for f in self._files_detail()],
            })
        else:
            self._json({"error": "مسار غير معروف"}, 404)

    def _index_file_stream(self, emit, name, ext, raw_bytes):
        """يحفظ الملف في documents/ ويفهرسه في COL مع بثّ التقدّم."""
        dest = ingest.DOCS_DIR / name
        ingest.DOCS_DIR.mkdir(exist_ok=True)
        dest.write_bytes(raw_bytes)
        emit({"status": "saved"})
        paragraphs = ingest.READERS[ext](dest)
        chunks = ingest.chunk_paragraphs(paragraphs)
        if not chunks:
            emit({"error": "تعذّر استخراج نص من الملف"})
            return
        COL.delete(where={"source": name})
        for i in range(0, len(chunks), ingest.BATCH_SIZE):
            batch = chunks[i:i + ingest.BATCH_SIZE]
            vectors = ingest.embed([c["text"] for c in batch])
            COL.add(
                ids=[f"{name}::{i + j}" for j in range(len(batch))],
                embeddings=vectors,
                documents=[c["text"] for c in batch],
                metadatas=[{"source": name, "pages": c["pages"], "chunk": i + j,
                            "heading": c.get("heading", "")} for j, c in enumerate(batch)],
            )
            emit({"done": min(i + ingest.BATCH_SIZE, len(chunks)), "total": len(chunks)})
        emit({"ok": True, "name": name, "chunks": len(chunks)})

    def _admin_post(self, path):
        global COL, DEFAULT_DEPTH
        if not self._admin_ok():
            self._json({"error": "غير مصرّح — كلمة مرور خاطئة"}, 401)
            return

        # رفع وفهرسة ملف جديد (بثّ ندجسون)
        if path == "/admin/api/upload":
            self._headers("application/x-ndjson; charset=utf-8")

            def emit(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            try:
                name = Path(unquote(self.headers.get("X-Filename", ""))).name
                ext = Path(name).suffix.lower()
                length = int(self.headers.get("Content-Length", 0))
                if not name or ext not in ingest.READERS:
                    emit({"error": "نوع الملف غير مدعوم — المسموح: pdf, docx, txt, md"})
                    return
                if length <= 0 or length > 200 * 1024 * 1024:
                    emit({"error": "حجم الملف غير صالح (الحد الأقصى 200MB)"})
                    return
                self._index_file_stream(emit, name, ext, self.rfile.read(length))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    emit({"error": str(e)})
                except Exception:
                    pass
            return

        # إعادة فهرسة كامل مجلد documents/ (مسح ثم بناء)
        if path == "/admin/api/reindex":
            self._headers("application/x-ndjson; charset=utf-8")

            def emit(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            try:
                files = sorted(p for p in ingest.DOCS_DIR.rglob("*")
                               if p.suffix.lower() in ingest.READERS)
                if not files:
                    emit({"error": "لا توجد ملفات في مجلد documents/"})
                    return
                emit({"phase": "clear"})
                try:
                    for cid_batch in [got["ids"] for got in [COL.get()]]:
                        if cid_batch:
                            COL.delete(ids=cid_batch)
                except Exception:
                    pass
                for fi, p in enumerate(files, 1):
                    emit({"phase": "file", "file": p.name, "i": fi, "n": len(files)})
                    self._index_file_stream(emit, p.name, p.suffix.lower(), p.read_bytes())
                emit({"ok": True, "files": len(files)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    emit({"error": str(e)})
                except Exception:
                    pass
            return

        # باقي أوامر JSON
        data = self._read_json()

        if path == "/admin/api/delete_file":
            name = str(data.get("name", ""))
            try:
                COL.delete(where={"source": name})
                fp = ingest.DOCS_DIR / Path(name).name
                if fp.exists():
                    fp.unlink()
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/admin/api/delete_chunk":
            cid = str(data.get("id", ""))
            try:
                COL.delete(ids=[cid])
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/admin/api/rename_file":
            old, new = str(data.get("name", "")), str(data.get("new_name", "")).strip()
            if not new:
                self._json({"error": "الاسم الجديد فارغ"}, 400)
                return
            try:
                got = COL.get(where={"source": old}, include=["metadatas"])
                if got["ids"]:
                    metas = []
                    for m in got["metadatas"]:
                        m = dict(m)
                        m["source"] = new
                        metas.append(m)
                    COL.update(ids=got["ids"], metadatas=metas)
                fp = ingest.DOCS_DIR / Path(old).name
                if fp.exists():
                    fp.rename(ingest.DOCS_DIR / Path(new).name)
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/admin/api/clear_db":
            try:
                _client.delete_collection(COLLECTION)
                COL = _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/admin/api/search":
            q = str(data.get("q", "")).strip()
            k = int(data.get("k", 8))
            if not q:
                self._json({"error": "استعلام فارغ"}, 400)
                return
            try:
                hits, n_cand, section = select_hits(COL, q, {"k": k})
                # select_hits قد يُرجع القسم كاملًا عند تطابق قوي (ميزة الإجابة)؛
                # أما في أداة الاختبار فنقيّد العرض بعدد المقاطع المطلوب k
                hits = hits[:k]
                self._json({"section": section, "candidates": n_cand, "hits": [
                    {"source": m["source"], "pages": m.get("pages", ""),
                     "heading": m.get("heading", ""), "score": match_score(d),
                     "text": doc[:500] + ("…" if len(doc) > 500 else "")}
                    for doc, m, d in hits]})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/admin/api/settings":
            try:
                if data.get("llm"):
                    query.LLM_MODEL = str(data["llm"])
                if data.get("max_distance") is not None:
                    query.MAX_DISTANCE = float(data["max_distance"])
                if data.get("chunk_words") is not None:
                    ingest.CHUNK_WORDS = int(data["chunk_words"])
                if data.get("overlap_words") is not None:
                    ingest.OVERLAP_WORDS = int(data["overlap_words"])
                if data.get("default_depth") in DEPTHS:
                    DEFAULT_DEPTH = data["default_depth"]
                if data.get("ocr_model"):
                    ingest.OCR_MODEL = str(data["ocr_model"])
                if data.get("ocr_enabled") is not None:
                    ingest.OCR_ENABLED = bool(data["ocr_enabled"])
                self._json({"ok": True, "llm": query.LLM_MODEL, "max_distance": query.MAX_DISTANCE,
                            "chunk_words": ingest.CHUNK_WORDS, "overlap_words": ingest.OVERLAP_WORDS,
                            "default_depth": DEFAULT_DEPTH,
                            "ocr_enabled": ingest.OCR_ENABLED, "ocr_model": ingest.OCR_MODEL})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        # ===== صلاحيات الوصول =====
        if path == "/admin/api/datagroup_save":
            name = str(data.get("name", "")).strip()
            old = str(data.get("old_name", "")).strip()
            files = [str(f) for f in data.get("files", [])]
            if not name or name == ALL_DATA:
                self._json({"error": "اسم غير صالح (محجوز أو فارغ)"}, 400)
                return
            acc = load_access()
            if old and old != name and old in acc["data_groups"]:
                del acc["data_groups"][old]
                for g in acc["user_groups"].values():  # حدّث الروابط عند إعادة التسمية
                    g["data_groups"] = [name if d == old else d for d in g.get("data_groups", [])]
            acc["data_groups"][name] = files
            save_access(acc)
            self._json({"ok": True})
            return

        if path == "/admin/api/datagroup_delete":
            name = str(data.get("name", ""))
            acc = load_access()
            acc["data_groups"].pop(name, None)
            for g in acc["user_groups"].values():
                g["data_groups"] = [d for d in g.get("data_groups", []) if d != name]
            save_access(acc)
            self._json({"ok": True})
            return

        if path == "/admin/api/usergroup_save":
            name = str(data.get("name", "")).strip()
            old = str(data.get("old_name", "")).strip()
            dgs = [str(d) for d in data.get("data_groups", [])]
            if not name:
                self._json({"error": "اسم فارغ"}, 400)
                return
            acc = load_access()
            if old and old != name and old in acc["user_groups"]:
                del acc["user_groups"][old]
                for u in acc["users"].values():  # حدّث انتماء المستخدمين
                    if u.get("group") == old:
                        u["group"] = name
            acc["user_groups"][name] = {"data_groups": dgs}
            save_access(acc)
            self._json({"ok": True})
            return

        if path == "/admin/api/usergroup_delete":
            name = str(data.get("name", ""))
            acc = load_access()
            acc["user_groups"].pop(name, None)
            for u in acc["users"].values():
                if u.get("group") == name:
                    u["group"] = ""
            save_access(acc)
            self._json({"ok": True})
            return

        if path == "/admin/api/user_save":
            username = str(data.get("username", "")).strip()
            old = str(data.get("old_username", "")).strip()
            group = str(data.get("group", "")).strip()
            password = data.get("password")
            if not username:
                self._json({"error": "اسم المستخدم فارغ"}, 400)
                return
            acc = load_access()
            if old and old != username and old in acc["users"]:
                acc["users"][username] = acc["users"].pop(old)
            entry = acc["users"].get(username, {"password": "", "group": ""})
            entry["group"] = group
            if password:  # لا نمسح كلمة المرور إن تُركت فارغة عند التعديل
                entry["password"] = hash_password(str(password))
            if not entry.get("password"):
                self._json({"error": "كلمة مرور المستخدم مطلوبة"}, 400)
                return
            acc["users"][username] = entry
            save_access(acc)
            self._json({"ok": True})
            return

        if path == "/admin/api/user_delete":
            username = str(data.get("username", ""))
            acc = load_access()
            acc["users"].pop(username, None)
            save_access(acc)
            self._json({"ok": True})
            return

        self._json({"error": "مسار غير معروف"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/admin/login":
            # يصدر توكن جلسة أدمن مؤقت — لتفادي تمرير كلمة المرور في رابط EventSource
            data = self._read_json()
            if not hmac.compare_digest(str(data.get("pass", "")), ADMIN_PASS):
                self._json({"error": "كلمة المرور غير صحيحة"}, 401)
                return
            token = secrets.token_urlsafe(24)
            ADMIN_SESSIONS.add(token)
            self._json({"ok": True, "token": token})
            return
        if url.path.startswith("/admin/api/"):
            try:
                self._admin_post(url.path)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if url.path == "/user/login":
            # ┌── نقطة دمج Azure/Entra ID ───────────────────────────────────────┐
            # │ هنا تُستبدل المصادقة المحلية بتدفق OIDC/OAuth2 ضد Entra ID (MSAL). │
            # │ بعد التحقق من توكن Azure: استخرج هوية الموظف ومطالبة المجموعة       │
            # │ (group/role claim)، واربطها بمجموعة في access.json ثم املأ         │
            # │ USER_SESSIONS كالمعتاد — بقية النظام (الصلاحيات) يبقى دون تغيير.   │
            # └──────────────────────────────────────────────────────────────────┘
            data = self._read_json()
            username = str(data.get("username", "")).strip()
            password = str(data.get("password", ""))
            acc = load_access()
            u = acc["users"].get(username)
            if not u or not verify_password(password, u.get("password", "")):
                audit_log("login", user=username, ok=False)
                self._json({"error": "اسم المستخدم أو كلمة المرور غير صحيحة"}, 401)
                return
            token = secrets.token_urlsafe(24)
            USER_SESSIONS[token] = username
            audit_log("login", user=username, ok=True)
            allowed = user_allowed_sources(username)
            grp = u.get("group", "")
            self._json({"ok": True, "token": token, "username": username, "group": grp,
                        "full_access": allowed is None,
                        "files_count": (-1 if allowed is None else len(allowed))})
            return
        if url.path == "/user/logout":
            data = self._read_json()
            USER_SESSIONS.pop(str(data.get("token", "")), None)
            self._json({"ok": True})
            return
        if url.path == "/feedback":
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "model": LLM_MODEL,
                         "question": str(data.get("q", ""))[:500],
                         "depth": str(data.get("depth", "")),
                         "rating": str(data.get("rating", "")),
                         "elapsed_s": data.get("elapsed")}
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._headers("application/json")
                self.wfile.write(b'{"ok": true}')
            except Exception as e:
                self._headers("application/json", 500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if url.path in ("/attach_clear", "/attach") and not self._user_ok():
            self._json({"error": "يجب تسجيل الدخول"}, 401)
            return
        if url.path == "/attach_clear":
            try:
                COL_TMP.delete(where={"session": "1"})
            except Exception:
                pass
            ATTACHED["inline"].clear()
            ATTACHED["temp"].clear()
            ATTACHED["names"].clear()
            self._headers("application/json")
            self.wfile.write(b'{"ok": true}')
            return
        if url.path == "/attach":
            self._headers("application/x-ndjson; charset=utf-8")

            def emit_a(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                name = Path(unquote(self.headers.get("X-Filename", ""))).name
                ext = Path(name).suffix.lower()
                length = int(self.headers.get("Content-Length", 0))
                if not name or ext not in ingest.READERS:
                    emit_a({"error": "نوع الملف غير مدعوم — المسموح: pdf, docx, txt, md"})
                    return
                if length <= 0 or length > 200 * 1024 * 1024:
                    emit_a({"error": "حجم الملف غير صالح (الحد الأقصى 200MB)"})
                    return
                if name in ATTACHED["names"]:
                    emit_a({"error": "الملف مُرفق بالفعل في هذه الجلسة"})
                    return

                ATTACH_DIR.mkdir(exist_ok=True)
                tmp = ATTACH_DIR / name
                tmp.write_bytes(self.rfile.read(length))
                emit_a({"status": "saved"})

                paragraphs = ingest.READERS[ext](tmp)
                full_text = "\n".join(t.replace(ingest.HEAD_MARK, "") for t, _pg in paragraphs).strip()
                if not full_text:
                    emit_a({"error": "تعذّر استخراج نص من الملف"})
                    tmp.unlink(missing_ok=True)
                    return

                if len(full_text) <= ATTACH_INLINE_CHARS:
                    # ملف صغير → نحقن نصه كاملًا في السياق
                    ATTACHED["inline"].append({"name": name, "text": full_text})
                    ATTACHED["names"].append(name)
                    emit_a({"ok": True, "name": name, "mode": "inline", "chars": len(full_text)})
                else:
                    # ملف كبير → فهرسة مؤقتة في COL_TMP
                    chunks = ingest.chunk_paragraphs(paragraphs)
                    if not chunks:
                        emit_a({"error": "تعذّر تقطيع الملف"})
                        tmp.unlink(missing_ok=True)
                        return
                    for i in range(0, len(chunks), ingest.BATCH_SIZE):
                        batch = chunks[i:i + ingest.BATCH_SIZE]
                        vectors = ingest.embed([c["text"] for c in batch])
                        COL_TMP.add(
                            ids=[f"attach::{name}::{i + j}" for j in range(len(batch))],
                            embeddings=vectors,
                            documents=[c["text"] for c in batch],
                            metadatas=[{"source": name, "pages": c["pages"], "chunk": i + j,
                                        "heading": c.get("heading", ""), "session": "1"}
                                       for j, c in enumerate(batch)],
                        )
                        emit_a({"done": min(i + ingest.BATCH_SIZE, len(chunks)), "total": len(chunks)})
                    ATTACHED["temp"].append(name)
                    ATTACHED["names"].append(name)
                    emit_a({"ok": True, "name": name, "mode": "rag", "chunks": len(chunks)})
                tmp.unlink(missing_ok=True)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    emit_a({"error": str(e)})
                except Exception:
                    pass
            return
        if url.path != "/upload":
            self._headers("text/plain", 404)
            return
        # الرفع الدائم للفهرسة صار خاصًّا بالإدارة فقط (كلمة مرور)
        if not self._admin_ok():
            self._json({"error": "الرفع الدائم متاح من لوحة الإدارة فقط"}, 403)
            return
        self._headers("application/x-ndjson; charset=utf-8")

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            name = Path(unquote(self.headers.get("X-Filename", ""))).name
            ext = Path(name).suffix.lower()
            length = int(self.headers.get("Content-Length", 0))
            if not name or ext not in ingest.READERS:
                emit({"error": "نوع الملف غير مدعوم — المسموح: pdf, docx, txt, md"})
                return
            if length <= 0 or length > 200 * 1024 * 1024:
                emit({"error": "حجم الملف غير صالح (الحد الأقصى 200MB)"})
                return

            dest = ingest.DOCS_DIR / name
            ingest.DOCS_DIR.mkdir(exist_ok=True)
            dest.write_bytes(self.rfile.read(length))
            emit({"status": "saved"})

            paragraphs = ingest.READERS[ext](dest)
            chunks = ingest.chunk_paragraphs(paragraphs)
            if not chunks:
                emit({"error": "تعذّر استخراج نص من الملف"})
                return

            COL.delete(where={"source": name})  # لا تكرار عند إعادة الرفع
            for i in range(0, len(chunks), ingest.BATCH_SIZE):
                batch = chunks[i:i + ingest.BATCH_SIZE]
                vectors = ingest.embed([c["text"] for c in batch])
                COL.add(
                    ids=[f"{name}::{i + j}" for j in range(len(batch))],
                    embeddings=vectors,
                    documents=[c["text"] for c in batch],
                    metadatas=[{"source": name, "pages": c["pages"], "chunk": i + j,
                                "heading": c.get("heading", "")}
                               for j, c in enumerate(batch)],
                )
                emit({"done": min(i + ingest.BATCH_SIZE, len(chunks)), "total": len(chunks)})
            emit({"ok": True, "name": name, "chunks": len(chunks)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                emit({"error": str(e)})
            except Exception:
                pass

    def _emit(self, event, data):
        self.wfile.write(sse(event, data))
        self.wfile.flush()

    def _stream_tokens(self, stream):
        last = {}

        def spy(s):
            nonlocal last
            for part in s:
                last = part
                yield part

        for token in iter_tokens(spy(stream)):
            self._emit("token", {"t": token})

        # إحصاءات التوليد الفعلية من Ollama (توكنز، مدد بالنانوثانية)
        stats = {k: _g(last, k) for k in
                 ("eval_count", "prompt_eval_count", "eval_duration",
                  "prompt_eval_duration", "total_duration")}
        if stats.get("eval_count"):
            self._emit("gen_stats", stats)

    def _answer_attached(self, q, cfg):
        """إجابة من المستندات المُرفقة للجلسة: الصغيرة تُحقن كاملة، الكبيرة تُسترجع من COL_TMP."""
        self._emit("stage", {"stage": "route"})
        has_temp = bool(ATTACHED["temp"])
        self._emit("mode", {"mode": "attached", "retrieval": has_temp})

        parts, sources_meta = [], []
        for f in ATTACHED["inline"]:
            parts.append(f"### ملف مرفق (كامل): {f['name']}\n{f['text']}")
            sources_meta.append({"source": f["name"], "pages": "الملف كامل", "score": 100})

        if has_temp:
            self._emit("stage", {"stage": "embed"})
            self._emit("stage", {"stage": "retrieve"})
            hits, _n, _sec = select_hits(COL_TMP, q, cfg)
            for doc, m, d in hits:
                parts.append(f"### مقطع من: {m['source']} (ص. {m.get('pages','')})\n{doc}")
                sources_meta.append({"source": m["source"], "pages": m.get("pages", ""),
                                     "score": match_score(d)})

        if sources_meta:
            self._emit("sources", sources_meta)

        self._emit("stage", {"stage": "generate"})
        context = "\n\n".join(parts)
        msgs = [{"role": "system", "content": ATTACH_PROMPT},
                {"role": "user", "content": f"المستندات المرفقة:\n\n{context}\n\n---\nالسؤال: {q}"}]
        self._stream_tokens(_chat(msgs))
        self._emit("done", {})

    def _answer(self, q, depth="balanced", source="docs", allowed=None):
        cfg = DEPTHS[depth]

        # 0-أ) مستندات مُرفقة للجلسة → لها الأولوية (النية صريحة: اسأل عن هذا الملف)
        if ATTACHED["inline"] or ATTACHED["temp"]:
            self._answer_attached(q, cfg)
            return

        # 0-ب) مصدر «المعرفة العامة»: تجاوز البحث في المستندات والإجابة من معرفة النموذج
        # مستوى التفكير يتحكم بطول الإجابة ودقتها ورسميتها عبر أسلوب مخصّص لكل مستوى
        if source == "general":
            self._emit("stage", {"stage": "route"})
            self._emit("mode", {"mode": "general"})
            self._emit("stage", {"stage": "generate"})
            gstyle = GENERAL_STYLES.get(depth, GENERAL_STYLES["balanced"])
            sys_prompt = f"{GENERAL_PROMPT}\n\nأسلوب الإجابة المطلوب:\n{gstyle}"
            self._stream_tokens(_chat([{"role": "system", "content": sys_prompt},
                                       {"role": "user", "content": q}]))
            self._emit("done", {})
            return

        # 1) التوجيه: دردشة أم بحث في المستندات؟
        self._emit("stage", {"stage": "route"})
        mode = route(q)
        self._emit("mode", {"mode": "chat" if mode == "greet" else mode})

        if mode == "greet":  # تحية → رد فوري بدون نموذج
            self._emit("stage", {"stage": "generate"})
            self._emit("token", {"t": greeting_reply(q)})
            self._emit("done", {})
            return

        if mode == "chat":
            self._emit("stage", {"stage": "generate"})
            self._stream_tokens(_chat([{"role": "system", "content": CHAT_PROMPT},
                                       {"role": "user", "content": q}]))
            self._emit("done", {})
            return

        # 2) تضمين السؤال + استرجاع مع اختيار أقل عدد ملفات يكفي (مقيّد بصلاحيات الوصول)
        self._emit("stage", {"stage": "embed"})
        self._emit("stage", {"stage": "retrieve"})
        hits, n_candidates, section = select_hits(COL, q, cfg, allowed=allowed)

        # لا مقاطع ذات صلة → رسالة ثابتة صريحة (بلا معرفة عامة، بلا هلوسة)
        if not hits:
            self._emit("stage", {"stage": "generate"})
            self._emit("token", {"t": NOT_FOUND_MSG})
            self._emit("done", {})
            return

        used_files = sorted({m["source"] for _, m, _d in hits})
        self._emit("selection", {"used": len(used_files), "candidates": n_candidates,
                                 "chunks": len(hits), "section": section})
        self._emit("sources", [
            {"source": m["source"], "pages": m.get("pages", ""), "score": match_score(d)}
            for _, m, d in hits
        ])
        # نصوص المقاطع كاملةً للوحة التحليل
        self._emit("chunks", [
            {"source": m["source"], "pages": m.get("pages", ""),
             "heading": m.get("heading", ""), "score": match_score(d),
             "text": doc[:700] + ("…" if len(doc) > 700 else "")}
            for doc, m, d in hits
        ])

        # 3) التوليد المقيد بالسياق وبأسلوب مستوى التفكير (مع مرحلتَي الدقة القصوى)
        self._emit("stage", {"stage": "generate"})
        msgs = build_answer_messages(
            q, hits, section, cfg,
            progress=lambda i, n: self._emit("phase", {"i": i, "n": n}))
        self._stream_tokens(_chat(msgs))
        self._emit("done", {})


def main():
    if ADMIN_PASS == "admin":
        print("⚠️  تحذير أمني: كلمة مرور لوحة الإدارة هي القيمة الافتراضية \"admin\".")
        print("    اضبط كلمة مرور قوية قبل أي تشغيل غير محلي:  export RAG_ADMIN_PASSWORD='...'")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"✅ الواجهة شغّالة: {url}  (Ctrl+C للإيقاف)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nتم الإيقاف.")


if __name__ == "__main__":
    main()
