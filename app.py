# -*- coding: utf-8 -*-
"""
app.py — واجهة ويب محلية للمحادثة مع مستنداتك (بدون أي اعتماديات إضافية).
الاستخدام: python3 app.py  ثم افتح http://localhost:8765
"""
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import time

import chromadb
import ollama

import ingest
from query import (CHAT_PROMPT, FALLBACK_PROMPT, COLLECTION, DB_DIR, DEPTHS,
                   LLM_MODEL, EMBED_MODEL, MAX_DISTANCE,
                   route, select_hits, iter_tokens, _chat,
                   greeting_reply, match_score, build_answer_messages)

FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"


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
except Exception as e:
    sys.exit(f"تعذّر فتح قاعدة البيانات: {e}")


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
            depth = params.get("depth", ["balanced"])[0]
            if depth not in DEPTHS:
                depth = "balanced"
            self._headers("text/event-stream; charset=utf-8")
            if not q:
                self.wfile.write(sse("err", {"msg": "سؤال فارغ"}))
                return
            try:
                self._answer(q, depth)
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

    def do_POST(self):
        url = urlparse(self.path)
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
        if url.path != "/upload":
            self._headers("text/plain", 404)
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

    def _answer(self, q, depth="balanced"):
        cfg = DEPTHS[depth]

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

        # 2) تضمين السؤال + استرجاع مع اختيار أقل عدد ملفات يكفي
        self._emit("stage", {"stage": "embed"})
        self._emit("stage", {"stage": "retrieve"})
        hits, n_candidates, section = select_hits(COL, q, cfg)

        # لا مقاطع ذات صلة → نصرّح ثم نقترح من المعرفة العامة
        if not hits:
            self._emit("stage", {"stage": "generate"})
            self._stream_tokens(_chat([{"role": "system", "content": FALLBACK_PROMPT},
                                       {"role": "user", "content": q}]))
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
