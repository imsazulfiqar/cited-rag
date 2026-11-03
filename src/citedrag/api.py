"""FastAPI service and web UI.

GET  /              web interface
POST /ask           cited answer for a question
GET  /documents     indexed documents
POST /upload        add .md/.txt/.pdf files and re-index
GET  /examples      sample questions from the golden set
GET  /eval          retrieval metrics on the golden set (no model calls)
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import ON_VERCEL, settings
from .evaluate import load_golden, run_eval
from .generate import has_api_key, respond
from .ingest import ingest_folder, load_index, save_index
from .retrieve import Retriever

STATIC = Path(__file__).parent / "static"
ALLOWED = {".md", ".txt", ".pdf"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="citedrag", version="0.1.0")
_lock = threading.Lock()
_retriever: Retriever | None = None
_asks: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(request: Request) -> None:
    """Simple per-IP sliding window. Per server instance, so a soft guard, not a hard cap."""
    limit = settings.ask_limit_per_hour
    if not limit:
        return
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now, window = time.monotonic(), _asks[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(429, f"Demo limit reached ({limit} questions per hour). Please try again later.")
    window.append(now)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=settings.top_k, ge=1, le=20)


def _folders() -> list[Path]:
    return [p for p in (Path(settings.data_dir), Path(settings.upload_dir)) if p.is_dir() and any(p.iterdir())]


def rebuild_index() -> Retriever:
    global _retriever
    chunks = [c for folder in _folders() for c in ingest_folder(folder, settings.chunk_words, settings.chunk_overlap)]
    save_index(chunks, settings.index_path)
    _retriever = Retriever(chunks)
    return _retriever


def get_retriever() -> Retriever:
    global _retriever
    with _lock:
        if _retriever is None:
            if Path(settings.index_path).exists():
                _retriever = Retriever(load_index(settings.index_path))
            else:
                rebuild_index()
        return _retriever


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(get_retriever().chunks)}


@app.get("/config")
def config() -> dict:
    return {"ephemeral_uploads": ON_VERCEL, "model": settings.model,
            "mode": "generative" if has_api_key() else "extractive"}


@app.get("/documents")
def documents() -> list[dict]:
    docs: dict[str, dict] = {}
    for c in get_retriever().chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "tags": c.tags, "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())


@app.get("/examples")
def examples() -> list[str]:
    path = Path(settings.golden_path)
    return [row["question"] for row in load_golden(path)][:6] if path.exists() else []


@app.get("/eval")
def evaluate(k: int = 3) -> dict:
    path = Path(settings.golden_path)
    if not path.exists():
        raise HTTPException(404, "No golden question set found")
    return run_eval(get_retriever(), load_golden(path), k=k)


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    target = Path(settings.upload_dir)
    target.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(f.filename or "").name)
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(400, f"{f.filename}: only .md, .txt and .pdf files are supported")
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename}: file is larger than 20 MB")
        (target / name).write_bytes(data)
        saved.append(name)
    with _lock:
        retriever = rebuild_index()
    return {"saved": saved, "chunks": len(retriever.chunks)}


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> dict:
    _check_rate_limit(request)
    hits = get_retriever().search(req.question, k=req.k)
    try:
        result = respond(req.question, hits)
    except anthropic.AuthenticationError:
        raise HTTPException(500, "The server's ANTHROPIC_API_KEY is missing or invalid")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the model provider, retry shortly")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Model provider error ({e.status_code})")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Could not reach the model provider")
    return {
        "answer": result.text,
        "citations": [c.__dict__ for c in result.citations],
        "refused": result.refused,
        "model": result.model,
        "mode": result.mode,
        "retrieved": [{"chunk_id": h.chunk.chunk_id, "label": h.chunk.label, "score": round(h.score, 3),
                       "text": h.chunk.text, "tags": h.chunk.tags} for h in hits],
    }
