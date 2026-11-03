"""FastAPI service and web UI.

GET  /              web interface
GET  /documents     built-in sample documents
POST /parse         turn uploaded files into chunks (nothing is stored on the server)
POST /ask           cited answer, searching selected sample docs + the user's own uploaded chunks
GET  /examples      sample questions
GET  /eval          retrieval accuracy on the golden question set (no model calls)
POST /eval/custom   retrieval accuracy on the user's own test questions and documents

Uploaded documents are never saved on the server. /parse returns their chunks
to the browser, which keeps them for the session and sends them with each
question. That keeps every visitor's documents private to them and works on
serverless hosts like Vercel, where server memory and disk don't persist.
"""

from __future__ import annotations

import re
import tempfile
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
from .ingest import Chunk, ingest_folder, load_document, load_index
from .retrieve import Retriever

STATIC = Path(__file__).parent / "static"
ALLOWED = {".md", ".txt", ".pdf"}
# Vercel rejects request bodies over 4.5 MB, so keep uploads (and the chunks sent back) under that.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_USER_CHUNKS = 3000
MAX_CHUNK_CHARS = 4000

app = FastAPI(title="citedrag", version="0.2.0")
_lock = threading.Lock()
_base: Retriever | None = None
_asks: dict[str, deque] = defaultdict(deque)


class ChunkIn(BaseModel):
    chunk_id: str = Field(max_length=300)
    doc_id: str = Field(max_length=200)
    title: str = Field(max_length=300)
    section: str = Field(default="", max_length=300)
    page: int = 0
    text: str = Field(max_length=MAX_CHUNK_CHARS)
    tags: dict[str, str] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=settings.top_k, ge=1, le=20)
    # Which documents to search. None = all built-in samples plus every uploaded doc.
    doc_ids: list[str] | None = Field(default=None, max_length=500)
    user_chunks: list[ChunkIn] = Field(default_factory=list, max_length=MAX_USER_CHUNKS)


class TestCase(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    expected_doc: str = Field(min_length=1, max_length=200)


class CustomEvalRequest(BaseModel):
    cases: list[TestCase] = Field(min_length=1, max_length=50)
    k: int = Field(default=3, ge=1, le=20)
    user_chunks: list[ChunkIn] = Field(default_factory=list, max_length=MAX_USER_CHUNKS)


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


def base_retriever() -> Retriever:
    """Retriever over the built-in sample documents (built once per server instance)."""
    global _base
    with _lock:
        if _base is None:
            path = Path(settings.index_path)
            chunks = load_index(path) if path.exists() else ingest_folder(
                settings.data_dir, settings.chunk_words, settings.chunk_overlap)
            _base = Retriever(chunks)
        return _base


def _doc_summaries(chunks: list[Chunk]) -> list[dict]:
    docs: dict[str, dict] = {}
    for c in chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "tags": c.tags, "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(base_retriever().chunks)}


@app.get("/config")
def config() -> dict:
    return {"model": settings.model, "mode": "generative" if has_api_key() else "extractive",
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


@app.get("/documents")
def documents() -> list[dict]:
    return _doc_summaries(base_retriever().chunks)


@app.get("/examples")
def examples() -> list[str]:
    path = Path(settings.golden_path)
    return [row["question"] for row in load_golden(path)][:4] if path.exists() else []


@app.get("/eval")
def evaluate(k: int = 3) -> dict:
    path = Path(settings.golden_path)
    if not path.exists():
        raise HTTPException(404, "No golden question set found")
    return run_eval(base_retriever(), load_golden(path), k=k)


@app.post("/eval/custom")
def evaluate_custom(req: CustomEvalRequest) -> dict:
    """Score the search on the user's own questions. Searches every document (samples + uploads),
    so the other documents act as distractors, exactly like on the Ask page with everything ticked."""
    base = base_retriever()
    chunks = base.chunks + [Chunk(**c.model_dump()) for c in req.user_chunks]
    known = {c.doc_id for c in chunks}
    unknown = sorted({c.expected_doc for c in req.cases} - known)
    if unknown:
        raise HTTPException(400, f"Unknown document: {unknown[0]}. Re-upload it on the Ask page and try again.")
    retriever = Retriever(chunks) if req.user_chunks else base
    golden = [{"id": f"t{i + 1}", "question": c.question, "expected_docs": [c.expected_doc]}
              for i, c in enumerate(req.cases)]
    return run_eval(retriever, golden, k=req.k)


@app.post("/parse")
async def parse(files: list[UploadFile] = File(...)) -> dict:
    """Chunk uploaded files and return them. Nothing is kept on the server."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(f.filename or "file").name)
            if Path(name).suffix.lower() not in ALLOWED:
                raise HTTPException(400, f"{f.filename}: only PDF, Markdown (.md) and text (.txt) files are supported")
            data = await f.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename}: files must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                chunks = load_document(path, settings.chunk_words, settings.chunk_overlap)
            except Exception:
                raise HTTPException(422, f"{f.filename}: couldn't read this file")
            if not chunks:
                raise HTTPException(422, f"{f.filename}: no readable text found (scanned PDFs aren't supported yet)")
            doc_id = f"upload-{path.stem}"
            for c in chunks:
                c.chunk_id = c.chunk_id.replace(path.stem, doc_id, 1)
                c.doc_id = doc_id
                c.text = c.text[:MAX_CHUNK_CHARS]
            out.append({**_doc_summaries(chunks)[0], "chunk_list": [c.__dict__ for c in chunks]})
    return {"documents": out}


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> dict:
    _check_rate_limit(request)
    base = base_retriever()
    if req.user_chunks:
        retriever = Retriever(base.chunks + [Chunk(**c.model_dump()) for c in req.user_chunks])
    else:
        retriever = base
    allowed = set(req.doc_ids) if req.doc_ids is not None else None
    if allowed is not None and not allowed:
        raise HTTPException(400, "Select at least one document to search")
    hits = retriever.search(req.question, k=req.k, allowed_docs=allowed)
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
        "retrieved": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "label": h.chunk.label,
                       "score": round(h.score, 3), "text": h.chunk.text} for h in hits],
    }
