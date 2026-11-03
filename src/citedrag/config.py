import os
from dataclasses import dataclass

# Vercel functions have a read-only filesystem except /tmp, so runtime files go there.
ON_VERCEL = bool(os.getenv("VERCEL"))
_RUNTIME_DIR = "/tmp/citedrag" if ON_VERCEL else "."


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("CITEDRAG_MODEL", "claude-opus-5-5")
    judge_model: str = os.getenv("CITEDRAG_JUDGE_MODEL", "claude-opus-5-5")
    # Claude Opus 5.5 defaults to "medium"; set explicitly so behaviour is stable.
    effort: str = os.getenv("CITEDRAG_EFFORT", "medium")
    top_k: int = int(os.getenv("CITEDRAG_TOP_K", "5"))
    chunk_words: int = int(os.getenv("CITEDRAG_CHUNK_WORDS", "180"))
    chunk_overlap: int = int(os.getenv("CITEDRAG_CHUNK_OVERLAP", "30"))
    index_path: str = os.getenv("CITEDRAG_INDEX", f"{_RUNTIME_DIR}/index.json")
    # Built-in sample documents, indexed on startup if no index file exists.
    data_dir: str = os.getenv("CITEDRAG_DATA_DIR", "data/sample")
    golden_path: str = os.getenv("CITEDRAG_GOLDEN", "eval/golden.jsonl")
    # Questions allowed per visitor IP per hour (protects your API bill on a public demo). 0 = unlimited.
    ask_limit_per_hour: int = int(os.getenv("CITEDRAG_ASK_LIMIT", "20" if ON_VERCEL else "0"))


settings = Settings()

# Server-side refusal fallback: if a request is declined by a safety classifier,
# the API re-runs it on Anthropic's recommended fallback model in the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
