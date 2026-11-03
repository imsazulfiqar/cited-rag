"""Turn a folder of documents into page/section-aware chunks with metadata.

Each chunk keeps where it came from (doc id, title, section, page) so answers
can cite it, plus document-level tags (e.g. market, topic) that retrieval uses
to boost on-topic results and suppress cross-market ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    page: int
    text: str
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        where = f"p.{self.page}" if self.page else self.section
        return f"{self.title} ({where})"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip('"')
    return meta, raw[match.end():]


def split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def _markdown_sections(body: str) -> list[tuple[str, str]]:
    sections, heading, buf = [], "Introduction", []
    for line in body.splitlines():
        if line.startswith("#"):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf)))
    return sections


def load_document(path: Path, size: int, overlap: int) -> list[Chunk]:
    doc_id = path.stem
    chunks: list[Chunk] = []

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("PDF support needs: pip install pypdf") from exc
        reader = PdfReader(str(path))
        title = (reader.metadata.title if reader.metadata else None) or doc_id
        for page_no, page in enumerate(reader.pages, start=1):
            for j, piece in enumerate(split_words(page.extract_text() or "", size, overlap)):
                chunks.append(Chunk(f"{doc_id}:p{page_no}:{j}", doc_id, title, f"Page {page_no}", page_no, piece))
        return chunks

    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    title = meta.pop("title", doc_id.replace("_", " ").title())
    for s_no, (heading, text) in enumerate(_markdown_sections(body), start=1):
        for j, piece in enumerate(split_words(text, size, overlap)):
            chunks.append(Chunk(f"{doc_id}:s{s_no}:{j}", doc_id, title, heading, 0, piece, dict(meta)))
    return chunks


def ingest_folder(folder: str | Path, size: int = 180, overlap: int = 30) -> list[Chunk]:
    folder = Path(folder)
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"})
    if not paths:
        raise SystemExit(f"No .md/.txt/.pdf files found in {folder}")
    return [c for p in paths for c in load_document(p, size, overlap)]


def save_index(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")


def load_index(path: str | Path) -> list[Chunk]:
    return [Chunk(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]
