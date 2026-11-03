"""Answer a question from retrieved chunks using Claude's native citations.

Each retrieved chunk is passed as its own `document` content block with
citations enabled. Claude's answer comes back split into text blocks, and the
cited ones carry the exact quoted span plus the index of the document it came
from, so every claim can be traced back to a page/section.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import anthropic

from .config import FALLBACK_BETA, settings
from .retrieve import Hit
from .text import tokenize

SYSTEM_PROMPT = """You answer questions for analysts using only the documents provided.

- Base every factual statement on the documents and cite it.
- If the documents do not contain the answer, say so plainly instead of guessing.
- If the question names a specific market, region or product, answer only for that one.
- Be concise: lead with the direct answer, then supporting detail."""


@dataclass
class Citation:
    number: int
    chunk_id: str
    label: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    mode: str = "generative"


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9£€$])")


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> Answer:
    """Answer without an LLM: quote the sentences that best match the question.

    Used when no API key is configured, so the app still works (at zero cost).
    Every sentence is quoted verbatim and cited, so nothing can be invented.
    """
    if not hits:
        return Answer("I couldn't find anything relevant to that question in the documents.", mode="extractive")
    q = set(tokenize(question))
    candidates = []
    for rank, hit in enumerate(hits):
        for sentence in _SENTENCE.split(hit.chunk.text.strip()):
            overlap = len(q & set(tokenize(sentence)))
            if overlap:
                # Prefer sentences sharing more query terms, from higher-ranked passages.
                candidates.append((overlap + hit.score / (hits[0].score or 1), rank, sentence.strip(), hit))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    picked = candidates[:max_sentences]
    if not picked:
        return Answer("The documents don't seem to contain an answer to that question.", mode="extractive")

    citations, parts = [], []
    for i, (_, _, sentence, hit) in enumerate(picked, start=1):
        citations.append(Citation(i, hit.chunk.chunk_id, hit.chunk.label, sentence))
        parts.append(f"{sentence}[{i}]")
    return Answer("\n\n".join(parts), citations, mode="extractive")


def build_documents(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.chunk.text},
            "title": h.chunk.label,
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def render_answer(content, hits: list[Hit]) -> tuple[str, list[Citation]]:
    """Stitch text blocks together and turn citations into numbered footnotes."""
    parts: list[str] = []
    citations: list[Citation] = []
    seen: dict[tuple[int, str], int] = {}
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        markers = []
        for cit in getattr(block, "citations", None) or []:
            doc_idx = cit.document_index
            key = (doc_idx, cit.cited_text)
            if key not in seen:
                chunk = hits[doc_idx].chunk
                seen[key] = len(citations) + 1
                citations.append(Citation(seen[key], chunk.chunk_id, chunk.label, cit.cited_text.strip()))
            markers.append(f"[{seen[key]}]")
        if markers:
            parts.append("".join(sorted(set(markers), key=markers.index)))
    return "".join(parts).strip(), citations


def answer(question: str, hits: list[Hit], client: anthropic.Anthropic | None = None) -> Answer:
    if not hits:
        return Answer("I couldn't find anything relevant to that question in the documents.")

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=settings.model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": settings.effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
        messages=[{
            "role": "user",
            "content": [*build_documents(hits), {"type": "text", "text": question}],
        }],
    )

    if response.stop_reason == "refusal":
        return Answer("This question couldn't be answered.", refused=True, model=response.model)

    text, citations = render_answer(response.content, hits)
    return Answer(text, citations, model=response.model)


def respond(question: str, hits: list[Hit], client: anthropic.Anthropic | None = None) -> Answer:
    """Use Claude when an API key is available, otherwise fall back to extractive mode."""
    if client is None and not has_api_key():
        return extractive_answer(question, hits)
    return answer(question, hits, client=client)
