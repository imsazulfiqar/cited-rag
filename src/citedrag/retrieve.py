"""BM25 retrieval with query normalisation and metadata tag boosting.

Plain BM25 is a strong baseline on dense, terminology-heavy reports. Two
additions target the failure modes seen most in practice:

* Tag boosting: if the query names a tag value (e.g. a market such as
  "Germany"), chunks carrying that tag are boosted.
* Cross-tag suppression: chunks tagged with a *different* value for the same
  key are down-weighted, so a question about Germany doesn't get answered
  from the UK report.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .ingest import Chunk
from .text import normalise, tokenize

BOOST = 1.5
PENALTY = 0.35
FILTER_KEYS = ("market", "region", "country")


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.tag_values: dict[str, set[str]] = {}
        for c in chunks:
            for key, value in c.tags.items():
                self.tag_values.setdefault(key, set()).add(value.lower())

    def detect_tags(self, query: str) -> dict[str, str]:
        """Find tag values (e.g. market names) mentioned in the query."""
        q = f" {normalise(query)} "
        found = {}
        for key, values in self.tag_values.items():
            for value in values:
                if f" {normalise(value)} " in q:
                    found[key] = value
        return found

    def search(self, query: str, k: int = 5, allowed_docs: set[str] | None = None) -> list[Hit]:
        """Top-k chunks for the query. If allowed_docs is given, only those documents are searched."""
        scores = self.bm25.get_scores(tokenize(query))
        wanted = self.detect_tags(query)
        hits = []
        for chunk, score in zip(self.chunks, scores):
            if score <= 0 or (allowed_docs is not None and chunk.doc_id not in allowed_docs):
                continue
            for key, value in wanted.items():
                have = chunk.tags.get(key, "").lower()
                if have == value:
                    score *= BOOST
                elif have and key in FILTER_KEYS:
                    score *= PENALTY
            hits.append(Hit(chunk, float(score)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
