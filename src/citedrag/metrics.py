"""Retrieval and citation metrics. Pure functions, no API calls, easy to test."""

from __future__ import annotations

from statistics import mean


def hit_at_k(retrieved_docs: list[str], expected: set[str], k: int) -> float:
    return 1.0 if any(d in expected for d in retrieved_docs[:k]) else 0.0


def reciprocal_rank(retrieved_docs: list[str], expected: set[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_docs: list[str], expected: set[str]) -> float | None:
    """Share of citations that point at an expected source. None if nothing cited."""
    if not cited_docs:
        return None
    return sum(d in expected for d in cited_docs) / len(cited_docs)


def safe_mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None
