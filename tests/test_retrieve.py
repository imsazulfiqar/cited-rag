import json
from pathlib import Path

import pytest

from citedrag.evaluate import run_eval

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"


def test_detects_market_in_query(retriever):
    assert retriever.detect_tags("Average price in Germany?") == {"market": "germany"}


@pytest.mark.parametrize("question,doc", [
    ("What was the average selling price in Germany?", "ebike_germany_2026"),
    ("What is the biggest purchase barrier in France?", "ebike_france_2026"),
    ("UK motor power rules", "ebike_uk_2026"),
])
def test_market_questions_hit_the_right_report(retriever, question, doc):
    assert retriever.search(question, k=1)[0].chunk.doc_id == doc


def test_cross_market_results_are_suppressed(retriever):
    hits = retriever.search("purchase barriers in the UK", k=len(retriever.chunks))
    assert [h.chunk.doc_id for h in hits[:3]] == ["ebike_uk_2026"] * 3
    best_uk = hits[0].score
    best_other = max(h.score for h in hits if h.chunk.doc_id != "ebike_uk_2026")
    assert best_uk > 2 * best_other


def test_golden_set_retrieval_quality(retriever):
    golden = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    result = run_eval(retriever, golden, k=3)
    assert result["summary"]["hit@3"] == 1.0
    assert result["summary"]["mrr"] >= 0.9
