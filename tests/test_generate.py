"""Tests for turning Claude's cited response into numbered footnotes (no API call)."""

from types import SimpleNamespace as NS

from citedrag.generate import answer, build_documents, render_answer


def test_documents_have_citations_enabled(retriever):
    docs = build_documents(retriever.search("UK price", k=2))
    assert len(docs) == 2
    assert all(d["citations"] == {"enabled": True} and d["type"] == "document" for d in docs)


def test_render_answer_numbers_citations(retriever):
    hits = retriever.search("UK average selling price", k=2)
    content = [
        NS(type="text", text="The UK average price was ", citations=None),
        NS(type="text", text="£1,850 in 2025",
           citations=[NS(document_index=0, cited_text="The average selling price in the UK was £1,850 in 2025")]),
        NS(type="text", text=".", citations=None),
    ]
    text, cits = render_answer(content, hits)
    assert text == "The UK average price was £1,850 in 2025[1]."
    assert cits[0].number == 1 and cits[0].chunk_id == hits[0].chunk.chunk_id


def test_no_hits_skips_the_api():
    result = answer("anything", [], client=object())
    assert "couldn't find" in result.text and not result.citations


def test_refusal_is_handled(retriever):
    class FakeClient:
        class beta:
            class messages:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["fallbacks"] == "default"
                    return NS(stop_reason="refusal", content=[], model="claude-opus-5-5")
    result = answer("q", retriever.search("UK price", k=1), client=FakeClient())
    assert result.refused


def test_extractive_answer_quotes_and_cites(retriever, monkeypatch):
    from citedrag.generate import respond
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    q = "What is the biggest barrier to buying an e-bike in the UK?"
    result = respond(q, retriever.search(q, k=3))
    assert result.mode == "extractive"
    assert "theft (47%)" in result.text
    # every cited quote appears verbatim in the source chunk
    by_id = {c.chunk_id: c for c in retriever.chunks}
    assert all(c.quote in by_id[c.chunk_id].text for c in result.citations)
