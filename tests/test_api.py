"""API tests with the model call stubbed out (no API key needed)."""

import io

import pytest
from fastapi.testclient import TestClient

from citedrag import api
from citedrag.generate import Answer, Citation


@pytest.fixture()
def client(tmp_path, monkeypatch):
    object.__setattr__(api.settings, "index_path", str(tmp_path / "index.json"))
    object.__setattr__(api.settings, "upload_dir", str(tmp_path / "uploads"))
    api._retriever = None

    def fake_answer(question, hits, client=None):
        c = hits[0].chunk
        return Answer("Fear of theft is the top barrier[1].", [Citation(1, c.chunk_id, c.label, "fear of theft (47%)")],
                      model="stub")
    monkeypatch.setattr(api, "respond", fake_answer)
    return TestClient(api.app)


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "citedrag" in r.text


def test_ask_returns_citations_and_passages(client):
    r = client.post("/ask", json={"question": "biggest barrier in the UK?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["number"] == 1
    assert body["retrieved"][0]["chunk_id"].startswith("ebike_uk_2026")
    assert "text" in body["retrieved"][0]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_documents_examples_eval(client):
    assert len(client.get("/documents").json()) == 3
    assert len(client.get("/examples").json()) == 6
    assert client.get("/eval?k=3").json()["summary"]["hit@3"] == 1.0


def test_upload_adds_document_and_rejects_bad_types(client):
    md = b"---\ntitle: Spain brief\nmarket: Spain\n---\n# Pricing\nThe average price in Spain was 1,900 euros."
    r = client.post("/upload", files=[("files", ("spain.md", io.BytesIO(md), "text/markdown"))])
    assert r.status_code == 200 and r.json()["saved"] == ["spain.md"]
    assert any(d["doc_id"] == "spain" for d in client.get("/documents").json())
    bad = client.post("/upload", files=[("files", ("x.exe", io.BytesIO(b"x"), "application/octet-stream"))])
    assert bad.status_code == 400


def test_rate_limit(client):
    object.__setattr__(api.settings, "ask_limit_per_hour", 2)
    api._asks.clear()
    try:
        codes = [client.post("/ask", json={"question": "UK price?"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
    finally:
        object.__setattr__(api.settings, "ask_limit_per_hour", 0)
