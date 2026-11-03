"""API tests with the model call stubbed out (no API key needed)."""

import io

import pytest
from fastapi.testclient import TestClient

from citedrag import api
from citedrag.generate import Answer, Citation

SPAIN = b"---\ntitle: Spain brief\nmarket: Spain\n---\n# Pricing\nThe average price in Spain was 1,900 euros in 2025."


@pytest.fixture()
def client(tmp_path, monkeypatch):
    object.__setattr__(api.settings, "index_path", str(tmp_path / "missing-index.json"))
    api._base = None

    def fake_respond(question, hits, client=None):
        if not hits:
            return Answer("nothing found")
        c = hits[0].chunk
        return Answer("Top answer[1].", [Citation(1, c.chunk_id, c.label, c.text[:40])], model="stub")
    monkeypatch.setattr(api, "respond", fake_respond)
    return TestClient(api.app)


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "citedrag" in r.text


def test_ask_returns_citations_and_passages(client):
    r = client.post("/ask", json={"question": "biggest barrier in the UK?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["number"] == 1
    assert body["retrieved"][0]["doc_id"] == "ebike_uk_2026"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "price?", "doc_ids": []}).status_code == 400


def test_doc_filter_limits_search(client):
    r = client.post("/ask", json={"question": "average selling price", "doc_ids": ["ebike_france_2026"]})
    assert {p["doc_id"] for p in r.json()["retrieved"]} == {"ebike_france_2026"}


def test_documents_examples_eval(client):
    assert len(client.get("/documents").json()) == 3
    assert len(client.get("/examples").json()) == 4
    result = client.get("/eval?k=3").json()
    assert result["summary"]["hit@3"] == 1.0
    assert result["rows"][0]["found_at"] == 1 and result["rows"][0]["expected"] == ["ebike_uk_2026"]


def test_parse_then_ask_own_document(client):
    r = client.post("/parse", files=[("files", ("spain.md", io.BytesIO(SPAIN), "text/markdown"))])
    assert r.status_code == 200
    doc = r.json()["documents"][0]
    assert doc["doc_id"] == "upload-spain" and doc["chunk_list"][0]["doc_id"] == "upload-spain"
    # nothing is stored server-side: samples are unchanged
    assert len(client.get("/documents").json()) == 3

    ask = client.post("/ask", json={"question": "average price in Spain?", "doc_ids": ["upload-spain"],
                                     "user_chunks": doc["chunk_list"]})
    assert ask.status_code == 200
    assert ask.json()["retrieved"][0]["doc_id"] == "upload-spain"


def test_parse_rejects_bad_files(client):
    bad = client.post("/parse", files=[("files", ("x.exe", io.BytesIO(b"x"), "application/octet-stream"))])
    assert bad.status_code == 400
    empty = client.post("/parse", files=[("files", ("empty.txt", io.BytesIO(b"   "), "text/plain"))])
    assert empty.status_code == 422


def test_rate_limit(client):
    object.__setattr__(api.settings, "ask_limit_per_hour", 2)
    api._asks.clear()
    try:
        codes = [client.post("/ask", json={"question": "UK price?"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
    finally:
        object.__setattr__(api.settings, "ask_limit_per_hour", 0)


def test_custom_eval_on_own_documents(client):
    doc = client.post("/parse", files=[("files", ("spain.md", io.BytesIO(SPAIN), "text/markdown"))]).json()["documents"][0]
    r = client.post("/eval/custom", json={
        "k": 3,
        "cases": [
            {"question": "What was the average price in Spain?", "expected_doc": "upload-spain"},
            {"question": "UK motor power rules", "expected_doc": "ebike_uk_2026"},
        ],
        "user_chunks": doc["chunk_list"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2 and body["summary"]["hit@1"] == 1.0
    assert body["rows"][0]["found_at"] == 1 and body["rows"][0]["expected"] == ["upload-spain"]


def test_custom_eval_rejects_unknown_document(client):
    r = client.post("/eval/custom", json={"cases": [{"question": "price?", "expected_doc": "upload-gone"}]})
    assert r.status_code == 400 and "Re-upload" in r.json()["detail"]
