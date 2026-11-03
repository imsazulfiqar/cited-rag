from citedrag.metrics import citation_precision, hit_at_k, reciprocal_rank, safe_mean


def test_hit_and_mrr():
    docs = ["a", "b", "c"]
    assert hit_at_k(docs, {"b"}, 1) == 0.0
    assert hit_at_k(docs, {"b"}, 2) == 1.0
    assert reciprocal_rank(docs, {"c"}) == 1 / 3
    assert reciprocal_rank(docs, {"z"}) == 0.0


def test_citation_precision():
    assert citation_precision(["a", "a", "b"], {"a"}) == 2 / 3
    assert citation_precision([], {"a"}) is None


def test_safe_mean_ignores_missing():
    assert safe_mean([1, None, 0]) == 0.5
    assert safe_mean([None]) is None
