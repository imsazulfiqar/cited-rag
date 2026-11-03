from citedrag.text import lemmatise, normalise, tokenize


def test_price_variants_share_a_stem():
    assert lemmatise("prices") == lemmatise("pricing") == lemmatise("price") == lemmatise("priced")


def test_stopwords_removed_and_case_folded():
    assert tokenize("What is THE price in the UK?") == ["pric", "uk"]


def test_normalise_strips_accents_and_keeps_currency():
    assert normalise("Café costs €2,950!") == "cafe costs €2 950"
