from citedrag.ingest import parse_front_matter, split_words


def test_front_matter_parsed(chunks):
    uk = [c for c in chunks if c.doc_id == "ebike_uk_2026"]
    assert uk and all(c.tags["market"] == "UK" for c in uk)
    assert uk[0].title.startswith("E-Bike Market Brief - United Kingdom")


def test_sections_become_chunks(chunks):
    sections = {c.section for c in chunks if c.doc_id == "ebike_germany_2026"}
    assert {"Market size", "Pricing", "Consumer barriers", "Distribution"} <= sections


def test_split_words_overlaps():
    pieces = split_words(" ".join(str(i) for i in range(100)), size=40, overlap=10)
    assert pieces[0].split()[-10:] == pieces[1].split()[:10]


def test_no_front_matter():
    meta, body = parse_front_matter("# Title\ntext")
    assert meta == {} and body.startswith("# Title")
