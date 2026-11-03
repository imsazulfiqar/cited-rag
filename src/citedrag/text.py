"""Lightweight text normalisation used for both indexing and querying.

Keeping the exact same pipeline on both sides is what makes lexical retrieval
reliable: "Pricing", "prices" and "priced" should all meet in the middle.
"""

import re
import unicodedata

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its may might more most of on or our over shall
    should so such than that the their them then there these they this those to
    under up was we were what when where which while who why will with within
    would you your about across after against all also any between both each
    per vs""".split()
)

_SUFFIXES = (
    ("ational", "ate"), ("ization", "ize"), ("fulness", "ful"),
    ("iveness", "ive"), ("ements", "ement"), ("ments", "ment"),
    ("ities", "ity"), ("ings", ""), ("ing", ""), ("ies", "y"),
    ("ied", "y"), ("sses", "ss"), ("es", "e"), ("ed", ""), ("s", ""),
)


def normalise(text: str) -> str:
    """Lower-case, strip accents and collapse punctuation to spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9%£$€.\-]+", " ", text).strip()


def lemmatise(token: str) -> str:
    """Rule-based suffix stripping (no model download needed)."""
    if len(token) <= 4 or token[0].isdigit():
        return token
    for suffix, repl in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: len(token) - len(suffix)] + repl
            # "prices" -> "price", "pricing" -> "pric" -> fold the trailing e away
            return stem.rstrip("e") if len(stem) > 4 else stem
    return token.rstrip("e") if len(token) > 4 else token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in normalise(text).split():
        raw = raw.strip(".-")
        if not raw or raw in STOPWORDS:
            continue
        tokens.append(lemmatise(raw))
    return tokens
