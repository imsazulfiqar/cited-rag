from pathlib import Path

import pytest

from citedrag.ingest import ingest_folder
from citedrag.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def chunks():
    return ingest_folder(ROOT / "data" / "sample", size=180, overlap=30)


@pytest.fixture(scope="session")
def retriever(chunks):
    return Retriever(chunks)
