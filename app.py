"""Vercel entrypoint. Vercel looks for a FastAPI instance named `app` in app.py.

Locally you can also run: uvicorn app:app --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from citedrag.api import app  # noqa: E402,F401
