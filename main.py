"""Compatibility entrypoint.

Use `uvicorn app.main:app --reload` for the modular architecture.
This file is kept so existing commands like `uvicorn main:app --reload` still work.
"""

from app.main import app

__all__ = ["app"]
