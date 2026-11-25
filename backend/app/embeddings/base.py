"""Embedder protocol.

An *embedder* maps text to a fixed-length vector so that semantically (or lexically)
similar texts end up close together. Two implementations ship with this project:
a from-scratch TF-IDF baseline and a real dense neural model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can turn texts into row vectors."""

    name: str
    needs_fit: bool

    def fit(self, texts: list[str]) -> None:
        """Learn corpus statistics (only meaningful when ``needs_fit`` is true)."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages to an ``(n, dim)`` float32 matrix."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query to a ``(dim,)`` float32 vector."""
