"""Test doubles."""

from __future__ import annotations

import zlib
from collections.abc import Iterator

import numpy as np

from app.text import tokenize


class FakeLLM:
    """Deterministic stand-in for a real model."""

    name = "fake"

    def __init__(self, reply: str = "fake reply", fail: bool = False) -> None:
        self.reply, self.fail, self.calls = reply, fail, 0

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.reply

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        if self.fail:
            raise RuntimeError("boom")
        yield from ("fake ", "answer [1]")


class HashEmbedder:
    """Deterministic non-fitted "dense" embedder (hashed bag of words) that counts its work."""

    name = "hash-dense"
    needs_fit = False

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.embedded_texts = 0

    def fit(self, texts: list[str]) -> None:
        """No-op."""

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text, remove_stopwords=True):
            vec[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.embedded_texts += len(texts)
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
