"""TF-IDF embeddings implemented from scratch with NumPy.

TF-IDF is the classic *sparse* baseline: each dimension is a vocabulary term and its
value is how often the term occurs in the text (term frequency, TF) scaled by how rare
the term is across the corpus (inverse document frequency, IDF)::

    idf(t)   = ln((n + 1) / (df(t) + 1)) + 1        # smoothed, always > 0
    tfidf    = tf(t, text) * idf(t)                 # then L2-normalised

It only matches *words*, not meaning ("car" will not match "automobile"), which is
precisely the gap dense embeddings close. It needs no model download, so it is also
the offline fallback for the whole system.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.text import tokenize


class TfidfEmbedder:
    """Corpus-fitted TF-IDF vectoriser (uni-grams, L2-normalised output)."""

    name = "tfidf"
    needs_fit = True

    def __init__(self, max_features: int = 20000) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)

    def fit(self, texts: list[str]) -> None:
        """Build the vocabulary and IDF weights from ``texts``."""
        n = len(texts)
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        terms = sorted(df, key=lambda t: (-df[t], t))[: self.max_features]
        terms.sort()
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.idf = np.array(
            [math.log((n + 1) / (df[t] + 1)) + 1.0 for t in terms], dtype=np.float32
        )

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for term, count in Counter(tokenize(text)).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = count * self.idf[idx]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages; returns an ``(n, vocab)`` matrix."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vectorize(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the fitted vocabulary (unknown terms are ignored)."""
        return self._vectorize(text)
