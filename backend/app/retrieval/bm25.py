"""BM25 keyword ranking implemented from scratch (Okapi BM25, Lucene-style IDF).

BM25 scores a document ``d`` for query terms ``q_i`` as::

    score(d, q) = sum_i  idf(q_i) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln( (N - df + 0.5) / (df + 0.5) + 1 )

* ``k1`` (1.2) controls term-frequency saturation: the 10th occurrence of a word adds far
  less than the 1st.
* ``b`` (0.75) controls length normalisation: long documents are penalised so they cannot
  win merely by containing more words.

Unlike embeddings BM25 needs exact word matches, which makes it excellent for rare
identifiers, names and numbers, and weak on paraphrases. Hybrid search combines both.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.text import tokenize


class BM25Index:
    """In-memory BM25 index over a list of texts."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: np.ndarray = np.zeros(0)
        self.avgdl = 0.0
        self.n_docs = 0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}

    def index(self, texts: list[str]) -> None:
        """(Re)build the index from ``texts``; document ids are their list positions."""
        self.n_docs = len(texts)
        self.postings = {}
        lengths: list[int] = []
        for doc_id, text in enumerate(texts):
            tokens = tokenize(text, remove_stopwords=True)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, tf))
        self.doc_len = np.array(lengths, dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if lengths else 0.0
        self.idf = {
            term: math.log((self.n_docs - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            for term, p in self.postings.items()
        }

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every indexed document for ``query``."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0 or self.avgdl == 0:
            return scores
        for term in set(tokenize(query, remove_stopwords=True)):
            for doc_id, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                scores[doc_id] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(
        self, query: str, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc_index, score)`` pairs with positive score.

        ``allowed`` is an optional boolean mask implementing metadata pre-filtering.
        """
        scores = self.score(query)
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
