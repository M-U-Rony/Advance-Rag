"""Reranking: a precise second pass over the retrieval candidates.

First-stage retrieval uses a **bi-encoder** (or BM25): query and passage are encoded
*separately*, so passages can be pre-computed and searched in milliseconds, but the model
never sees query and passage together. A **cross-encoder** reads ``[query, passage]`` jointly
and outputs a relevance score, which is far more accurate but too slow to run on the whole
corpus. The standard recipe is therefore: retrieve ~30 candidates cheaply, rerank them with
the cross-encoder, keep the best 5.

Two rerankers are provided: :class:`CrossEncoderReranker` (fastembed ONNX model, lazy-loaded)
and :class:`HeuristicReranker`, a dependency-free fallback using term overlap, bigram
matches and an early-position boost.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Protocol

from app.text import content_terms, tokenize

logger = logging.getLogger(__name__)

RerankItem = tuple[str, str]  # (chunk id, text)


class Reranker(Protocol):
    """Scores ``(id, text)`` candidates against a query."""

    name: str

    def rerank(self, query: str, items: list[RerankItem]) -> list[tuple[str, float]]:
        """Return all ids with relevance scores, best first."""


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(tokens))


class HeuristicReranker:
    """Lexical reranker: stop-word filtered term overlap + bigram matches + early position."""

    name = "heuristic"

    def score(self, query: str, text: str) -> float:
        """Relevance in roughly [0, 1.2] of ``text`` for ``query``."""
        q_terms = content_terms(query)
        if not q_terms:
            return 0.0
        doc_tokens = tokenize(text)
        doc_terms = set(doc_tokens)
        unique_q = set(q_terms)
        overlap = len(unique_q & doc_terms) / len(unique_q)
        q_bigrams = _bigrams(tokenize(query))
        bigram_hits = len(q_bigrams & _bigrams(doc_tokens)) / len(q_bigrams) if q_bigrams else 0.0
        early = 0.0
        head = set(doc_tokens[: max(len(doc_tokens) // 5, 10)])
        if unique_q & head:
            early = len(unique_q & head) / len(unique_q)
        return 0.6 * overlap + 0.3 * bigram_hits + 0.1 * early

    def rerank(self, query: str, items: list[RerankItem]) -> list[tuple[str, float]]:
        """Score and sort candidates (ties keep the incoming order)."""
        scored = [(cid, self.score(query, text)) for cid, text in items]
        return sorted(scored, key=lambda x: -x[1])


class CrossEncoderReranker:
    """fastembed cross-encoder (``ms-marco-MiniLM-L-6-v2``) with a heuristic fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.name = "cross_encoder"
        self._model: Any = None
        self._failed: str | None = None
        self._fallback = HeuristicReranker()

    @property
    def available(self) -> bool | None:
        """``None`` until first use, then whether the model loaded."""
        if self._model is not None:
            return True
        return False if self._failed else None

    @property
    def failure(self) -> str | None:
        """Why the model could not be loaded, if it could not."""
        return self._failed

    def _load(self) -> Any | None:
        if self._model is None and self._failed is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(model_name=self.model_name)
            except Exception as exc:
                self._failed = f"{type(exc).__name__}: {str(exc)[:160]}"
                logger.warning("Cross-encoder unavailable (%s); using heuristic reranker", exc)
        return self._model

    def rerank(self, query: str, items: list[RerankItem]) -> list[tuple[str, float]]:
        """Cross-encode ``(query, passage)`` pairs; falls back to the heuristic if unavailable."""
        model = self._load()
        if model is None or not items:
            self.name = "heuristic (cross-encoder unavailable)"
            return self._fallback.rerank(query, items)
        self.name = "cross_encoder"
        scores = list(model.rerank(query, [text for _, text in items]))
        scored = [(cid, float(s)) for (cid, _), s in zip(items, scores, strict=True)]
        return sorted(scored, key=lambda x: -x[1])
