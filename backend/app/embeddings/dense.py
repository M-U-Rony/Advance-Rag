"""Dense neural embeddings via ``fastembed`` (ONNX runtime, no PyTorch needed).

A dense *bi-encoder* (default ``BAAI/bge-small-en-v1.5``, 384 dimensions) encodes
queries and passages independently into the same semantic space, so "automobile" can
match "car". The model is downloaded lazily on first use; if it is unavailable
(offline machine) :func:`create_embedder` transparently falls back to TF-IDF.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import Settings
from app.embeddings.base import Embedder
from app.embeddings.tfidf import TfidfEmbedder

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseEmbedder:
    """Wrapper around a fastembed ``TextEmbedding`` model."""

    needs_fit = False

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy import

        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self._model: Any = TextEmbedding(model_name=model_name)
        self._prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def fit(self, texts: list[str]) -> None:
        """No-op: dense models are pre-trained."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages to an ``(n, dim)`` float32 matrix."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack(list(self._model.embed(texts))).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query (with the BGE retrieval instruction prefix)."""
        return np.asarray(next(iter(self._model.embed([self._prefix + text]))), dtype=np.float32)


def create_embedder(settings: Settings) -> tuple[Embedder, str | None]:
    """Build the configured embedder.

    Returns ``(embedder, fallback_reason)``; ``fallback_reason`` is ``None`` unless a dense
    model was requested but could not be loaded.
    """
    if settings.embedding_backend == "tfidf":
        return TfidfEmbedder(), None
    try:
        return DenseEmbedder(settings.embedding_model), None
    except Exception as exc:  # network failure, missing model, onnx error...
        reason = f"{type(exc).__name__}: {str(exc)[:160]}"
        logger.warning("Dense embedder unavailable (%s); falling back to TF-IDF", reason)
        return TfidfEmbedder(), reason
