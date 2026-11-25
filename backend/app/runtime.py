"""Application container: builds the pipeline from settings."""

from __future__ import annotations

from app.config import Settings
from app.embeddings.dense import create_embedder
from app.generation.llm import create_llm
from app.pipeline import RAGPipeline


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Create embedder, LLM client and pipeline according to ``settings``."""
    embedder, reason = create_embedder(settings)
    return RAGPipeline(settings, embedder, create_llm(settings), embedder_fallback_reason=reason)
