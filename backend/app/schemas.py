"""Pydantic schemas for the API and the pipeline trace."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["vector", "bm25", "hybrid"]
FusionMethod = Literal["rrf", "weighted"]
RerankMethod = Literal["none", "heuristic", "cross_encoder"]
QueryTransform = Literal["none", "rewrite", "hyde", "multi_query"]
SimilarityMetric = Literal["cosine", "dot", "l2"]


class RetrievalSettings(BaseModel):
    """Every knob of the retrieval pipeline (all optional, with sensible defaults)."""

    retrieval_mode: RetrievalMode = "hybrid"
    fusion: FusionMethod = "rrf"
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="0 = keyword only, 1 = vector only")
    top_k: int = Field(5, ge=1, le=20)
    candidates: int = Field(30, ge=1, le=100, description="Candidates fetched before reranking")
    rerank: RerankMethod = "heuristic"
    query_transform: QueryTransform = "none"
    parent_child: bool = False
    compression: bool = False
    similarity: SimilarityMetric = "cosine"
    doc_ids: list[str] | None = None
    page_min: int | None = Field(None, ge=1)
    page_max: int | None = Field(None, ge=1)


class HistoryTurn(BaseModel):
    """One chat message of the conversation history."""

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """Body of ``POST /api/query``."""

    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list)
    settings: RetrievalSettings = Field(default_factory=RetrievalSettings)


class Hit(BaseModel):
    """A retrieved passage at some stage of the pipeline."""

    chunk_id: str
    filename: str
    page: int
    rank: int
    score: float
    snippet: str
    is_parent: bool = False


class Stage(BaseModel):
    """Trace of one pipeline stage."""

    key: str
    label: str
    ms: float = 0.0
    skipped: bool = True
    info: dict[str, Any] = Field(default_factory=dict)
    hits: list[Hit] = Field(default_factory=list)


class Trace(BaseModel):
    """Full, inspectable record of how an answer was produced."""

    question: str
    condensed_query: str
    rewritten_query: str | None = None
    hyde_text: str | None = None
    multi_queries: list[str] = Field(default_factory=list)
    stages: list[Stage]
    ranked_chunk_ids: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    total_ms: float = 0.0
    embedder: str
    llm: str
    reranker: str | None = None
    warnings: list[str] = Field(default_factory=list)
    settings: RetrievalSettings


class Citation(BaseModel):
    """A source passage the answer may cite as ``[index]``."""

    index: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    score: float
    text: str


class QueryResponse(BaseModel):
    """Answer with citations and the pipeline trace."""

    answer: str
    citations: list[Citation]
    trace: Trace


class DocumentInfo(BaseModel):
    """Public view of an indexed document."""

    id: str
    filename: str
    pages: int
    chunks: int
    parent_chunks: int
    strategy: str
    chunk_size: int
    chunk_overlap: int
    created_at: str


class EvalRow(BaseModel):
    """Retrieval and generation metrics for one pipeline configuration."""

    config: str
    recall: dict[str, float]
    precision: float
    mrr: float
    faithfulness: float
    answer_relevance: float
    avg_ms: float


class EvalReport(BaseModel):
    """Result of ``POST /api/evaluate``."""

    embedder: str
    questions: int
    ks: list[int]
    precision_k: int
    source: str
    rows: list[EvalRow]


class EvalRequest(BaseModel):
    """Body of ``POST /api/evaluate``."""

    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    max_questions: int = Field(30, ge=1, le=200)
    doc_ids: list[str] | None = None
