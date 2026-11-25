"""RAGPipeline: orchestrates indexing (offline) and querying (online).

**Offline indexing**:  PDF -> pages -> chunks (+ parents) -> embeddings -> vector store + BM25.

**Online querying**: Retrieve -> Augment -> Generate, expanded with the advanced stages::

    condense history -> query transform -> vector search \\
                                        -> BM25 search    -> fusion -> rerank
        -> parent expansion -> contextual compression -> grounded prompt -> LLM

Every stage records its latency, parameters and hits in a :class:`~app.schemas.Trace` so the
UI (and the evaluation harness) can show exactly what happened.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.embeddings.base import Embedder
from app.generation.llm import ExtractiveGenerator, LLMClient
from app.generation.prompts import (
    NO_ANSWER,
    ContextItem,
    Turn,
    build_grounded_prompt,
    prompt_token_estimate,
)
from app.ingestion.chunking import ChunkingParams, chunk_document
from app.ingestion.pdf_loader import load_pdf
from app.models import Chunk, ChunkSet
from app.retrieval.bm25 import BM25Index
from app.retrieval.compression import compress_text
from app.retrieval.filters import MetadataFilter
from app.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from app.retrieval.query_transform import (
    condense_with_history,
    hyde_document,
    multi_query,
    rewrite_query,
)
from app.retrieval.rerank import CrossEncoderReranker, HeuristicReranker, Reranker
from app.retrieval.vector_store import VectorStore
from app.schemas import (
    Citation,
    Hit,
    QueryRequest,
    QueryResponse,
    RetrievalSettings,
    Stage,
    Trace,
)
from app.store import DocumentRecord, DocumentStore
from app.text import snippet, word_count

logger = logging.getLogger(__name__)

STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("transform", "Query transform"),
    ("vector", "Vector search"),
    ("bm25", "BM25 keyword search"),
    ("fusion", "Fusion"),
    ("rerank", "Rerank"),
    ("parent", "Parent expansion"),
    ("compression", "Contextual compression"),
    ("generate", "Generate"),
)


@dataclass(slots=True)
class CorpusIndex:
    """Immutable-by-convention snapshot of everything searchable."""

    chunks: list[Chunk] = field(default_factory=list)
    by_id: dict[str, Chunk] = field(default_factory=dict)
    parents: dict[str, Chunk] = field(default_factory=dict)
    vectors: VectorStore = field(default_factory=VectorStore)
    bm25: BM25Index = field(default_factory=BM25Index)


@dataclass(slots=True)
class SearchResult:
    """Ranked ``(chunk id, score)`` lists from first-stage retrieval of one query."""

    vector: list[tuple[str, float]]
    keyword: list[tuple[str, float]]
    fused: list[tuple[str, float]]
    vector_ms: float
    bm25_ms: float


@dataclass(slots=True)
class Retrieval:
    """Output of the retrieval half of the pipeline."""

    trace: Trace
    contexts: list[Chunk]
    context_scores: list[float]
    texts: list[str]
    items: list[ContextItem]
    system: str
    user: str


class _Clock:
    """Tiny stopwatch helper."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def lap(self) -> float:
        """Milliseconds since the previous lap."""
        now = time.perf_counter()
        elapsed = (now - self.start) * 1000
        self.start = now
        return round(elapsed, 2)


def _hit(
    chunk: Chunk, score: float, rank: int, *, is_parent: bool = False, limit: int = 220
) -> Hit:
    return Hit(
        chunk_id=chunk.id,
        filename=chunk.filename,
        page=chunk.page,
        rank=rank,
        score=round(float(score), 5),
        snippet=snippet(chunk.text, limit),
        is_parent=is_parent,
    )


class RAGPipeline:
    """Document indexing and advanced question answering."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        llm: LLMClient | None,
        *,
        embedder_fallback_reason: str | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.embedder_fallback_reason = embedder_fallback_reason
        self.llm = llm
        self.extractive = ExtractiveGenerator()
        self.heuristic_reranker = HeuristicReranker()
        self.cross_encoder = CrossEncoderReranker(settings.reranker_model)
        self.store = DocumentStore(settings.data_dir)
        self.documents: dict[str, DocumentRecord] = {}
        self.index = CorpusIndex()
        self._lock = threading.RLock()
        self._cache_dir = settings.data_dir / "index" / embedder.name.replace("/", "_")
        self._load_persisted()

    # ------------------------------------------------------------------ indexing
    def _load_persisted(self) -> None:
        chunk_sets: dict[str, ChunkSet] = {}
        for record, chunk_set in self.store.load_all():
            self.documents[record.id] = record
            chunk_sets[record.id] = chunk_set
        vectors = VectorStore()
        if not self.embedder.needs_fit and (self._cache_dir / "vectors.json").exists():
            try:
                vectors = VectorStore.load(self._cache_dir)
            except Exception:
                logger.warning("Ignoring unreadable vector cache", exc_info=True)
        self._rebuild(chunk_sets, cached=vectors)

    def _rebuild(self, chunk_sets: dict[str, ChunkSet], cached: VectorStore | None = None) -> None:
        """Recompute the searchable index from stored chunks, re-using cached vectors."""
        chunks = [c for cs in chunk_sets.values() for c in cs.chunks]
        parents = {pid: p for cs in chunk_sets.values() for pid, p in cs.parents.items()}
        texts = [c.text for c in chunks]
        store = VectorStore()
        if chunks:
            if self.embedder.needs_fit:
                self.embedder.fit(texts)
                matrix = self.embedder.embed_documents(texts)
            else:
                known = cached if cached is not None else self.index.vectors
                row = {cid: i for i, cid in enumerate(known.ids)}
                missing = [c for c in chunks if c.id not in row]
                fresh = (
                    self.embedder.embed_documents([c.text for c in missing]) if missing else None
                )
                fresh_row = {c.id: i for i, c in enumerate(missing)}
                matrix = np.vstack(
                    [
                        known.matrix[row[c.id]] if c.id in row else fresh[fresh_row[c.id]]  # type: ignore[index]
                        for c in chunks
                    ]
                )
            store.add([c.id for c in chunks], matrix, [c.metadata() for c in chunks])
            if not self.embedder.needs_fit:
                store.save(self._cache_dir)
        bm25 = BM25Index()
        bm25.index(texts)
        self.index = CorpusIndex(
            chunks=chunks,
            by_id={c.id: c for c in chunks},
            parents=parents,
            vectors=store,
            bm25=bm25,
        )

    def _chunk_sets(self) -> dict[str, ChunkSet]:
        sets: dict[str, ChunkSet] = {doc_id: ChunkSet([], {}) for doc_id in self.documents}
        for chunk in self.index.chunks:
            sets[chunk.doc_id].chunks.append(chunk)
        for pid, parent in self.index.parents.items():
            sets[parent.doc_id].parents[pid] = parent
        return sets

    def add_document(self, filename: str, data: bytes, params: ChunkingParams) -> DocumentRecord:
        """Parse, chunk, embed and index a PDF. Raises ``ValueError`` for unusable input."""
        pages = load_pdf(data)
        doc_id = uuid.uuid4().hex[:10]
        chunk_set = chunk_document(pages, doc_id=doc_id, filename=filename, params=params)
        if not chunk_set.chunks:
            raise ValueError("No text chunks could be produced from this PDF.")
        record = DocumentRecord(
            id=doc_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunk_set.chunks),
            parent_chunks=len(chunk_set.parents),
            strategy=params.strategy,
            chunk_size=params.child_size
            if params.strategy == "parent_child"
            else params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            created_at=DocumentRecord.now(),
        )
        with self._lock:
            sets = self._chunk_sets()
            sets[doc_id] = chunk_set
            self.documents[doc_id] = record
            self._rebuild(sets)
            self.store.save(record, chunk_set)
        return record

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the registry and the indexes."""
        with self._lock:
            if doc_id not in self.documents:
                return False
            sets = self._chunk_sets()
            sets.pop(doc_id, None)
            del self.documents[doc_id]
            self.store.delete(doc_id)
            self._rebuild(sets)
        return True

    def find_by_filename(self, filename: str) -> DocumentRecord | None:
        """Look up an indexed document by its file name."""
        return next((d for d in self.documents.values() if d.filename == filename), None)

    def add_pdf_file(self, path: Path, params: ChunkingParams) -> DocumentRecord:
        """Index a PDF from disk."""
        return self.add_document(path.name, path.read_bytes(), params)

    # ------------------------------------------------------------------ retrieval
    def _metadata_filter(self, s: RetrievalSettings) -> MetadataFilter:
        return MetadataFilter(
            doc_ids=frozenset(s.doc_ids) if s.doc_ids else None,
            page_min=s.page_min,
            page_max=s.page_max,
        )

    def _search_one(
        self,
        index: CorpusIndex,
        vector_query: str,
        keyword_query: str,
        s: RetrievalSettings,
        flt: MetadataFilter,
    ) -> SearchResult:
        """Run first-stage retrieval for one query (vector + BM25 + fusion), timing each part."""
        vec_hits: list[tuple[str, float]] = []
        kw_hits: list[tuple[str, float]] = []
        clock = _Clock()
        if s.retrieval_mode in ("vector", "hybrid"):
            qvec = self.embedder.embed_query(vector_query)
            vec_hits = index.vectors.search(qvec, s.candidates, s.similarity, flt)
        vector_ms = clock.lap()
        if s.retrieval_mode in ("bm25", "hybrid"):
            mask = (
                np.array([flt.matches(c.metadata()) for c in index.chunks]) if flt.active else None
            )
            kw_hits = [
                (index.chunks[i].id, score)
                for i, score in index.bm25.search(keyword_query, s.candidates, mask)
            ]
        bm25_ms = clock.lap()
        if s.retrieval_mode == "vector":
            fused = vec_hits
        elif s.retrieval_mode == "bm25":
            fused = kw_hits
        elif s.fusion == "rrf":
            fused = reciprocal_rank_fusion([[i for i, _ in vec_hits], [i for i, _ in kw_hits]])
        else:
            fused = weighted_fusion(dict(vec_hits), dict(kw_hits), s.alpha)
        return SearchResult(vec_hits, kw_hits, fused[: s.candidates], vector_ms, bm25_ms)

    def _reranker(self, method: str) -> Reranker | None:
        if method == "heuristic":
            return self.heuristic_reranker
        if method == "cross_encoder":
            return self.cross_encoder
        return None

    def retrieve(self, request: QueryRequest) -> Retrieval:
        """Run every stage up to (and including) prompt construction."""
        s = request.settings
        index = self.index
        clock = _Clock()
        total = _Clock()
        history = [Turn(t.role, t.content) for t in request.history]
        stages = {k: Stage(key=k, label=label) for k, label in STAGE_LABELS}
        trace = Trace(
            question=request.question,
            condensed_query=request.question,
            stages=list(stages.values()),
            embedder=self.embedder.name,
            llm=self.llm.name if self.llm else self.extractive.name,
            settings=s,
        )
        flt = self._metadata_filter(s)

        # 1. Query transformation ------------------------------------------------
        query = condense_with_history(request.question, history, self.llm)
        trace.condensed_query = query
        vector_query = keyword_query = query
        extra_queries: list[str] = []
        info: dict[str, Any] = {
            "transform": s.query_transform,
            "condensed": query != request.question,
        }
        if s.query_transform == "rewrite":
            trace.rewritten_query = vector_query = keyword_query = rewrite_query(query, self.llm)
        elif s.query_transform == "hyde":
            trace.hyde_text = vector_query = hyde_document(query, self.llm)
        elif s.query_transform == "multi_query":
            trace.multi_queries = extra_queries = multi_query(query, self.llm)
        info["uses_llm"] = self.llm is not None and (
            s.query_transform != "none" or info["condensed"]
        )
        stage = stages["transform"]
        stage.skipped = s.query_transform == "none" and not info["condensed"]
        stage.info, stage.ms = info, clock.lap()

        # 2-4. First-stage retrieval + fusion ------------------------------------
        clock.lap()
        primary = self._search_one(index, vector_query, keyword_query, s, flt)
        vec_hits, kw_hits, fused = primary.vector, primary.keyword, primary.fused
        per_query = [fused]
        for variant in extra_queries:
            per_query.append(self._search_one(index, variant, variant, s, flt).fused)
        if extra_queries:
            fused = reciprocal_rank_fusion([[i for i, _ in lst] for lst in per_query])[
                : s.candidates
            ]
        if s.retrieval_mode in ("vector", "hybrid"):
            st = stages["vector"]
            st.skipped, st.ms = False, primary.vector_ms
            st.info = {"metric": s.similarity, "embedder": self.embedder.name, "k": s.candidates}
            st.hits = [_hit(index.by_id[i], sc, r) for r, (i, sc) in enumerate(vec_hits, 1)]
        if s.retrieval_mode in ("bm25", "hybrid"):
            st = stages["bm25"]
            st.skipped, st.ms = False, primary.bm25_ms
            st.info = {"k1": index.bm25.k1, "b": index.bm25.b, "k": s.candidates}
            st.hits = [_hit(index.by_id[i], sc, r) for r, (i, sc) in enumerate(kw_hits, 1)]
        clock.lap()
        fusion_stage = stages["fusion"]
        fusion_stage.skipped = s.retrieval_mode != "hybrid" and not extra_queries
        method = "rrf" if extra_queries and s.retrieval_mode != "hybrid" else s.fusion
        fusion_stage.info = {
            "method": method,
            "alpha": s.alpha if method == "weighted" else None,
            "rrf_k": 60 if method == "rrf" else None,
            "queries": 1 + len(extra_queries),
        }
        fusion_stage.hits = [_hit(index.by_id[i], sc, r) for r, (i, sc) in enumerate(fused, 1)]
        fusion_stage.ms = clock.lap()

        # 5. Rerank ----------------------------------------------------------------
        ranked = fused
        reranker = self._reranker(s.rerank)
        if reranker is not None and fused:
            items = [(i, index.by_id[i].text) for i, _ in fused]
            ranked = reranker.rerank(query, items)
            trace.reranker = reranker.name
            stage = stages["rerank"]
            stage.skipped = False
            stage.info = {"reranker": reranker.name, "candidates": len(items), "kept": s.top_k}
            stage.hits = [
                _hit(index.by_id[i], sc, r) for r, (i, sc) in enumerate(ranked[: s.top_k], 1)
            ]
            if isinstance(reranker, CrossEncoderReranker) and reranker.failure:
                trace.warnings.append(
                    f"Cross-encoder unavailable, used heuristic: {reranker.failure}"
                )
        stages["rerank"].ms = clock.lap()
        selected = ranked[: s.top_k]
        trace.ranked_chunk_ids = [i for i, _ in selected]

        # 6. Parent expansion ----------------------------------------------------
        contexts: list[Chunk] = [index.by_id[i] for i, _ in selected]
        scores = [sc for _, sc in selected]
        if s.parent_child and any(c.parent_id for c in contexts):
            expanded: dict[str, tuple[Chunk, float]] = {}
            for chunk, score in zip(contexts, scores, strict=True):
                target = index.parents.get(chunk.parent_id or "", chunk)
                if target.id not in expanded:
                    expanded[target.id] = (target, score)
            stage = stages["parent"]
            stage.skipped = False
            stage.info = {"children": len(contexts), "parents": len(expanded)}
            contexts = [c for c, _ in expanded.values()]
            scores = [sc for _, sc in expanded.values()]
            stage.hits = [
                _hit(c, sc, r, is_parent=True, limit=320)
                for r, (c, sc) in enumerate(zip(contexts, scores, strict=True), 1)
            ]
        stages["parent"].ms = clock.lap()

        # 7. Contextual compression ------------------------------------------------
        texts = [c.text for c in contexts]
        if s.compression and contexts:
            before = sum(word_count(t) for t in texts)
            texts = [compress_text(query, t) for t in texts]
            stage = stages["compression"]
            stage.skipped = False
            stage.info = {"words_before": before, "words_after": sum(word_count(t) for t in texts)}
            stage.hits = [
                Hit(
                    chunk_id=c.id,
                    filename=c.filename,
                    page=c.page,
                    rank=r,
                    score=round(sc, 5),
                    snippet=snippet(t, 400),
                    is_parent=c.id in index.parents,
                )
                for r, (c, sc, t) in enumerate(zip(contexts, scores, texts, strict=True), 1)
            ]
        stages["compression"].ms = clock.lap()

        # 8. Prompt construction (generation happens in answer()/stream) -----------
        items = [
            ContextItem(index=n, filename=c.filename, page=c.page, text=t)
            for n, (c, t) in enumerate(zip(contexts, texts, strict=True), 1)
        ]
        system, user = build_grounded_prompt(request.question, items, history)
        trace.prompt_tokens = prompt_token_estimate(system, user)
        trace.total_ms = total.lap()
        return Retrieval(trace, contexts, scores, texts, items, system, user)

    # ------------------------------------------------------------------ generation
    def _citations(self, r: Retrieval) -> list[Citation]:
        return [
            Citation(
                index=item.index,
                chunk_id=c.id,
                doc_id=c.doc_id,
                filename=c.filename,
                page=c.page,
                score=round(sc, 5),
                text=item.text,
            )
            for item, c, sc in zip(r.items, r.contexts, r.context_scores, strict=True)
        ]

    def _finish(self, r: Retrieval, answer: str, generate_ms: float) -> QueryResponse:
        stage = next(s for s in r.trace.stages if s.key == "generate")
        stage.skipped = False
        stage.ms = round(generate_ms, 2)
        stage.info = {
            "llm": r.trace.llm,
            "prompt_tokens": r.trace.prompt_tokens,
            "context_passages": len(r.items),
        }
        r.trace.total_ms = round(sum(s.ms for s in r.trace.stages), 2)
        return QueryResponse(answer=answer, citations=self._citations(r), trace=r.trace)

    def query(self, request: QueryRequest) -> QueryResponse:
        """Answer a question (non-streaming)."""
        r = self.retrieve(request)
        t0 = time.perf_counter()
        answer = "".join(self._generate_tokens(r, request.question))
        return self._finish(r, answer, (time.perf_counter() - t0) * 1000)

    def query_stream(self, request: QueryRequest) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event, data)`` tuples: ``trace``, many ``token``, then ``done``."""
        r = self.retrieve(request)
        yield "trace", r.trace.model_dump()
        t0 = time.perf_counter()
        parts: list[str] = []
        for token in self._generate_tokens(r, request.question):
            parts.append(token)
            yield "token", {"text": token}
        response = self._finish(r, "".join(parts), (time.perf_counter() - t0) * 1000)
        yield "done", response.model_dump()

    def _generate_tokens(self, r: Retrieval, question: str) -> Iterator[str]:
        if not r.items:
            yield NO_ANSWER
            return
        if self.llm is not None:
            produced = False
            try:
                for token in self.llm.stream(r.system, r.user):
                    produced = True
                    yield token
                return
            except Exception as exc:
                r.trace.warnings.append(
                    f"LLM call failed ({type(exc).__name__}); used offline answer."
                )
                logger.warning("LLM call failed", exc_info=True)
                if produced:
                    yield "\n\n"
        yield from self.extractive.stream(self.extractive.answer(question, r.items))
