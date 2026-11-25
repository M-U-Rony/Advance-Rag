"""Evaluation harness: compare retrieval configurations on a question set.

Four configurations are compared, mirroring the progression taught in advanced-RAG courses:

1. ``vector``           - dense (or TF-IDF) similarity search only
2. ``bm25``             - keyword search only
3. ``hybrid``           - vector + BM25 fused with RRF
4. ``hybrid + rerank``  - hybrid candidates reranked by the cross-encoder (heuristic fallback)

A question's *relevant chunks* are those whose text contains the question's ``evidence``
phrase, which keeps the gold labels independent of the chunking strategy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation import metrics
from app.models import Chunk
from app.pipeline import RAGPipeline
from app.retrieval.query_transform import keywords_of
from app.schemas import (
    EvalReport,
    EvalRow,
    QueryRequest,
    RetrievalSettings,
)
from app.text import content_terms, split_sentences


@dataclass(frozen=True, slots=True)
class QAItem:
    """A question and the phrase that must appear in a relevant chunk."""

    question: str
    evidence: str


CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("vector", {"retrieval_mode": "vector", "rerank": "none"}),
    ("bm25", {"retrieval_mode": "bm25", "rerank": "none"}),
    ("hybrid", {"retrieval_mode": "hybrid", "fusion": "rrf", "rerank": "none"}),
    ("hybrid + rerank", {"retrieval_mode": "hybrid", "fusion": "rrf", "rerank": "cross_encoder"}),
)


def load_qa(path: Path) -> list[QAItem]:
    """Load ``[{"question": ..., "evidence": ...}, ...]`` from a JSON file."""
    return [QAItem(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def auto_generate_qa(chunks: list[Chunk], n: int, seed: int = 7) -> list[QAItem]:
    """Create keyword-style questions from chunk sentences (used when no QA file exists).

    For a random sample of chunks, the most informative sentence is turned into
    ``"What is <keywords>?"``. Questions are lexically close to their evidence, so this
    favours keyword search; a hand-written QA file gives a fairer comparison.
    """
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) >= 20]
    rng.shuffle(pool)
    items: list[QAItem] = []
    for chunk in pool:
        sentences = [s for s in split_sentences(chunk.text) if len(content_terms(s)) >= 4]
        if not sentences:
            continue
        sentence = max(sentences, key=lambda s: len(set(content_terms(s))))
        terms = keywords_of(sentence).split()
        picked = terms[: min(5, len(terms))]
        if len(picked) >= 3:
            items.append(
                QAItem(question=f"What is said about {' '.join(picked)}?", evidence=sentence[:60])
            )
        if len(items) >= n:
            break
    return items


def relevant_ids(chunks: list[Chunk], evidence: str) -> set[str]:
    """Ids of chunks whose text contains ``evidence`` (case/whitespace-insensitive)."""
    needle = " ".join(evidence.lower().split())
    return {c.id for c in chunks if needle in " ".join(c.text.lower().split())}


def evaluate(
    pipeline: RAGPipeline,
    qa: list[QAItem],
    *,
    ks: list[int],
    doc_ids: list[str] | None = None,
    source: str = "provided",
) -> EvalReport:
    """Run every configuration over ``qa`` and aggregate retrieval + generation metrics."""
    chunks = [c for c in pipeline.index.chunks if not doc_ids or c.doc_id in doc_ids]
    labelled = [(item, relevant_ids(chunks, item.evidence)) for item in qa]
    labelled = [(item, rel) for item, rel in labelled if rel]
    top_k = max(ks)
    rows: list[EvalRow] = []
    for name, overrides in CONFIGS:
        settings = RetrievalSettings(top_k=top_k, doc_ids=doc_ids, **overrides)
        if labelled:  # warm-up so lazy model loading does not distort the latency column
            pipeline.query(QueryRequest(question=labelled[0][0].question, settings=settings))
        recalls = dict.fromkeys(ks, 0.0)
        precision = mrr = faith = relevance = elapsed = 0.0
        for item, rel in labelled:
            result = pipeline.query(QueryRequest(question=item.question, settings=settings))
            ids = result.trace.ranked_chunk_ids
            for k in ks:
                recalls[k] += metrics.recall_at_k(ids, rel, k)
            precision += metrics.precision_at_k(ids, rel, top_k)
            mrr += metrics.reciprocal_rank(ids, rel)
            contexts = [c.text for c in result.citations]
            faith += metrics.faithfulness_overlap(result.answer, contexts)
            relevance += metrics.answer_relevance(item.question, result.answer)
            elapsed += result.trace.total_ms
        n = max(len(labelled), 1)
        rows.append(
            EvalRow(
                config=name,
                recall={f"@{k}": round(v / n, 3) for k, v in recalls.items()},
                precision=round(precision / n, 3),
                mrr=round(mrr / n, 3),
                faithfulness=round(faith / n, 3),
                answer_relevance=round(relevance / n, 3),
                avg_ms=round(elapsed / n, 1),
            )
        )
    return EvalReport(
        embedder=pipeline.embedder.name,
        questions=len(labelled),
        ks=ks,
        precision_k=top_k,
        source=source,
        rows=rows,
    )
