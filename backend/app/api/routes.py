"""HTTP API routes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app import __version__
from app.evaluation.evaluator import auto_generate_qa, evaluate, load_qa
from app.generation.llm import provider_label
from app.ingestion.chunking import ChunkingParams, Strategy
from app.pipeline import RAGPipeline
from app.schemas import (
    DocumentInfo,
    EvalReport,
    EvalRequest,
    QueryRequest,
    QueryResponse,
)
from app.store import DocumentRecord

router = APIRouter(prefix="/api")

SAMPLE_PDF = "rag_handbook.pdf"


def get_pipeline(request: Request) -> RAGPipeline:
    """The pipeline attached to the running app."""
    pipeline: RAGPipeline = request.app.state.pipeline
    return pipeline


def _info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(**asdict(record))


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@router.get("/config")
def config(request: Request) -> dict[str, object]:
    """Which LLM, embedding and reranking backends are active."""
    pipeline = get_pipeline(request)
    s = pipeline.settings
    return {
        "version": __version__,
        "llm": {
            "provider": s.llm_provider,
            "label": provider_label(s),
            "model": (s.llm_model or None) if pipeline.llm else None,
            "offline": pipeline.llm is None,
        },
        "embeddings": {
            "requested": s.embedding_backend,
            "active": pipeline.embedder.name,
            "dense": not pipeline.embedder.needs_fit,
            "fallback_reason": pipeline.embedder_fallback_reason,
        },
        "reranker": {
            "model": s.reranker_model,
            "cross_encoder_available": pipeline.cross_encoder.available,
        },
        "sample_available": (s.samples_dir / SAMPLE_PDF).exists(),
    }


@router.get("/documents")
def list_documents(request: Request) -> list[DocumentInfo]:
    """All indexed documents."""
    return [_info(d) for d in get_pipeline(request).documents.values()]


@router.post("/documents", status_code=201)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    strategy: Annotated[Strategy, Form()] = "recursive",
    chunk_size: Annotated[int, Form(ge=10, le=2000)] = 120,
    chunk_overlap: Annotated[int, Form(ge=0, le=500)] = 20,
    parent_size: Annotated[int, Form(ge=20, le=4000)] = 200,
    child_size: Annotated[int, Form(ge=5, le=1000)] = 50,
) -> DocumentInfo:
    """Upload a PDF and index it with the chosen chunking strategy."""
    pipeline = get_pipeline(request)
    data = await file.read()
    if len(data) > pipeline.settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {pipeline.settings.max_upload_mb} MB limit")
    params = ChunkingParams(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_size=parent_size,
        child_size=child_size,
    )
    try:
        params.validate()
        record = pipeline.add_document(file.filename or "document.pdf", data, params)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _info(record)


@router.post("/documents/sample", status_code=201)
def load_sample(request: Request) -> DocumentInfo:
    """Index the bundled sample PDF (parent-child chunking so every feature can be tried)."""
    pipeline = get_pipeline(request)
    existing = pipeline.find_by_filename(SAMPLE_PDF)
    if existing:
        return _info(existing)
    path = pipeline.settings.samples_dir / SAMPLE_PDF
    if not path.exists():
        raise HTTPException(404, "Sample PDF is not available on this server")
    return _info(pipeline.add_pdf_file(path, ChunkingParams(strategy="parent_child")))


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(request: Request, doc_id: str) -> None:
    """Remove a document and its chunks."""
    if not get_pipeline(request).delete_document(doc_id):
        raise HTTPException(404, "Document not found")


@router.post("/query")
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer a question with citations and the full pipeline trace."""
    pipeline = get_pipeline(request)
    _require_documents(pipeline)
    return pipeline.query(body)


@router.post("/query/stream")
def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Server-sent events: ``trace`` first, then ``token`` events, then ``done``."""
    pipeline = get_pipeline(request)
    _require_documents(pipeline)

    def events() -> Iterator[str]:
        for name, data in pipeline.query_stream(body):
            yield f"event: {name}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/evaluate")
def run_evaluation(request: Request, body: EvalRequest) -> EvalReport:
    """Recall@k comparison of vector / BM25 / hybrid / hybrid+rerank on the indexed documents."""
    pipeline = get_pipeline(request)
    _require_documents(pipeline)
    docs = [d for d in pipeline.documents.values() if not body.doc_ids or d.id in body.doc_ids]
    chunks = [c for c in pipeline.index.chunks if c.doc_id in {d.id for d in docs}]
    sample_qa = pipeline.settings.samples_dir / "rag_handbook_qa.json"
    if sample_qa.exists() and any(d.filename == SAMPLE_PDF for d in docs):
        qa, source = load_qa(sample_qa), sample_qa.name
    else:
        qa, source = auto_generate_qa(chunks, body.max_questions), "auto-generated"
    if not qa:
        raise HTTPException(422, "Could not build evaluation questions from the indexed text")
    return evaluate(
        pipeline,
        qa[: body.max_questions],
        ks=sorted(set(body.ks)),
        doc_ids=[d.id for d in docs],
        source=source,
    )


def _require_documents(pipeline: RAGPipeline) -> None:
    if not pipeline.documents:
        raise HTTPException(409, "No documents indexed yet. Upload a PDF or load the sample.")
