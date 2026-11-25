from __future__ import annotations

import pytest

from app.config import Settings
from app.embeddings.tfidf import TfidfEmbedder
from app.generation.prompts import NO_ANSWER
from app.ingestion.chunking import ChunkingParams
from app.pipeline import RAGPipeline
from app.schemas import HistoryTurn, QueryRequest, RetrievalSettings
from tests.fakes import FakeLLM


def ask(pipe: RAGPipeline, question: str, **settings: object):
    return pipe.query(QueryRequest(question=question, settings=RetrievalSettings(**settings)))  # type: ignore[arg-type]


def stage(response, key: str):
    return next(s for s in response.trace.stages if s.key == key)


def test_add_document_registers_metadata(pipeline: RAGPipeline, pdf_bytes: bytes) -> None:
    record = pipeline.add_document("nature.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    assert record.pages == 3 and record.chunks >= 3
    assert pipeline.documents[record.id].filename == "nature.pdf"
    assert pipeline.find_by_filename("nature.pdf") == record
    assert {c.page for c in pipeline.index.chunks} == {1, 2, 3}


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
@pytest.mark.parametrize("fusion", ["rrf", "weighted"])
def test_answer_with_citation_for_each_mode(indexed: RAGPipeline, mode: str, fusion: str) -> None:
    response = ask(
        indexed,
        "What do chlorophyll and chloroplasts absorb?",
        retrieval_mode=mode,
        fusion=fusion,
        rerank="none",
    )
    assert response.citations
    assert response.citations[0].page == 1
    assert response.citations[0].filename == "nature.pdf"
    assert "[1]" in response.answer
    trace = response.trace
    assert trace.embedder == "tfidf" and trace.llm == "Offline (extractive)"
    assert stage(response, "vector").skipped is (mode == "bm25")
    assert stage(response, "bm25").skipped is (mode == "vector")
    assert stage(response, "fusion").skipped is (mode != "hybrid")
    assert stage(response, "generate").skipped is False
    assert trace.prompt_tokens > 0 and trace.total_ms > 0


def test_unanswerable_question_gets_refusal(indexed: RAGPipeline) -> None:
    response = ask(indexed, "Who won the football world cup in 1998?")
    assert response.answer == NO_ANSWER


def test_rerank_stage_recorded(indexed: RAGPipeline) -> None:
    response = ask(indexed, "How do tides depend on the Moon gravity?", rerank="heuristic", top_k=2)
    st = stage(response, "rerank")
    assert not st.skipped and st.info["reranker"] == "heuristic"
    assert len(st.hits) == 2 and len(response.citations) == 2
    assert response.citations[0].page == 3


def test_cross_encoder_selection_never_crashes(indexed: RAGPipeline) -> None:
    indexed.cross_encoder.model_name = "does-not/exist-model"
    response = ask(indexed, "volcano magma eruption", rerank="cross_encoder")
    assert response.citations[0].page == 2
    assert any("Cross-encoder unavailable" in w for w in response.trace.warnings)


def test_metadata_filter_by_document_and_pages(indexed: RAGPipeline, pdf_bytes: bytes) -> None:
    other = indexed.add_document("other.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    doc_id = next(d.id for d in indexed.documents.values() if d.filename == "nature.pdf")
    r1 = ask(indexed, "moon tides", doc_ids=[doc_id], top_k=10)
    assert {c.filename for c in r1.citations} == {"nature.pdf"}
    r2 = ask(indexed, "moon tides", doc_ids=[other.id], top_k=10)
    assert {c.filename for c in r2.citations} == {"other.pdf"}
    r3 = ask(indexed, "moon tides volcano", page_min=2, page_max=2, top_k=10)
    assert {c.page for c in r3.citations} == {2}
    r4 = ask(indexed, "moon", doc_ids=["missing"])
    assert r4.citations == [] and r4.answer == NO_ANSWER


@pytest.mark.parametrize("transform", ["rewrite", "hyde", "multi_query"])
def test_query_transforms_offline(indexed: RAGPipeline, transform: str) -> None:
    response = ask(indexed, "How do volcanoes erupt?", query_transform=transform)
    trace = response.trace
    assert not stage(response, "transform").skipped
    if transform == "rewrite":
        assert trace.rewritten_query
    if transform == "hyde":
        assert trace.hyde_text and len(trace.hyde_text.split()) > 10
    if transform == "multi_query":
        assert trace.multi_queries
        assert stage(response, "fusion").info["queries"] == 1 + len(trace.multi_queries)
        assert not stage(response, "fusion").skipped
    assert response.citations


def test_transform_uses_llm_when_present(settings: Settings, pdf_bytes: bytes) -> None:
    llm = FakeLLM("volcano magma")
    pipe = RAGPipeline(settings, TfidfEmbedder(), llm)
    pipe.add_document("n.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    response = ask(pipe, "tell me things", query_transform="rewrite")
    assert response.trace.rewritten_query == "volcano magma"
    assert response.trace.llm == "fake"
    assert response.answer.startswith("fake answer")


def test_llm_failure_falls_back_to_extractive(settings: Settings, pdf_bytes: bytes) -> None:
    pipe = RAGPipeline(settings, TfidfEmbedder(), FakeLLM(fail=True))
    pipe.add_document("n.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    response = ask(pipe, "What absorbs red and blue light?", query_transform="none")
    assert "[1]" in response.answer
    assert any("LLM call failed" in w for w in response.trace.warnings)


def test_conversation_condensing(indexed: RAGPipeline) -> None:
    request = QueryRequest(
        question="What about its pull?",
        history=[
            HistoryTurn(role="user", content="Explain ocean tides"),
            HistoryTurn(role="assistant", content="Tides come from the Moon."),
        ],
    )
    response = indexed.query(request)
    assert "ocean tides" in response.trace.condensed_query
    assert response.citations[0].page == 3
    assert stage(response, "transform").info["condensed"] is True


def test_parent_child_expansion(pipeline: RAGPipeline, pdf_bytes: bytes) -> None:
    pipeline.add_document(
        "n.pdf", pdf_bytes, ChunkingParams(strategy="parent_child", parent_size=60, child_size=15)
    )
    plain = ask(pipeline, "What drives explosive eruptions?", top_k=1)
    expanded = ask(pipeline, "What drives explosive eruptions?", top_k=1, parent_child=True)
    st = stage(expanded, "parent")
    assert not st.skipped and st.info["parents"] == 1
    assert len(expanded.citations[0].text.split()) > len(plain.citations[0].text.split())
    assert expanded.citations[0].chunk_id.split("-")[1].startswith("p")
    assert stage(plain, "parent").skipped


def test_parent_child_flag_ignored_for_flat_chunks(indexed: RAGPipeline) -> None:
    response = ask(indexed, "tides moon", parent_child=True)
    assert stage(response, "parent").skipped


def test_compression_shrinks_context(indexed: RAGPipeline) -> None:
    plain = ask(indexed, "What do volcanoes monitor seismic tremors?", top_k=1)
    compressed = ask(
        indexed, "What do volcanoes monitor seismic tremors?", top_k=1, compression=True
    )
    st = stage(compressed, "compression")
    assert not st.skipped and st.info["words_after"] <= st.info["words_before"]
    assert len(compressed.citations[0].text) <= len(plain.citations[0].text)


def test_similarity_metrics_selectable(indexed: RAGPipeline) -> None:
    for metric in ("cosine", "dot", "l2"):
        response = ask(indexed, "chlorophyll", retrieval_mode="vector", similarity=metric)
        assert stage(response, "vector").info["metric"] == metric
        assert response.citations[0].page == 1


def test_stream_events_order(indexed: RAGPipeline) -> None:
    events = list(indexed.query_stream(QueryRequest(question="What absorbs red and blue light?")))
    names = [n for n, _ in events]
    assert names[0] == "trace" and names[-1] == "done" and "token" in names
    streamed = "".join(d["text"] for n, d in events if n == "token")
    assert streamed == events[-1][1]["answer"]
    assert events[-1][1]["citations"]


def test_persistence_roundtrip_and_delete(settings: Settings, pdf_bytes: bytes) -> None:
    first = RAGPipeline(settings, TfidfEmbedder(), None)
    record = first.add_document("n.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    reloaded = RAGPipeline(settings, TfidfEmbedder(), None)
    assert list(reloaded.documents) == [record.id]
    assert ask(reloaded, "chlorophyll").citations
    assert reloaded.delete_document(record.id) is True
    assert reloaded.delete_document(record.id) is False
    assert not RAGPipeline(settings, TfidfEmbedder(), None).documents
    assert len(reloaded.index.chunks) == 0


def test_corpus_idf_refits_when_documents_change(indexed: RAGPipeline, pdf_bytes: bytes) -> None:
    before = indexed.embedder.dim  # type: ignore[attr-defined]
    indexed.add_document("extra.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    assert indexed.embedder.dim == before  # same vocabulary, idf recomputed
    assert len(indexed.index.vectors) == len(indexed.index.chunks)
