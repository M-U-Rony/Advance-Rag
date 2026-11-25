"""Dense (non-fitted) embedders: vectors are cached and only new chunks are embedded."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generation.llm import AnthropicClient, OpenAICompatibleClient
from app.ingestion.chunking import ChunkingParams
from app.pipeline import RAGPipeline
from app.schemas import QueryRequest
from tests.conftest import make_pdf
from tests.fakes import HashEmbedder


def test_incremental_embedding_and_disk_cache(settings: Settings, pdf_bytes: bytes) -> None:
    emb = HashEmbedder()
    pipe = RAGPipeline(settings, emb, None)
    first = pipe.add_document("a.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    after_first = emb.embedded_texts
    assert after_first == first.chunks

    other_pdf = make_pdf([("Bees", "Honey bees dance to communicate the location of nectar.")])
    second = pipe.add_document("b.pdf", other_pdf, ChunkingParams(chunk_size=30))
    assert emb.embedded_texts == after_first + second.chunks  # old vectors reused

    pipe.delete_document(first.id)
    assert emb.embedded_texts == after_first + second.chunks  # deletion re-embeds nothing
    assert len(pipe.index.vectors) == second.chunks

    reloaded_emb = HashEmbedder()
    reloaded = RAGPipeline(settings, reloaded_emb, None)
    assert reloaded_emb.embedded_texts == 0  # served from the persisted vector store
    answer = reloaded.query(QueryRequest(question="How do bees communicate nectar location?"))
    assert answer.citations and answer.citations[0].filename == "b.pdf"
    assert answer.trace.embedder == "hash-dense"


def test_corrupt_cache_is_ignored(settings: Settings, pdf_bytes: bytes) -> None:
    pipe = RAGPipeline(settings, HashEmbedder(), None)
    pipe.add_document("a.pdf", pdf_bytes, ChunkingParams(chunk_size=30))
    cache = settings.data_dir / "index" / "hash-dense" / "vectors.npz"
    cache.write_bytes(b"garbage")
    emb = HashEmbedder()
    reloaded = RAGPipeline(settings, emb, None)
    assert emb.embedded_texts == len(reloaded.index.chunks) > 0


def _chunk(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_openai_compatible_client_with_fake_sdk() -> None:
    client = OpenAICompatibleClient("k", "m", "http://localhost:1/v1", "Test")
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_chunk("Hel"), _chunk(None), _chunk("lo"), SimpleNamespace(choices=[])])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" hi "))])

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert client.complete("sys", "user") == "hi"
    assert "".join(client.stream("sys", "user")) == "Hello"
    assert calls[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_anthropic_client_with_fake_sdk() -> None:
    client = AnthropicClient("k", "claude-test")

    class _Stream:
        text_stream = iter(["a", "b"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    messages = SimpleNamespace(
        create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text="ok ")]),
        stream=lambda **kw: _Stream(),
    )
    client._client = SimpleNamespace(messages=messages)
    assert client.complete("s", "u") == "ok"
    assert "".join(client.stream("s", "u")) == "ab"


@pytest.mark.parametrize("k", [0])
def test_empty_corpus_query_returns_refusal(pipeline: RAGPipeline, k: int) -> None:
    response = pipeline.query(QueryRequest(question="anything"))
    assert response.citations == [] and "don't have enough information" in response.answer
