from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.embeddings.dense import create_embedder
from app.embeddings.tfidf import TfidfEmbedder
from app.retrieval.filters import MetadataFilter
from app.retrieval.vector_store import VectorStore, similarity


def test_tfidf_idf_formula_and_normalisation() -> None:
    emb = TfidfEmbedder()
    emb.fit(["cat dog", "cat bird"])
    assert emb.dim == 3
    # idf = ln((n + 1) / (df + 1)) + 1, with n = 2 documents
    assert emb.idf[emb.vocab["cat"]] == pytest.approx(1.0)
    assert emb.idf[emb.vocab["dog"]] == pytest.approx(math.log(3 / 2) + 1)
    vec = emb.embed_query("cat dog")
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    expected = np.array([1.0, math.log(1.5) + 1])
    expected /= np.linalg.norm(expected)
    assert vec[emb.vocab["cat"]] == pytest.approx(expected[0])
    assert vec[emb.vocab["dog"]] == pytest.approx(expected[1])


def test_tfidf_unknown_terms_and_empty() -> None:
    emb = TfidfEmbedder()
    emb.fit(["alpha beta"])
    assert not emb.embed_query("zzz").any()
    assert emb.embed_documents([]).shape == (0, 2)


def test_tfidf_similar_texts_score_higher() -> None:
    emb = TfidfEmbedder()
    docs = ["red apple pie recipe", "blue ocean waves", "apple pie with cream"]
    emb.fit(docs)
    scores = emb.embed_documents(docs) @ emb.embed_query("apple pie")
    assert scores[1] == 0
    assert scores[0] > 0 and scores[2] > 0


def test_create_embedder_tfidf(settings: Settings) -> None:
    emb, reason = create_embedder(settings)
    assert emb.name == "tfidf" and reason is None


def test_create_embedder_falls_back_when_model_missing(settings: Settings) -> None:
    bad = settings.model_copy(
        update={"embedding_backend": "dense", "embedding_model": "does-not/exist-model"}
    )
    emb, reason = create_embedder(bad)
    assert emb.name == "tfidf"
    assert reason


MATRIX = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float32)


def _store() -> VectorStore:
    store = VectorStore()
    store.add(
        ["a", "b", "c"],
        MATRIX,
        [{"doc_id": "d1", "page": 1}, {"doc_id": "d1", "page": 5}, {"doc_id": "d2", "page": 2}],
    )
    return store


def test_similarity_metrics_known_values() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert similarity(MATRIX, q, "cosine").tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert similarity(MATRIX, q, "dot").tolist() == pytest.approx([1.0, 0.0, 2.0])
    assert similarity(MATRIX, q, "l2").tolist() == pytest.approx([0.0, -math.sqrt(2), -1.0])
    with pytest.raises(ValueError):
        similarity(MATRIX, q, "manhattan")  # type: ignore[arg-type]


def test_search_orders_by_metric() -> None:
    store = _store()
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert [i for i, _ in store.search(q, 3, "dot")] == ["c", "a", "b"]
    assert [i for i, _ in store.search(q, 3, "l2")] == ["a", "c", "b"]
    assert store.search(q, 0) == []


def test_search_metadata_filter() -> None:
    store = _store()
    q = np.array([1.0, 0.0], dtype=np.float32)
    only_d1 = MetadataFilter(doc_ids=frozenset({"d1"}))
    assert {i for i, _ in store.search(q, 5, "cosine", only_d1)} == {"a", "b"}
    pages = MetadataFilter(page_min=2, page_max=5)
    assert {i for i, _ in store.search(q, 5, "cosine", pages)} == {"b", "c"}
    assert store.search(q, 5, "cosine", MetadataFilter(doc_ids=frozenset({"zzz"}))) == []


def test_add_validates_lengths() -> None:
    with pytest.raises(ValueError):
        VectorStore().add(["a"], MATRIX, [{}])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = _store()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.ids == store.ids
    assert loaded.metadata == store.metadata
    assert np.allclose(loaded.matrix, store.matrix)
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert loaded.search(q, 2) == store.search(q, 2)


def test_metadata_filter_inactive_matches_everything() -> None:
    flt = MetadataFilter()
    assert not flt.active
    assert flt.matches({"doc_id": "x", "page": 3})
