from __future__ import annotations

import math

import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.compression import compress_text
from app.retrieval.fusion import (
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from app.retrieval.rerank import CrossEncoderReranker, HeuristicReranker


def _bm25_term(tf: int, dl: int, avgdl: float, df: int, n: int, k1=1.2, b=0.75) -> float:
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_bm25_known_values() -> None:
    docs = ["apple banana", "apple cherry", "banana banana cherry"]
    index = BM25Index()
    index.index(docs)
    assert index.avgdl == pytest.approx(7 / 3)
    scores = index.score("apple")
    assert scores[0] == pytest.approx(_bm25_term(1, 2, 7 / 3, 2, 3))
    assert scores[1] == pytest.approx(_bm25_term(1, 2, 7 / 3, 2, 3))
    assert scores[2] == 0
    banana = index.score("banana")
    assert banana[2] == pytest.approx(_bm25_term(2, 3, 7 / 3, 2, 3))
    assert banana[2] > banana[0]  # more occurrences wins despite being longer


def test_bm25_length_normalisation_and_saturation() -> None:
    docs = ["rocket " + "filler " * 3, "rocket " + "filler " * 40, "rocket rocket rocket rocket"]
    index = BM25Index()
    index.index(docs)
    s = index.score("rocket")
    assert s[0] > s[1]  # same tf, shorter doc scores higher
    gain_1_to_2 = _bm25_term(2, 4, 4, 1, 1) - _bm25_term(1, 4, 4, 1, 1)
    gain_3_to_4 = _bm25_term(4, 4, 4, 1, 1) - _bm25_term(3, 4, 4, 1, 1)
    assert gain_1_to_2 > gain_3_to_4  # saturation


def test_bm25_search_and_mask() -> None:
    index = BM25Index()
    index.index(["red fox jumps", "blue whale swims", "red whale rests"])
    assert [i for i, _ in index.search("red whale", 3)][0] == 2
    mask = np.array([True, True, False])
    assert 2 not in [i for i, _ in index.search("red whale", 3, mask)]
    assert index.search("unknownterm", 3) == []


def test_bm25_empty_index() -> None:
    index = BM25Index()
    index.index([])
    assert index.search("anything", 3) == []


def test_rrf_known_values() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    ordered = [i for i, _ in reciprocal_rank_fusion([["x", "y"], ["y"]])]
    assert ordered == ["y", "x"]


def test_rrf_custom_k() -> None:
    assert dict(reciprocal_rank_fusion([["a"]], k=10))["a"] == pytest.approx(1 / 11)


def test_min_max_normalize() -> None:
    assert min_max_normalize({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert min_max_normalize({"a": 3.0}) == {"a": 1.0}
    assert min_max_normalize({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert min_max_normalize({}) == {}


def test_weighted_fusion_alpha_extremes() -> None:
    vec = {"a": 1.0, "b": 0.5}
    kw = {"a": 0.0, "b": 10.0}
    assert weighted_fusion(vec, kw, 1.0)[0][0] == "a"
    assert weighted_fusion(vec, kw, 0.0)[0][0] == "b"
    blended = dict(weighted_fusion(vec, kw, 0.5))
    assert blended["a"] == pytest.approx(0.5)
    assert blended["b"] == pytest.approx(0.5)


def test_weighted_fusion_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        weighted_fusion({}, {}, 1.5)


def test_heuristic_reranker_prefers_relevant_text() -> None:
    items = [
        ("noise", "Bananas are yellow and grow in tropical climates."),
        ("hit", "Reciprocal rank fusion merges two ranked lists using ranks only."),
        ("partial", "A list of fruit names and ranks."),
    ]
    ranked = HeuristicReranker().rerank("how does reciprocal rank fusion work", items)
    assert [i for i, _ in ranked][0] == "hit"
    assert ranked[0][1] > ranked[-1][1]


def test_heuristic_reranker_bigram_and_empty_query() -> None:
    r = HeuristicReranker()
    assert r.score("the of a", "anything at all") == 0.0
    assert r.score("vector search", "vector search is fast") > r.score(
        "vector search", "search the vector"
    )


def test_cross_encoder_falls_back_when_model_unavailable() -> None:
    reranker = CrossEncoderReranker("does-not/exist-model")
    ranked = reranker.rerank("fusion", [("a", "unrelated text"), ("b", "rank fusion method")])
    assert ranked[0][0] == "b"
    assert reranker.available is False
    assert reranker.failure
    assert "heuristic" in reranker.name


def test_compression_keeps_relevant_sentences() -> None:
    text = (
        "The museum opens at nine. Bananas are rich in potassium and fibre. "
        "Tickets cost ten dollars. The cafe sells coffee."
    )
    out = compress_text("how much potassium do bananas have", text)
    assert out == "Bananas are rich in potassium and fibre."


def test_compression_always_keeps_something() -> None:
    text = "First sentence here. Second sentence there."
    assert compress_text("zebra", text) == "First sentence here."
    assert compress_text("anything", "Single sentence.") == "Single sentence."
