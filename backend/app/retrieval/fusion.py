"""Fusing several ranked lists into one: the heart of hybrid search.

Vector scores (cosine in [-1, 1]) and BM25 scores (unbounded) live on different scales, so
they cannot simply be added. Two standard solutions are implemented:

* **Weighted (alpha)**: min-max normalise each score list to [0, 1] then blend
  ``alpha * vector + (1 - alpha) * keyword`` (alpha=1 pure vector, alpha=0 pure keyword).
* **Reciprocal Rank Fusion (RRF)**: ignore scores, use only ranks:
  ``score(d) = sum over lists of 1 / (k + rank(d))`` with k=60. Robust, parameter-light and
  the default in most production systems.
"""

from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first) with RRF; returns ``(id, score)`` best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1]; a constant list maps to 1.0 (or 0.0 when all are zero)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0 if hi > 0 else 0.0)
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def weighted_fusion(
    vector_scores: dict[str, float], keyword_scores: dict[str, float], alpha: float
) -> list[tuple[str, float]]:
    """Blend normalised vector and keyword scores: ``alpha*vector + (1-alpha)*keyword``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    vec = min_max_normalize(vector_scores)
    kw = min_max_normalize(keyword_scores)
    fused = {
        key: alpha * vec.get(key, 0.0) + (1 - alpha) * kw.get(key, 0.0)
        for key in vec.keys() | kw.keys()
    }
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
