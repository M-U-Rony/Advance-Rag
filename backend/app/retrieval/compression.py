"""Contextual compression: keep only the sentences of a chunk that matter for the query.

Retrieved chunks often contain a relevant sentence surrounded by filler. Dropping the
filler cuts prompt tokens (cost and latency) and reduces the "lost in the middle" effect.
This implementation is extractive: sentences are kept when they share informative terms
with the query; at least the best sentence is always kept.
"""

from __future__ import annotations

from app.text import content_terms, split_sentences


def sentence_scores(query: str, sentences: list[str]) -> list[float]:
    """Fraction of query content terms present in each sentence."""
    q_terms = set(content_terms(query))
    if not q_terms:
        return [0.0] * len(sentences)
    return [len(q_terms & set(content_terms(s))) / len(q_terms) for s in sentences]


def compress_text(query: str, text: str, min_score: float = 0.15, max_sentences: int = 4) -> str:
    """Return the query-relevant sentences of ``text`` in their original order."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    scores = sentence_scores(query, sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    keep = {i for i in ranked[:max_sentences] if scores[i] >= min_score}
    if not keep:
        keep = {ranked[0]}
    return " ".join(sentences[i] for i in sorted(keep))
