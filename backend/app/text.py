"""Shared text utilities: tokenisation, stop words, sentence splitting.

Every lexical component (TF-IDF, BM25, the heuristic reranker, contextual
compression, the extractive generator) shares one tokenizer so that they all
agree on what a "term" is.
"""

from __future__ import annotations

import re

_STOPWORD_TEXT = """
a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what when where which while who
whom why will with would you your tell explain describe give please
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric terms."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def content_terms(text: str) -> list[str]:
    """Tokens of ``text`` without stop words (the informative terms)."""
    return tokenize(text, remove_stopwords=True)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using punctuation followed by a capital letter."""
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(flat) if s.strip()]


def word_count(text: str) -> int:
    """Number of whitespace separated words."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate (about 1.3 tokens per word for English)."""
    return int(word_count(text) * 1.3) + 1


def snippet(text: str, limit: int = 220) -> str:
    """Single-line preview of ``text`` truncated to ``limit`` characters."""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
