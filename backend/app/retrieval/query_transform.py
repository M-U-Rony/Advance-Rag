"""Query transformation: improve the query *before* searching.

User questions are often a poor search key: they are conversational, ambiguous, or phrased
differently from the documents. Four techniques are implemented; each uses the LLM when one
is configured and a deterministic heuristic otherwise, so everything works offline.

* **Condense with history** - turn a follow-up ("what about its cost?") into a standalone query.
* **Rewrite** - reformulate the question into a keyword-rich search query.
* **HyDE** (Hypothetical Document Embeddings) - have the LLM write a *hypothetical answer* and
  embed that instead of the question: an answer looks more like the documents than a question does.
* **Multi-query** - generate several paraphrases, retrieve for each, and fuse the lists with RRF.
"""

from __future__ import annotations

import re

from app.generation.llm import LLMClient
from app.generation.prompts import CONDENSE_SYSTEM, Turn, recent_history
from app.text import content_terms

_QUESTION_WORDS = re.compile(
    r"^(what|which|who|whom|when|where|why|how|is|are|was|were|do|does|did|can|could|should)\b"
    r"(\s+(?:is|are|was|were|does|did|do|the|a|an)\b)*\s*",
    re.IGNORECASE,
)
_FOLLOW_UP_HINT = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|there|former|latter|the same)\b",
    re.IGNORECASE,
)


def keywords_of(question: str) -> str:
    """Strip question scaffolding and stop words, leaving the topical keywords."""
    stripped = _QUESTION_WORDS.sub("", question.strip().rstrip("?"))
    terms = content_terms(stripped) or content_terms(question)
    return " ".join(dict.fromkeys(terms))


def _llm_text(llm: LLMClient, system: str, user: str, max_tokens: int = 200) -> str | None:
    try:
        out = llm.complete(system, user, max_tokens=max_tokens).strip()
    except Exception:
        return None
    return out or None


def condense_with_history(question: str, history: list[Turn], llm: LLMClient | None) -> str:
    """Make a follow-up question standalone using the last three exchanges."""
    turns = recent_history(history)
    if not turns:
        return question
    if llm is not None:
        convo = "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)
        rewritten = _llm_text(llm, CONDENSE_SYSTEM, f"{convo}\n\nFollow-up: {question}")
        if rewritten:
            return rewritten.strip().strip('"')
    if _FOLLOW_UP_HINT.search(question) or len(content_terms(question)) < 3:
        last_user = next((t.content for t in reversed(turns) if t.role == "user"), "")
        topic = keywords_of(last_user)
        if topic:
            return f"{question.rstrip('?').strip()} ({topic})"
    return question


def rewrite_query(question: str, llm: LLMClient | None) -> str:
    """Rewrite a question into a concise, keyword-rich search query."""
    if llm is not None:
        out = _llm_text(
            llm,
            "Rewrite the question as a concise search query with the most informative keywords "
            "and synonyms. Output only the query.",
            question,
            max_tokens=60,
        )
        if out:
            return out.splitlines()[0].strip().strip('"')
    return keywords_of(question) or question


def hyde_document(question: str, llm: LLMClient | None) -> str:
    """Hypothetical Document Embeddings: a made-up answer used as the search vector."""
    if llm is not None:
        out = _llm_text(
            llm,
            "Write a short, factual passage (3-4 sentences) that would answer the question, "
            "as it might appear in a technical document. Do not mention that it is hypothetical.",
            question,
            max_tokens=180,
        )
        if out:
            return out
    topic = keywords_of(question) or question
    return (
        f"{topic.capitalize()} is described in the document. The main points about {topic} "
        f"are explained in detail, including how {topic} works and why it matters."
    )


def multi_query(question: str, llm: LLMClient | None, n: int = 3) -> list[str]:
    """Generate ``n`` alternative phrasings of ``question`` (original not included)."""
    if llm is not None:
        out = _llm_text(
            llm,
            f"Generate {n} different search queries that capture other ways to ask the question. "
            "One per line, no numbering.",
            question,
            max_tokens=150,
        )
        if out:
            lines = [re.sub(r"^[\d.\-*)\s]+", "", ln).strip() for ln in out.splitlines()]
            variants = [ln for ln in lines if ln and ln.lower() != question.lower()]
            if variants:
                return variants[:n]
    topic = keywords_of(question) or question
    terms = topic.split()
    candidates = [
        topic,
        f"definition and explanation of {topic}",
        f"how does {topic} work",
        " ".join(reversed(terms)) if len(terms) > 1 else f"{topic} overview",
    ]
    unique = [c for c in dict.fromkeys(candidates) if c.lower() != question.lower()]
    return unique[:n]
