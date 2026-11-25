"""Evaluation metrics for retrieval and generation.

Retrieval metrics compare the ranked chunk ids returned by a pipeline against the set of
chunk ids known to contain the answer:

* **Recall@k**    - did at least one relevant chunk appear in the top k? (hit-rate form)
* **Precision@k** - what fraction of the top k are relevant?
* **MRR**         - mean of 1/rank of the first relevant chunk.

Generation metrics (cheap, deterministic approximations of LLM-judge metrics):

* **Faithfulness**      - fraction of answer claims (sentences) whose content words overlap the
  retrieved context by at least 50%: a low score signals hallucination.
* **Answer relevance**  - share of question keywords that the answer addresses.

An optional LLM judge (:func:`llm_faithfulness`) can replace the overlap heuristic.
"""

from __future__ import annotations

from app.generation.llm import LLMClient
from app.generation.prompts import NO_ANSWER
from app.text import content_terms, split_sentences

FAITHFULNESS_THRESHOLD = 0.5


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any of the top-``k`` retrieved ids is relevant, else 0.0."""
    return 1.0 if relevant and any(r in relevant for r in retrieved[:k]) else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are relevant."""
    top = retrieved[:k]
    return sum(r in relevant for r in top) / len(top) if top else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant id (0.0 if none retrieved)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def faithfulness_overlap(
    answer: str, contexts: list[str], threshold: float = FAITHFULNESS_THRESHOLD
) -> float:
    """Fraction of answer sentences supported by the context (word-overlap >= ``threshold``)."""
    if answer.strip() == NO_ANSWER:
        return 1.0
    context_terms = {t for c in contexts for t in content_terms(c)}
    claims = [s for s in split_sentences(answer) if content_terms(s)]
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        terms = set(content_terms(claim))
        if len(terms & context_terms) / len(terms) >= threshold:
            supported += 1
    return supported / len(claims)


def answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content words that also appear in the answer."""
    q_terms = set(content_terms(question))
    if not q_terms or answer.strip() == NO_ANSWER:
        return 0.0
    return len(q_terms & set(content_terms(answer))) / len(q_terms)


def llm_faithfulness(answer: str, contexts: list[str], llm: LLMClient) -> float:
    """Optional LLM judge: ask the model to rate how well the context supports the answer (0-1)."""
    reply = llm.complete(
        "You are a strict evaluator. Reply with a single number between 0 and 1: the fraction of "
        "claims in the ANSWER that are supported by the CONTEXT.",
        "CONTEXT:\n" + "\n---\n".join(contexts) + f"\n\nANSWER:\n{answer}",
        max_tokens=8,
    )
    try:
        return min(max(float(reply.strip().split()[0]), 0.0), 1.0)
    except (ValueError, IndexError):
        return 0.0
