"""Prompt construction: turning retrieved context into a grounded prompt.

Grounding means the LLM may only use the supplied context, every source is labelled
``[Source i]`` so the answer can cite ``[i]``, and an explicit refusal sentence is defined
for questions the context cannot answer (the main defence against hallucination).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.text import estimate_tokens

NO_ANSWER = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's documents.\n"
    "Rules:\n"
    "1. Use ONLY the information inside the numbered context sources below. Do not use "
    "outside knowledge.\n"
    "2. Cite the sources that support each statement with bracketed numbers such as [1] or "
    "[2][3].\n"
    f'3. If the context does not contain the answer, reply exactly: "{NO_ANSWER}"\n'
    "4. Be concise and precise."
)

CONDENSE_SYSTEM = (
    "Rewrite the user's follow-up question as a single standalone question that can be "
    "understood without the conversation. Resolve pronouns and references. Output only the "
    "rewritten question."
)

MAX_HISTORY_EXCHANGES = 3


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One context passage handed to the LLM."""

    index: int
    filename: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class Turn:
    """One chat message."""

    role: str
    content: str


def recent_history(history: list[Turn], exchanges: int = MAX_HISTORY_EXCHANGES) -> list[Turn]:
    """Keep the last ``exchanges`` user/assistant pairs (2 messages each)."""
    return history[-2 * exchanges :]


def format_context(items: list[ContextItem]) -> str:
    """Render labelled sources: ``[Source 1] (file.pdf, p. 3)`` followed by the passage."""
    return "\n\n".join(f"[Source {c.index}] ({c.filename}, p. {c.page})\n{c.text}" for c in items)


def build_grounded_prompt(
    question: str, items: list[ContextItem], history: list[Turn] | None = None
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the answer-generation call."""
    parts = [f"Context:\n{format_context(items)}" if items else "Context:\n(no relevant passages)"]
    turns = recent_history(history or [])
    if turns:
        convo = "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)
        parts.append(f"Conversation so far:\n{convo}")
    parts.append(f"Question: {question}\nAnswer:")
    return SYSTEM_PROMPT, "\n\n".join(parts)


def prompt_token_estimate(system: str, user: str) -> int:
    """Approximate token count of the full prompt."""
    return estimate_tokens(system) + estimate_tokens(user)
