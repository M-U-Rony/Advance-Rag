"""LLM abstraction: OpenAI-compatible APIs, Anthropic, and an offline extractive fallback.

``openai_compatible`` works with anything that speaks the OpenAI chat-completions protocol:
OpenAI, Groq, Google Gemini (OpenAI-compat endpoint), Ollama, vLLM, LM Studio...
When no provider is configured the :class:`ExtractiveGenerator` composes an answer from the
most query-relevant sentences of the retrieved context, so the whole demo works offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Protocol

from app.config import Settings
from app.generation.prompts import NO_ANSWER, ContextItem
from app.retrieval.compression import sentence_scores
from app.text import split_sentences


class LLMClient(Protocol):
    """Minimal chat-completion interface used by the pipeline."""

    name: str

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        """Return the full completion."""

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        """Yield completion text fragments as they arrive."""


class OpenAICompatibleClient:
    """Chat client for any OpenAI-style endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str | None, label: str) -> None:
        from openai import OpenAI

        self.name = label
        self.model = model
        self._client: Any = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        """Single non-streaming completion."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        """Streaming completion."""
        events = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            max_tokens=max_tokens,
            temperature=0.2,
            stream=True,
        )
        for event in events:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class AnthropicClient:
    """Chat client for the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic

        self.name = "Anthropic"
        self.model = model
        self._client: Any = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        """Single non-streaming completion."""
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        """Streaming completion."""
        with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        ) as stream:
            yield from stream.text_stream


def provider_label(settings: Settings) -> str:
    """Human-friendly provider name for the UI badge."""
    if settings.llm_provider == "none":
        return "Offline"
    if settings.llm_provider == "anthropic":
        return "Anthropic"
    if settings.llm_provider == "openai":
        return "OpenAI"
    url = settings.llm_base_url.lower()
    for needle, label in (
        ("groq", "Groq"),
        ("generativelanguage", "Gemini"),
        ("localhost", "Ollama"),
        ("127.0.0.1", "Ollama"),
        ("openai.com", "OpenAI"),
    ):
        if needle in url:
            return label
    return "OpenAI-compatible"


def create_llm(settings: Settings) -> LLMClient | None:
    """Build the configured LLM client, or ``None`` to use the offline extractive generator."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "anthropic":
        return AnthropicClient(
            settings.llm_api_key, settings.llm_model or "claude-3-5-haiku-latest"
        )
    base_url = settings.llm_base_url or (
        "https://api.openai.com/v1" if settings.llm_provider == "openai" else ""
    )
    return OpenAICompatibleClient(
        settings.llm_api_key,
        settings.llm_model or "gpt-4o-mini",
        base_url,
        provider_label(settings),
    )


class ExtractiveGenerator:
    """Offline "generator": answers by quoting the most query-relevant context sentences.

    It is deterministic, needs no network, and still produces cited, grounded answers; it
    cannot paraphrase or reason across passages like a real LLM.
    """

    name = "Offline (extractive)"

    def __init__(self, max_sentences: int = 3, min_score: float = 0.3) -> None:
        self.max_sentences = max_sentences
        self.min_score = min_score

    def answer(self, question: str, items: list[ContextItem]) -> str:
        """Compose a cited answer, or the standard refusal when nothing is relevant."""
        candidates: list[tuple[float, int, int, str]] = []
        for item in items:
            sentences = split_sentences(item.text)
            for pos, (sent, score) in enumerate(
                zip(sentences, sentence_scores(question, sentences), strict=True)
            ):
                if score >= self.min_score:
                    candidates.append((score - 0.01 * item.index, item.index, pos, sent))
        if not candidates:
            return NO_ANSWER
        best = sorted(candidates, key=lambda c: -c[0])[: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))
        return " ".join(f"{sent.rstrip()} [{idx}]" for _, idx, _, sent in best)

    def stream(self, text: str) -> Iterator[str]:
        """Yield an already-composed answer word by word (to exercise the streaming path)."""
        yield from re.findall(r"\S+\s*", text)
