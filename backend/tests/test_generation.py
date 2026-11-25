from __future__ import annotations

import pytest

from app.config import Settings
from app.generation.llm import ExtractiveGenerator, create_llm, provider_label
from app.generation.prompts import (
    NO_ANSWER,
    ContextItem,
    Turn,
    build_grounded_prompt,
    format_context,
    prompt_token_estimate,
    recent_history,
)
from app.retrieval.query_transform import (
    condense_with_history,
    hyde_document,
    keywords_of,
    multi_query,
    rewrite_query,
)
from tests.fakes import FakeLLM

ITEMS = [
    ContextItem(1, "a.pdf", 2, "RRF uses the constant k equal to 60. It ignores raw scores."),
    ContextItem(2, "b.pdf", 7, "Bananas are yellow."),
]


def test_format_context_labels_sources() -> None:
    rendered = format_context(ITEMS)
    assert rendered.startswith("[Source 1] (a.pdf, p. 2)\nRRF uses")
    assert "[Source 2] (b.pdf, p. 7)" in rendered


def test_grounded_prompt_contains_rules_and_question() -> None:
    system, user = build_grounded_prompt("What is k?", ITEMS)
    assert "ONLY" in system
    assert NO_ANSWER in system
    assert "[1]" in system
    assert "Question: What is k?" in user
    assert "[Source 1]" in user
    assert prompt_token_estimate(system, user) > 20


def test_prompt_with_no_context_and_history() -> None:
    history = [Turn("user", "hi"), Turn("assistant", "hello")]
    _, user = build_grounded_prompt("q", [], history)
    assert "(no relevant passages)" in user
    assert "User: hi" in user and "Assistant: hello" in user


def test_recent_history_keeps_last_three_exchanges() -> None:
    turns = [Turn("user" if i % 2 == 0 else "assistant", str(i)) for i in range(10)]
    kept = recent_history(turns)
    assert [t.content for t in kept] == ["4", "5", "6", "7", "8", "9"]


def test_extractive_generator_cites_relevant_sentence() -> None:
    answer = ExtractiveGenerator().answer("What constant does RRF use?", ITEMS)
    assert "constant k equal to 60" in answer
    assert answer.endswith("[1]")
    assert "Bananas" not in answer


def test_extractive_generator_refuses_when_irrelevant() -> None:
    assert ExtractiveGenerator().answer("Who won the football match?", ITEMS) == NO_ANSWER


def test_extractive_stream_reassembles() -> None:
    gen = ExtractiveGenerator()
    text = "Alpha beta gamma [1]."
    assert "".join(gen.stream(text)) == text


@pytest.mark.parametrize(
    ("provider", "url", "label"),
    [
        ("none", "", "Offline"),
        ("anthropic", "", "Anthropic"),
        ("openai", "", "OpenAI"),
        ("openai_compatible", "https://api.groq.com/openai/v1", "Groq"),
        ("openai_compatible", "https://generativelanguage.googleapis.com/v1beta/openai", "Gemini"),
        ("openai_compatible", "http://localhost:11434/v1", "Ollama"),
        ("openai_compatible", "https://example.com/v1", "OpenAI-compatible"),
    ],
)
def test_provider_label(provider: str, url: str, label: str) -> None:
    s = Settings(llm_provider=provider, llm_base_url=url, _env_file=None)  # type: ignore[arg-type,call-arg]
    assert provider_label(s) == label


def test_create_llm_variants() -> None:
    assert create_llm(Settings(llm_provider="none", _env_file=None)) is None  # type: ignore[call-arg]
    groq = create_llm(
        Settings(
            llm_provider="openai_compatible",
            llm_api_key="k",
            llm_model="llama-3.3-70b-versatile",
            llm_base_url="https://api.groq.com/openai/v1",
            _env_file=None,  # type: ignore[call-arg]
        )
    )
    assert groq is not None and groq.name == "Groq"
    claude = create_llm(Settings(llm_provider="anthropic", llm_api_key="k", _env_file=None))  # type: ignore[call-arg]
    assert claude is not None and claude.name == "Anthropic"


def test_keywords_of_strips_question_scaffolding() -> None:
    assert keywords_of("What is the value of k in reciprocal rank fusion?") == (
        "value k reciprocal rank fusion"
    )


def test_condense_without_history_is_identity() -> None:
    assert condense_with_history("What is RRF?", [], None) == "What is RRF?"


def test_condense_heuristic_adds_topic_for_follow_up() -> None:
    history = [Turn("user", "Explain reciprocal rank fusion"), Turn("assistant", "It merges...")]
    out = condense_with_history("What about its constant?", history, None)
    assert "reciprocal rank fusion" in out
    assert out.startswith("What about its constant")


def test_condense_with_llm_and_failure_fallback() -> None:
    history = [Turn("user", "Explain BM25"), Turn("assistant", "A ranking function")]
    assert condense_with_history("and its k1?", history, FakeLLM("What is k1 in BM25?")) == (
        "What is k1 in BM25?"
    )
    out = condense_with_history("and its k1?", history, FakeLLM(fail=True))
    assert "bm25" in out


def test_standalone_question_not_modified_by_heuristic() -> None:
    history = [Turn("user", "Explain BM25"), Turn("assistant", "A ranking function")]
    q = "How does reciprocal rank fusion combine ranked lists"
    assert condense_with_history(q, history, None) == q


def test_rewrite_hyde_multi_query_offline() -> None:
    q = "How does hybrid search work?"
    assert rewrite_query(q, None) == "hybrid search work"
    hyde = hyde_document(q, None)
    assert "hybrid search work" in hyde.lower() and len(hyde.split()) > 15
    variants = multi_query(q, None, n=3)
    assert 1 <= len(variants) <= 3
    assert q not in variants


def test_transforms_use_llm_when_available() -> None:
    assert rewrite_query("q?", FakeLLM("rewritten keywords\nextra")) == "rewritten keywords"
    assert hyde_document("q?", FakeLLM("Hypothetical answer.")) == "Hypothetical answer."
    assert multi_query("q?", FakeLLM("1. first\n2. second\n- third\n4. fourth")) == [
        "first",
        "second",
        "third",
    ]


def test_transforms_fall_back_when_llm_fails() -> None:
    assert rewrite_query("What is BM25?", FakeLLM(fail=True)) == "bm25"
    assert hyde_document("What is BM25?", FakeLLM(fail=True)).lower().startswith("bm25")
