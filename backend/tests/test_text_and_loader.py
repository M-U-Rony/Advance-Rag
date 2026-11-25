from __future__ import annotations

import pytest

from app.ingestion.pdf_loader import clean_page_text, is_heading, load_pdf
from app.text import content_terms, estimate_tokens, snippet, split_sentences, tokenize
from tests.conftest import make_pdf


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World! It's k-means 2024.") == [
        "hello",
        "world",
        "it's",
        "k-means",
        "2024",
    ]


def test_content_terms_removes_stopwords() -> None:
    assert content_terms("What is the capital of France") == ["capital", "france"]


def test_split_sentences() -> None:
    assert split_sentences("One is here. Two is there! Is three? Yes.") == [
        "One is here.",
        "Two is there!",
        "Is three?",
        "Yes.",
    ]
    assert split_sentences("   ") == []


def test_snippet_truncates() -> None:
    assert snippet("a " * 300, 20).endswith("…")
    assert snippet("short") == "short"


def test_estimate_tokens_grows_with_words() -> None:
    assert estimate_tokens("one two three four") > estimate_tokens("one two")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Reciprocal Rank Fusion", True),
        ("1.2 Chunking strategies", True),
        ("INTRODUCTION", True),
        ("This is a normal sentence.", False),
        ("a short line without a period but lowercase", False),
        ("Heading ending with colon:", False),
        ("", False),
    ],
)
def test_is_heading(line: str, expected: bool) -> None:
    assert is_heading(line) is expected


def test_clean_page_text_restores_paragraphs_and_dehyphenates() -> None:
    raw = (
        "Main Heading\n"
        "This paragraph is long enough to fill an entire line of the page easily and\n"
        "continues with a hyphen-\nated word and ends here.\n"
        "Short end.\n"
        "Next paragraph starts here."
    )
    cleaned = clean_page_text(raw)
    paragraphs = cleaned.split("\n\n")
    assert paragraphs[0] == "Main Heading"
    assert "hyphenated" in paragraphs[1]
    assert paragraphs[-1] == "Next paragraph starts here."


def test_clean_page_text_empty() -> None:
    assert clean_page_text("  \n \n") == ""


def test_load_pdf_returns_numbered_pages(pdf_bytes: bytes) -> None:
    pages = load_pdf(pdf_bytes)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "Photosynthesis" in pages[0].text
    assert "Moon" in pages[2].text


def test_load_pdf_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Could not read PDF"):
        load_pdf(b"not a pdf")


def test_load_pdf_rejects_empty_text() -> None:
    blank = make_pdf([("", "")])
    with pytest.raises(ValueError, match="no extractable text"):
        load_pdf(blank)
