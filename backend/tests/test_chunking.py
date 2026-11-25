from __future__ import annotations

import pytest

from app.ingestion.chunking import (
    ChunkingParams,
    chunk_document,
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
    split_sections,
)
from app.models import Page
from app.text import word_count


def test_fixed_size_window_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(10))
    chunks = fixed_size_chunks(text, size=4, overlap=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_fixed_size_short_text_and_empty() -> None:
    assert fixed_size_chunks("a b c", 10, 2) == ["a b c"]
    assert fixed_size_chunks("", 10, 2) == []


def test_recursive_respects_size_and_paragraphs() -> None:
    para1 = "Alpha one. Alpha two. Alpha three."
    para2 = "Beta one. Beta two."
    chunks = recursive_chunks(f"{para1}\n\n{para2}", size=6)
    assert all(word_count(c) <= 6 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == f"{para1} {para2}"


def test_recursive_splits_overlong_sentence_by_words() -> None:
    long_sentence = " ".join(["word"] * 25) + "."
    chunks = recursive_chunks(long_sentence, size=10)
    assert len(chunks) == 3
    assert all(word_count(c) <= 10 for c in chunks)


def test_split_sections_detects_headings() -> None:
    text = "Intro Section\n\nBody of intro here.\n\nSecond Section\n\nBody two here."
    assert split_sections(text) == [
        ("Intro Section", "Body of intro here."),
        ("Second Section", "Body two here."),
    ]


def test_semantic_chunks_prefix_heading() -> None:
    text = "Intro Section\n\nBody of intro here."
    assert semantic_chunks(text, 50) == [("Intro Section", "Intro Section. Body of intro here.")]


def _pages() -> list[Page]:
    body = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(30))
    return [Page(1, "Title Here\n\n" + body), Page(2, "Another Heading\n\nShort page text here.")]


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_chunk_document_metadata(strategy: str) -> None:
    result = chunk_document(
        _pages(),
        doc_id="d1",
        filename="f.pdf",
        params=ChunkingParams(strategy=strategy, chunk_size=40, chunk_overlap=5),  # type: ignore[arg-type]
    )
    assert result.parents == {}
    assert len(result.chunks) > 2
    assert [c.position for c in result.chunks] == list(range(len(result.chunks)))
    assert {c.page for c in result.chunks} == {1, 2}
    assert all(c.doc_id == "d1" and c.filename == "f.pdf" for c in result.chunks)
    assert len({c.id for c in result.chunks}) == len(result.chunks)


def test_parent_child_mapping() -> None:
    result = chunk_document(
        _pages(),
        doc_id="d1",
        filename="f.pdf",
        params=ChunkingParams(strategy="parent_child", parent_size=60, child_size=15),
    )
    assert result.parents
    for child in result.chunks:
        assert child.parent_id in result.parents
        parent = result.parents[child.parent_id or ""]
        assert parent.page == child.page
        assert word_count(child.text) <= 15
        assert child.text.split()[0] in parent.text
    assert max(word_count(p.text) for p in result.parents.values()) <= 60
    assert len(result.chunks) > len(result.parents)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "nope"},
        {"chunk_size": 3},
        {"chunk_overlap": 500},
        {"parent_size": 10, "child_size": 20},
    ],
)
def test_invalid_params(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ChunkingParams(**kwargs).validate()  # type: ignore[arg-type]
