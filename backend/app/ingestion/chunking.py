"""Chunking strategies: how documents are cut into retrievable passages.

Chunk size is the central trade-off of RAG indexing. Small chunks give precise
embeddings but lose context; large chunks keep context but dilute the embedding.
Four strategies are implemented, all operating page by page so page metadata stays exact:

* ``fixed``        - sliding window of N words with M words of overlap.
* ``recursive``    - split on the largest natural boundary first (sections/paragraphs),
                     then sentences, then words, and merge small pieces back up to N words.
* ``semantic``     - heading aware: each section stays together and carries its heading.
* ``parent_child`` - index small *child* chunks (precise matching) but remember the larger
                     *parent* chunk (rich context) to return to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.ingestion.pdf_loader import is_heading
from app.models import Chunk, ChunkSet, Page
from app.text import split_sentences, word_count

Strategy = Literal["fixed", "recursive", "semantic", "parent_child"]
STRATEGIES: tuple[str, ...] = ("fixed", "recursive", "semantic", "parent_child")


@dataclass(frozen=True, slots=True)
class ChunkingParams:
    """Tunable chunking parameters (sizes are in words)."""

    strategy: Strategy = "recursive"
    chunk_size: int = 120
    chunk_overlap: int = 20
    parent_size: int = 200
    child_size: int = 50

    def validate(self) -> None:
        """Raise ``ValueError`` for impossible parameter combinations."""
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
        if self.chunk_size < 10 or self.child_size < 5:
            raise ValueError("chunk sizes are too small")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.parent_size <= self.child_size:
            raise ValueError("parent_size must be larger than child_size")


def fixed_size_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Sliding word window: ``size`` words per chunk, advancing by ``size - overlap``."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def _split_sentence_or_words(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    for sentence in split_sentences(text) or [text]:
        if word_count(sentence) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(fixed_size_chunks(sentence, size, 0))
    return pieces


def recursive_chunks(text: str, size: int) -> list[str]:
    """Split by paragraphs, falling back to sentences then words, merging up to ``size`` words."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if word_count(paragraph) <= size:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_sentence_or_words(paragraph, size))
    return _merge(pieces, size)


def _merge(pieces: list[str], size: int) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = word_count(piece)
        if current and count + n > size:
            merged.append(" ".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        merged.append(" ".join(current))
    return merged


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split page text into ``(heading, body)`` sections using heading detection."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    body: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if "\n" not in paragraph and is_heading(paragraph):
            if body:
                sections.append((heading, "\n\n".join(body)))
                body = []
            heading = paragraph
        else:
            body.append(paragraph)
    if body:
        sections.append((heading, "\n\n".join(body)))
    return sections


def semantic_chunks(text: str, size: int) -> list[tuple[str | None, str]]:
    """Heading-aware chunks: ``(heading, text)``; long sections are split recursively."""
    result: list[tuple[str | None, str]] = []
    for heading, body in split_sections(text):
        for part in recursive_chunks(body, size):
            result.append((heading, f"{heading}. {part}" if heading else part))
    return result


def _page_blocks(page: Page, params: ChunkingParams) -> list[tuple[str | None, str]]:
    if params.strategy == "fixed":
        flat = " ".join(page.text.split())
        return [(None, c) for c in fixed_size_chunks(flat, params.chunk_size, params.chunk_overlap)]
    if params.strategy == "semantic":
        return semantic_chunks(page.text, params.chunk_size)
    return [(None, c) for c in recursive_chunks(page.text, params.chunk_size)]


def chunk_document(
    pages: list[Page],
    *,
    doc_id: str,
    filename: str,
    params: ChunkingParams,
) -> ChunkSet:
    """Chunk every page of a document according to ``params``.

    Chunk ids are ``{doc_id}-c{position}`` (children) and ``{doc_id}-p{position}`` (parents).
    """
    params.validate()
    chunks: list[Chunk] = []
    parents: dict[str, Chunk] = {}

    if params.strategy == "parent_child":
        for page in pages:
            for parent_text in recursive_chunks(page.text, params.parent_size):
                parent_id = f"{doc_id}-p{len(parents)}"
                parents[parent_id] = Chunk(
                    id=parent_id,
                    doc_id=doc_id,
                    filename=filename,
                    page=page.number,
                    position=len(parents),
                    text=parent_text,
                )
                overlap = max(params.child_size // 5, 0)
                for child_text in fixed_size_chunks(parent_text, params.child_size, overlap):
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}-c{len(chunks)}",
                            doc_id=doc_id,
                            filename=filename,
                            page=page.number,
                            position=len(chunks),
                            text=child_text,
                            parent_id=parent_id,
                        )
                    )
        return ChunkSet(chunks=chunks, parents=parents)

    for page in pages:
        for heading, text in _page_blocks(page, params):
            chunks.append(
                Chunk(
                    id=f"{doc_id}-c{len(chunks)}",
                    doc_id=doc_id,
                    filename=filename,
                    page=page.number,
                    position=len(chunks),
                    text=text,
                    heading=heading,
                )
            )
    return ChunkSet(chunks=chunks, parents=parents)
