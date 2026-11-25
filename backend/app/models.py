"""Core domain models shared across ingestion, retrieval and the API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Page:
    """One page of extracted PDF text (``number`` is 1-based)."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage together with the metadata needed for citations.

    ``parent_id`` is set for *child* chunks produced by parent-child chunking; it points
    to the larger parent passage that is handed to the LLM instead of the child.
    """

    id: str
    doc_id: str
    filename: str
    page: int
    position: int
    text: str
    heading: str | None = None
    parent_id: str | None = None

    def metadata(self) -> dict[str, object]:
        """Metadata used for pre-filtering."""
        return {"doc_id": self.doc_id, "page": self.page, "filename": self.filename}


@dataclass(slots=True)
class ChunkSet:
    """Result of chunking a document: searchable ``chunks`` plus their ``parents`` (if any)."""

    chunks: list[Chunk]
    parents: dict[str, Chunk]
