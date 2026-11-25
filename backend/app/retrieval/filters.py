"""Metadata pre-filtering.

Applying a filter *before* the similarity search (rather than discarding hits afterwards)
guarantees that ``top_k`` results come from the allowed scope, e.g. a single document or
a page range, and it shrinks the search space.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Restrict retrieval to some documents and/or an inclusive page range."""

    doc_ids: frozenset[str] | None = None
    page_min: int | None = None
    page_max: int | None = None

    @property
    def active(self) -> bool:
        """True if the filter excludes anything at all."""
        return bool(self.doc_ids) or self.page_min is not None or self.page_max is not None

    def matches(self, meta: dict[str, object]) -> bool:
        """Whether a chunk's metadata passes the filter."""
        if self.doc_ids and meta.get("doc_id") not in self.doc_ids:
            return False
        page = meta.get("page")
        if not isinstance(page, int):
            return True
        if self.page_min is not None and page < self.page_min:
            return False
        return not (self.page_max is not None and page > self.page_max)
