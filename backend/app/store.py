"""Persistent document registry: documents, their chunks and parents, stored as JSON.

One JSON file per document in ``DATA_DIR/docs``. Embeddings and indexes are derived data and
are rebuilt (or loaded from the vector-store cache) by the pipeline at startup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models import Chunk, ChunkSet


@dataclass(slots=True)
class DocumentRecord:
    """Registry entry for an indexed PDF."""

    id: str
    filename: str
    pages: int
    chunks: int
    strategy: str
    chunk_size: int
    chunk_overlap: int
    created_at: str
    parent_chunks: int = 0

    @staticmethod
    def now() -> str:
        """ISO timestamp for ``created_at``."""
        return datetime.now(UTC).isoformat(timespec="seconds")


class DocumentStore:
    """Reads and writes document records and their chunks under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.docs_dir = root / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: DocumentRecord, chunk_set: ChunkSet) -> None:
        """Persist a document and its chunks."""
        payload = {
            "record": asdict(record),
            "chunks": [asdict(c) for c in chunk_set.chunks],
            "parents": [asdict(c) for c in chunk_set.parents.values()],
        }
        (self.docs_dir / f"{record.id}.json").write_text(json.dumps(payload), encoding="utf-8")

    def load_all(self) -> list[tuple[DocumentRecord, ChunkSet]]:
        """Load every stored document, oldest first."""
        loaded: list[tuple[DocumentRecord, ChunkSet]] = []
        for path in sorted(self.docs_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            record = DocumentRecord(**data["record"])
            chunks = [Chunk(**c) for c in data["chunks"]]
            parents = {c["id"]: Chunk(**c) for c in data["parents"]}
            loaded.append((record, ChunkSet(chunks=chunks, parents=parents)))
        loaded.sort(key=lambda pair: pair[0].created_at)
        return loaded

    def delete(self, doc_id: str) -> bool:
        """Remove a document; returns whether it existed."""
        path = self.docs_dir / f"{doc_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
