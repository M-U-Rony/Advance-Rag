"""A from-scratch NumPy vector store: brute-force ("flat") nearest-neighbour search.

A flat index compares the query against *every* stored vector. That is exact and, for the
tens of thousands of chunks a PDF corpus produces, fast enough; approximate indexes
(HNSW, IVF) only pay off at millions of vectors.

Three similarity metrics are supported, all returned as "higher is better":

* ``cosine`` - angle between vectors (length-invariant); the usual choice for text.
* ``dot``    - inner product; equals cosine for normalised vectors, rewards long vectors otherwise.
* ``l2``     - negative Euclidean distance (so that closer vectors score higher).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from app.retrieval.filters import MetadataFilter

Metric = Literal["cosine", "dot", "l2"]
METRICS: tuple[str, ...] = ("cosine", "dot", "l2")


def similarity(matrix: np.ndarray, query: np.ndarray, metric: Metric) -> np.ndarray:
    """Score every row of ``matrix`` against ``query`` (higher = more similar)."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if metric == "dot":
        return matrix @ query
    if metric == "l2":
        return -np.linalg.norm(matrix - query, axis=1)
    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        norms[norms == 0] = 1.0
        return (matrix @ query) / norms
    raise ValueError(f"Unknown metric: {metric}")


class VectorStore:
    """Flat in-memory index with metadata and ``.npz`` + ``.json`` persistence."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metadata: list[dict[str, object]] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, object]]) -> None:
        """Append vectors (one row per id) with their metadata."""
        if not (len(ids) == vectors.shape[0] == len(metadata)):
            raise ValueError("ids, vectors and metadata must have the same length")
        if not ids:
            return
        vectors = vectors.astype(np.float32, copy=False)
        self.matrix = vectors.copy() if len(self) == 0 else np.vstack([self.matrix, vectors])
        self.ids.extend(ids)
        self.metadata.extend(metadata)

    def search(
        self,
        query: np.ndarray,
        k: int,
        metric: Metric = "cosine",
        metadata_filter: MetadataFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Return the ``k`` best ``(id, score)`` pairs among vectors passing the filter."""
        if len(self) == 0 or k <= 0:
            return []
        allowed = np.arange(len(self))
        if metadata_filter is not None and metadata_filter.active:
            allowed = np.array(
                [i for i, m in enumerate(self.metadata) if metadata_filter.matches(m)], dtype=int
            )
            if allowed.size == 0:
                return []
        scores = similarity(self.matrix[allowed], query.astype(np.float32), metric)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[allowed[i]], float(scores[i])) for i in top]

    def save(self, directory: Path, name: str = "vectors") -> None:
        """Persist to ``directory/{name}.npz`` and ``directory/{name}.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{name}.npz", matrix=self.matrix)
        (directory / f"{name}.json").write_text(
            json.dumps({"ids": self.ids, "metadata": self.metadata}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, name: str = "vectors") -> VectorStore:
        """Load a store previously written by :meth:`save`."""
        store = cls()
        meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        with np.load(directory / f"{name}.npz") as data:
            store.matrix = data["matrix"].astype(np.float32)
        store.ids = list(meta["ids"])
        store.metadata = list(meta["metadata"])
        return store
