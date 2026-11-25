/**
 * Persistent document registry: documents, their chunks and parents, stored as JSON.
 *
 * One JSON file per document in `DATA_DIR/docs`. Embeddings and indexes are derived data and
 * are rebuilt (or loaded from the vector-store cache) by the pipeline at startup.
 */
import { mkdirSync, readdirSync, readFileSync, rmSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { DocumentInfo } from "./schemas.js";
import type { Chunk, ChunkSet } from "./types.js";

/** Registry entry for an indexed PDF (same fields as the public `DocumentInfo`). */
export type DocumentRecord = DocumentInfo;

/** ISO timestamp (UTC, second precision) for `created_at`, e.g. `2026-01-31T12:00:00+00:00`. */
export function nowIso(): string {
  return new Date().toISOString().slice(0, 19) + "+00:00";
}

interface StoredDocument {
  record: DocumentRecord;
  chunks: Chunk[];
  parents: Chunk[];
}

/** Reads and writes document records and their chunks under `root`. */
export class DocumentStore {
  readonly docsDir: string;

  constructor(readonly root: string) {
    this.docsDir = path.join(root, "docs");
    mkdirSync(this.docsDir, { recursive: true });
  }

  /** Persist a document and its chunks. */
  save(record: DocumentRecord, chunkSet: ChunkSet): void {
    const payload: StoredDocument = {
      record,
      chunks: chunkSet.chunks,
      parents: [...chunkSet.parents.values()],
    };
    writeFileSync(path.join(this.docsDir, `${record.id}.json`), JSON.stringify(payload), "utf-8");
  }

  /** Load every stored document, oldest first. */
  loadAll(): [DocumentRecord, ChunkSet][] {
    const files = readdirSync(this.docsDir)
      .filter((f) => f.endsWith(".json"))
      .sort();
    const loaded = files.map((file): [DocumentRecord, ChunkSet] => {
      const data = JSON.parse(
        readFileSync(path.join(this.docsDir, file), "utf-8"),
      ) as StoredDocument;
      return [
        data.record,
        { chunks: data.chunks, parents: new Map(data.parents.map((p) => [p.id, p])) },
      ];
    });
    return loaded.sort((a, b) =>
      a[0].created_at < b[0].created_at ? -1 : a[0].created_at > b[0].created_at ? 1 : 0,
    );
  }

  /** Remove a document; returns whether it existed. */
  delete(docId: string): boolean {
    const file = path.join(this.docsDir, `${docId}.json`);
    if (!existsSync(file)) return false;
    rmSync(file);
    return true;
  }
}
