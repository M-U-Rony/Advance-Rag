/**
 * A from-scratch vector store: brute-force ("flat") nearest-neighbour search.
 *
 * A flat index compares the query against *every* stored vector. That is exact and, for the
 * tens of thousands of chunks a PDF corpus produces, fast enough; approximate indexes
 * (HNSW, IVF) only pay off at millions of vectors.
 *
 * Three similarity metrics are supported, all returned as "higher is better":
 *
 * - `cosine` - angle between vectors (length-invariant); the usual choice for text.
 * - `dot`    - inner product; equals cosine for normalised vectors, rewards long vectors otherwise.
 * - `l2`     - negative Euclidean distance (so that closer vectors score higher).
 *
 * Vectors live in one contiguous `Float32Array` (`rows * dim`), which is also exactly the
 * byte layout written to disk.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { ChunkMetadata } from "../types.js";
import { topK } from "./bm25.js";
import type { MetadataFilter } from "./filters.js";

export const METRICS = ["cosine", "dot", "l2"] as const;
export type Metric = (typeof METRICS)[number];

/** Score one stored vector (`matrix[offset .. offset+dim]`) against `query`; higher = more similar. */
function scoreRow(
  matrix: Float32Array,
  offset: number,
  query: Float32Array,
  metric: Metric,
): number {
  const dim = query.length;
  if (metric === "l2") {
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      const d = (matrix[offset + j] as number) - (query[j] as number);
      sum += d * d;
    }
    return -Math.sqrt(sum);
  }
  let dot = 0;
  let rowSq = 0;
  for (let j = 0; j < dim; j++) {
    const a = matrix[offset + j] as number;
    dot += a * (query[j] as number);
    rowSq += a * a;
  }
  if (metric === "dot") return dot;
  let qSq = 0;
  for (let j = 0; j < dim; j++) qSq += (query[j] as number) ** 2;
  const norm = Math.sqrt(rowSq) * (Math.sqrt(qSq) || 1);
  return dot / (norm === 0 ? 1 : norm);
}

/** Flat in-memory index with metadata and `.bin` + `.json` persistence. */
export class VectorStore {
  ids: string[] = [];
  metadata: ChunkMetadata[] = [];
  dim = 0;
  matrix = new Float32Array(0);

  get size(): number {
    return this.ids.length;
  }

  /** The stored vector at row `i` (a view, not a copy). */
  row(i: number): Float32Array {
    return this.matrix.subarray(i * this.dim, (i + 1) * this.dim);
  }

  /** Append vectors (one row per id) with their metadata. */
  add(ids: string[], vectors: Float32Array[], metadata: ChunkMetadata[]): void {
    if (ids.length !== vectors.length || ids.length !== metadata.length) {
      throw new Error("ids, vectors and metadata must have the same length");
    }
    if (ids.length === 0) return;
    const dim = (vectors[0] as Float32Array).length;
    if (this.size > 0 && dim !== this.dim) throw new Error("vector dimension mismatch");
    const grown = new Float32Array(this.matrix.length + ids.length * dim);
    grown.set(this.matrix);
    vectors.forEach((v, i) => grown.set(v, this.matrix.length + i * dim));
    this.matrix = grown;
    this.dim = dim;
    this.ids.push(...ids);
    this.metadata.push(...metadata);
  }

  /** Return the `k` best `[id, score]` pairs among vectors passing the filter. */
  search(
    query: Float32Array,
    k: number,
    metric: Metric = "cosine",
    metadataFilter: MetadataFilter | null = null,
  ): [string, number][] {
    if (this.size === 0 || k <= 0) return [];
    let allowed = Array.from({ length: this.size }, (_, i) => i);
    if (metadataFilter?.active) {
      allowed = allowed.filter((i) => metadataFilter.matches(this.metadata[i] as ChunkMetadata));
      if (allowed.length === 0) return [];
    }
    const scores = allowed.map((i) => scoreRow(this.matrix, i * this.dim, query, metric));
    return topK(scores, k).map(([pos, score]) => [
      this.ids[allowed[pos] as number] as string,
      score,
    ]);
  }

  /** Persist to `directory/{name}.bin` (raw float32) and `directory/{name}.json` (ids, metadata). */
  save(directory: string, name = "vectors"): void {
    mkdirSync(directory, { recursive: true });
    writeFileSync(
      path.join(directory, `${name}.bin`),
      new Uint8Array(this.matrix.buffer, this.matrix.byteOffset, this.matrix.byteLength),
    );
    writeFileSync(
      path.join(directory, `${name}.json`),
      JSON.stringify({ dim: this.dim, ids: this.ids, metadata: this.metadata }),
      "utf-8",
    );
  }

  /** Load a store previously written by {@link VectorStore.save}. */
  static load(directory: string, name = "vectors"): VectorStore {
    const meta = JSON.parse(readFileSync(path.join(directory, `${name}.json`), "utf-8")) as {
      dim: number;
      ids: string[];
      metadata: ChunkMetadata[];
    };
    const bytes = readFileSync(path.join(directory, `${name}.bin`));
    const store = new VectorStore();
    // Copy into a fresh ArrayBuffer: a Buffer's byteOffset need not be 4-byte aligned.
    store.matrix = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    if (store.matrix.length !== meta.ids.length * meta.dim) {
      throw new Error("vector cache is corrupt (size mismatch)");
    }
    store.dim = meta.dim;
    store.ids = meta.ids;
    store.metadata = meta.metadata;
    return store;
  }
}
