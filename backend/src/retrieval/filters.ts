/**
 * Metadata pre-filtering.
 *
 * Applying a filter *before* the similarity search (rather than discarding hits afterwards)
 * guarantees that `top_k` results come from the allowed scope, e.g. a single document or
 * a page range, and it shrinks the search space.
 */
import type { ChunkMetadata } from "../types.js";

/** Restrict retrieval to some documents and/or an inclusive page range. */
export class MetadataFilter {
  constructor(
    readonly docIds: ReadonlySet<string> | null = null,
    readonly pageMin: number | null = null,
    readonly pageMax: number | null = null,
  ) {}

  /** True if the filter excludes anything at all. */
  get active(): boolean {
    return (
      (this.docIds !== null && this.docIds.size > 0) ||
      this.pageMin !== null ||
      this.pageMax !== null
    );
  }

  /** Whether a chunk's metadata passes the filter. */
  matches(meta: ChunkMetadata): boolean {
    if (this.docIds && this.docIds.size > 0 && !this.docIds.has(meta.doc_id)) return false;
    if (this.pageMin !== null && meta.page < this.pageMin) return false;
    return !(this.pageMax !== null && meta.page > this.pageMax);
  }
}
