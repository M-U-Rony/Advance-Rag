/** Core domain models shared across ingestion, retrieval and the API. */

/** One page of extracted PDF text (`number` is 1-based). */
export interface Page {
  number: number;
  text: string;
}

/**
 * A retrievable passage together with the metadata needed for citations.
 *
 * `parent_id` is set for *child* chunks produced by parent-child chunking; it points to the
 * larger parent passage that is handed to the LLM instead of the child. Property names are
 * snake_case because chunks are persisted to disk and sent over the API unchanged.
 */
export interface Chunk {
  id: string;
  doc_id: string;
  filename: string;
  page: number;
  position: number;
  text: string;
  heading: string | null;
  parent_id: string | null;
}

/** Metadata used for pre-filtering. */
export interface ChunkMetadata {
  doc_id: string;
  page: number;
  filename: string;
}

/** The pre-filtering metadata of a chunk. */
export function chunkMetadata(chunk: Chunk): ChunkMetadata {
  return { doc_id: chunk.doc_id, page: chunk.page, filename: chunk.filename };
}

/** Result of chunking a document: searchable `chunks` plus their `parents` (if any). */
export interface ChunkSet {
  chunks: Chunk[];
  parents: Map<string, Chunk>;
}
