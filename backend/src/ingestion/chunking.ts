/**
 * Chunking strategies: how documents are cut into retrievable passages.
 *
 * Chunk size is the central trade-off of RAG indexing. Small chunks give precise
 * embeddings but lose context; large chunks keep context but dilute the embedding.
 * Four strategies are implemented, all operating page by page so page metadata stays exact:
 *
 * - `fixed`        - sliding window of N words with M words of overlap.
 * - `recursive`    - split on the largest natural boundary first (sections/paragraphs),
 *                    then sentences, then words, and merge small pieces back up to N words.
 * - `semantic`     - heading aware: each section stays together and carries its heading.
 * - `parent_child` - index small *child* chunks (precise matching) but remember the larger
 *                    *parent* chunk (rich context) to return to the LLM.
 */
import { InvalidInputError } from "../errors.js";
import type { Chunk, ChunkSet, Page } from "../types.js";
import { splitSentences, wordCount, words } from "../text.js";
import { isHeading } from "./pdfLoader.js";

export const STRATEGIES = ["fixed", "recursive", "semantic", "parent_child"] as const;
export type Strategy = (typeof STRATEGIES)[number];

/** Tunable chunking parameters (sizes are in words). */
export interface ChunkingParams {
  strategy: Strategy;
  chunk_size: number;
  chunk_overlap: number;
  parent_size: number;
  child_size: number;
}

/** The defaults of the upload form: recursive chunks of 120 words with 20 words of overlap. */
export const DEFAULT_CHUNKING: ChunkingParams = {
  strategy: "recursive",
  chunk_size: 120,
  chunk_overlap: 20,
  parent_size: 200,
  child_size: 50,
};

/** Fill in defaults for any parameter not given. */
export function chunkingParams(partial: Partial<ChunkingParams> = {}): ChunkingParams {
  return { ...DEFAULT_CHUNKING, ...partial };
}

/** @throws InvalidInputError for impossible parameter combinations. */
export function validateChunkingParams(params: ChunkingParams): void {
  if (!STRATEGIES.includes(params.strategy)) {
    throw new InvalidInputError(`Unknown chunking strategy: ${params.strategy}`);
  }
  if (params.chunk_size < 10 || params.child_size < 5) {
    throw new InvalidInputError("chunk sizes are too small");
  }
  if (!(params.chunk_overlap >= 0 && params.chunk_overlap < params.chunk_size)) {
    throw new InvalidInputError("chunk_overlap must be >= 0 and smaller than chunk_size");
  }
  if (params.parent_size <= params.child_size) {
    throw new InvalidInputError("parent_size must be larger than child_size");
  }
}

/** Sliding word window: `size` words per chunk, advancing by `size - overlap`. */
export function fixedSizeChunks(text: string, size: number, overlap: number): string[] {
  const tokens = words(text);
  if (tokens.length === 0) return [];
  const step = Math.max(size - overlap, 1);
  const chunks: string[] = [];
  for (let start = 0; start < tokens.length; start += step) {
    chunks.push(tokens.slice(start, start + size).join(" "));
    if (start + size >= tokens.length) break;
  }
  return chunks;
}

function splitSentenceOrWords(text: string, size: number): string[] {
  const sentences = splitSentences(text);
  const pieces: string[] = [];
  for (const sentence of sentences.length > 0 ? sentences : [text]) {
    if (wordCount(sentence) <= size) pieces.push(sentence);
    else pieces.push(...fixedSizeChunks(sentence, size, 0));
  }
  return pieces;
}

function merge(pieces: string[], size: number): string[] {
  const merged: string[] = [];
  let current: string[] = [];
  let count = 0;
  for (const piece of pieces) {
    const n = wordCount(piece);
    if (current.length > 0 && count + n > size) {
      merged.push(current.join(" "));
      current = [];
      count = 0;
    }
    current.push(piece);
    count += n;
  }
  if (current.length > 0) merged.push(current.join(" "));
  return merged;
}

/** Split by paragraphs, falling back to sentences then words, merging up to `size` words. */
export function recursiveChunks(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (const raw of text.split(/\n\s*\n/)) {
    const paragraph = raw.trim();
    if (!paragraph) continue;
    if (wordCount(paragraph) <= size) pieces.push(paragraph);
    else pieces.push(...splitSentenceOrWords(paragraph, size));
  }
  return merge(pieces, size);
}

/** A `[heading, body]` pair; `heading` is null for text before the first heading. */
export type Section = [heading: string | null, body: string];

/** Split page text into `[heading, body]` sections using heading detection. */
export function splitSections(text: string): Section[] {
  const sections: Section[] = [];
  let heading: string | null = null;
  let body: string[] = [];
  for (const raw of text.split(/\n\s*\n/)) {
    const paragraph = raw.trim();
    if (!paragraph) continue;
    if (!paragraph.includes("\n") && isHeading(paragraph)) {
      if (body.length > 0) {
        sections.push([heading, body.join("\n\n")]);
        body = [];
      }
      heading = paragraph;
    } else {
      body.push(paragraph);
    }
  }
  if (body.length > 0) sections.push([heading, body.join("\n\n")]);
  return sections;
}

/** Heading-aware chunks: `[heading, text]`; long sections are split recursively. */
export function semanticChunks(text: string, size: number): Section[] {
  const result: Section[] = [];
  for (const [heading, body] of splitSections(text)) {
    for (const part of recursiveChunks(body, size)) {
      result.push([heading, heading ? `${heading}. ${part}` : part]);
    }
  }
  return result;
}

function pageBlocks(page: Page, params: ChunkingParams): Section[] {
  if (params.strategy === "fixed") {
    const flat = words(page.text).join(" ");
    return fixedSizeChunks(flat, params.chunk_size, params.chunk_overlap).map((c) => [null, c]);
  }
  if (params.strategy === "semantic") return semanticChunks(page.text, params.chunk_size);
  return recursiveChunks(page.text, params.chunk_size).map((c) => [null, c]);
}

/**
 * Chunk every page of a document according to `params`.
 *
 * Chunk ids are `{docId}-c{position}` (children) and `{docId}-p{position}` (parents).
 */
export function chunkDocument(
  pages: Page[],
  docId: string,
  filename: string,
  params: ChunkingParams,
): ChunkSet {
  validateChunkingParams(params);
  const chunks: Chunk[] = [];
  const parents = new Map<string, Chunk>();

  if (params.strategy === "parent_child") {
    const overlap = Math.max(Math.floor(params.child_size / 5), 0);
    for (const page of pages) {
      for (const parentText of recursiveChunks(page.text, params.parent_size)) {
        const parentId = `${docId}-p${parents.size}`;
        parents.set(parentId, {
          id: parentId,
          doc_id: docId,
          filename,
          page: page.number,
          position: parents.size,
          text: parentText,
          heading: null,
          parent_id: null,
        });
        for (const childText of fixedSizeChunks(parentText, params.child_size, overlap)) {
          chunks.push({
            id: `${docId}-c${chunks.length}`,
            doc_id: docId,
            filename,
            page: page.number,
            position: chunks.length,
            text: childText,
            heading: null,
            parent_id: parentId,
          });
        }
      }
    }
    return { chunks, parents };
  }

  for (const page of pages) {
    for (const [heading, text] of pageBlocks(page, params)) {
      chunks.push({
        id: `${docId}-c${chunks.length}`,
        doc_id: docId,
        filename,
        page: page.number,
        position: chunks.length,
        text,
        heading,
        parent_id: null,
      });
    }
  }
  return { chunks, parents };
}
