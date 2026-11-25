import type { ChunkingOptions, RetrievalSettings } from "./types";

export const DEFAULT_SETTINGS: RetrievalSettings = {
  retrieval_mode: "hybrid",
  fusion: "rrf",
  alpha: 0.5,
  top_k: 5,
  candidates: 30,
  rerank: "heuristic",
  query_transform: "none",
  parent_child: false,
  compression: false,
  similarity: "cosine",
  doc_ids: null,
  page_min: null,
  page_max: null,
};

export const DEFAULT_CHUNKING: ChunkingOptions = {
  strategy: "recursive",
  chunk_size: 120,
  chunk_overlap: 20,
  parent_size: 200,
  child_size: 50,
};

export const STARTER_QUESTIONS = [
  "How does Reciprocal Rank Fusion combine two ranked lists?",
  "Why use a cross-encoder to rerank candidates?",
  "What is HyDE and why does it help retrieval?",
  "What problem does parent-child chunking solve?",
];
