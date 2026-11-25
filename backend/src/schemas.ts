/**
 * Zod schemas for the HTTP API and the pipeline trace.
 *
 * These double as runtime validators (a bad request becomes a 422) and as the source of the
 * TypeScript types (`z.infer`), so the contract is written down exactly once.
 */
import { z } from "zod";

export const RETRIEVAL_MODES = ["vector", "bm25", "hybrid"] as const;
export const FUSION_METHODS = ["rrf", "weighted"] as const;
export const RERANK_METHODS = ["none", "heuristic", "cross_encoder"] as const;
export const QUERY_TRANSFORMS = ["none", "rewrite", "hyde", "multi_query"] as const;
export const SIMILARITY_METRICS = ["cosine", "dot", "l2"] as const;

/** Every knob of the retrieval pipeline (all optional, with sensible defaults). */
export const retrievalSettingsSchema = z.object({
  retrieval_mode: z.enum(RETRIEVAL_MODES).default("hybrid"),
  fusion: z.enum(FUSION_METHODS).default("rrf"),
  /** 0 = keyword only, 1 = vector only */
  alpha: z.number().min(0).max(1).default(0.5),
  top_k: z.number().int().min(1).max(20).default(5),
  /** Candidates fetched before reranking */
  candidates: z.number().int().min(1).max(100).default(30),
  rerank: z.enum(RERANK_METHODS).default("heuristic"),
  query_transform: z.enum(QUERY_TRANSFORMS).default("none"),
  parent_child: z.boolean().default(false),
  compression: z.boolean().default(false),
  similarity: z.enum(SIMILARITY_METRICS).default("cosine"),
  doc_ids: z.array(z.string()).nullable().default(null),
  page_min: z.number().int().min(1).nullable().default(null),
  page_max: z.number().int().min(1).nullable().default(null),
});
export type RetrievalSettings = z.infer<typeof retrievalSettingsSchema>;

/** One chat message of the conversation history. */
export const historyTurnSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string(),
});
export type HistoryTurn = z.infer<typeof historyTurnSchema>;

/** Body of `POST /api/query` and `POST /api/query/stream`. */
export const queryRequestSchema = z.object({
  question: z.string().min(1).max(2000),
  history: z.array(historyTurnSchema).default([]),
  settings: retrievalSettingsSchema.default(() => retrievalSettingsSchema.parse({})),
});
export type QueryRequest = z.infer<typeof queryRequestSchema>;

/** A retrieved passage at some stage of the pipeline. */
export interface Hit {
  chunk_id: string;
  filename: string;
  page: number;
  rank: number;
  score: number;
  snippet: string;
  is_parent: boolean;
}

/** Trace of one pipeline stage. */
export interface Stage {
  key: string;
  label: string;
  ms: number;
  skipped: boolean;
  info: Record<string, unknown>;
  hits: Hit[];
}

/** Full, inspectable record of how an answer was produced. */
export interface Trace {
  question: string;
  condensed_query: string;
  rewritten_query: string | null;
  hyde_text: string | null;
  multi_queries: string[];
  stages: Stage[];
  ranked_chunk_ids: string[];
  prompt_tokens: number;
  total_ms: number;
  embedder: string;
  llm: string;
  reranker: string | null;
  warnings: string[];
  settings: RetrievalSettings;
}

/** A source passage the answer may cite as `[index]`. */
export interface Citation {
  index: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number;
  score: number;
  text: string;
}

/** Answer with citations and the pipeline trace. */
export interface QueryResponse {
  answer: string;
  citations: Citation[];
  trace: Trace;
}

/** Public view of an indexed document. */
export interface DocumentInfo {
  id: string;
  filename: string;
  pages: number;
  chunks: number;
  parent_chunks: number;
  strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
}

/** Retrieval and generation metrics for one pipeline configuration. */
export interface EvalRow {
  config: string;
  recall: Record<string, number>;
  precision: number;
  mrr: number;
  faithfulness: number;
  answer_relevance: number;
  avg_ms: number;
}

/** Result of `POST /api/evaluate`. */
export interface EvalReport {
  embedder: string;
  questions: number;
  ks: number[];
  precision_k: number;
  source: string;
  rows: EvalRow[];
}

/** Body of `POST /api/evaluate`. */
export const evalRequestSchema = z.object({
  ks: z.array(z.number().int().min(1).max(20)).min(1).default([1, 3, 5]),
  max_questions: z.number().int().min(1).max(200).default(30),
  doc_ids: z.array(z.string()).nullable().default(null),
});
export type EvalRequest = z.infer<typeof evalRequestSchema>;
