export type RetrievalMode = "vector" | "bm25" | "hybrid";
export type Fusion = "rrf" | "weighted";
export type Rerank = "none" | "heuristic" | "cross_encoder";
export type QueryTransform = "none" | "rewrite" | "hyde" | "multi_query";
export type Similarity = "cosine" | "dot" | "l2";
export type Strategy = "fixed" | "recursive" | "semantic" | "parent_child";

export interface RetrievalSettings {
  retrieval_mode: RetrievalMode;
  fusion: Fusion;
  alpha: number;
  top_k: number;
  candidates: number;
  rerank: Rerank;
  query_transform: QueryTransform;
  parent_child: boolean;
  compression: boolean;
  similarity: Similarity;
  doc_ids: string[] | null;
  page_min: number | null;
  page_max: number | null;
}

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

export interface Hit {
  chunk_id: string;
  filename: string;
  page: number;
  rank: number;
  score: number;
  snippet: string;
  is_parent: boolean;
}

export interface Stage {
  key: string;
  label: string;
  ms: number;
  skipped: boolean;
  info: Record<string, unknown>;
  hits: Hit[];
}

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

export interface Citation {
  index: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number;
  score: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  trace: Trace;
}

export interface AppConfig {
  version: string;
  llm: { provider: string; label: string; model: string | null; offline: boolean };
  embeddings: {
    requested: string;
    active: string;
    dense: boolean;
    fallback_reason: string | null;
  };
  reranker: { model: string; cross_encoder_available: boolean | null };
  sample_available: boolean;
}

export interface EvalRow {
  config: string;
  recall: Record<string, number>;
  precision: number;
  mrr: number;
  faithfulness: number;
  answer_relevance: number;
  avg_ms: number;
}

export interface EvalReport {
  embedder: string;
  questions: number;
  ks: number[];
  precision_k: number;
  source: string;
  rows: EvalRow[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  trace?: Trace;
  streaming?: boolean;
  error?: boolean;
}

export interface ChunkingOptions {
  strategy: Strategy;
  chunk_size: number;
  chunk_overlap: number;
  parent_size: number;
  child_size: number;
}
