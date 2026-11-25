/**
 * Evaluation harness: compare retrieval configurations on a question set.
 *
 * Four configurations are compared, mirroring the progression taught in advanced-RAG courses:
 *
 * 1. `vector`           - dense (or TF-IDF) similarity search only
 * 2. `bm25`             - keyword search only
 * 3. `hybrid`           - vector + BM25 fused with RRF
 * 4. `hybrid + rerank`  - hybrid candidates reranked by the cross-encoder (heuristic fallback)
 *
 * A question's *relevant chunks* are those whose text contains the question's `evidence`
 * phrase, which keeps the gold labels independent of the chunking strategy.
 */
import { readFileSync } from "node:fs";
import type { RAGPipeline } from "../pipeline.js";
import { keywordsOf } from "../retrieval/queryTransform.js";
import { retrievalSettingsSchema, type EvalReport, type EvalRow } from "../schemas.js";
import { contentTerms, splitSentences, words } from "../text.js";
import type { Chunk } from "../types.js";
import * as metrics from "./metrics.js";

/** A question and the phrase that must appear in a relevant chunk. */
export interface QAItem {
  question: string;
  evidence: string;
}

/** `[name, overrides applied on top of the default retrieval settings]` */
export const CONFIGS: readonly (readonly [string, Record<string, unknown>])[] = [
  ["vector", { retrieval_mode: "vector", rerank: "none" }],
  ["bm25", { retrieval_mode: "bm25", rerank: "none" }],
  ["hybrid", { retrieval_mode: "hybrid", fusion: "rrf", rerank: "none" }],
  ["hybrid + rerank", { retrieval_mode: "hybrid", fusion: "rrf", rerank: "cross_encoder" }],
];

/** Load `[{"question": ..., "evidence": ...}, ...]` from a JSON file. */
export function loadQa(path: string): QAItem[] {
  return JSON.parse(readFileSync(path, "utf-8")) as QAItem[];
}

/** Small deterministic PRNG (mulberry32) so auto-generated questions are reproducible. */
function seededRandom(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Create keyword-style questions from chunk sentences (used when no QA file exists).
 *
 * For a random sample of chunks, the most informative sentence is turned into
 * `"What is said about <keywords>?"`. Questions are lexically close to their evidence, so this
 * favours keyword search; a hand-written QA file gives a fairer comparison.
 */
export function autoGenerateQa(chunks: Chunk[], n: number, seed = 7): QAItem[] {
  const random = seededRandom(seed);
  const pool = chunks.filter((c) => words(c.text).length >= 20);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j] as Chunk, pool[i] as Chunk];
  }
  const items: QAItem[] = [];
  for (const chunk of pool) {
    const sentences = splitSentences(chunk.text).filter((s) => contentTerms(s).length >= 4);
    if (sentences.length === 0) continue;
    const sentence = sentences.reduce((best, s) =>
      new Set(contentTerms(s)).size > new Set(contentTerms(best)).size ? s : best,
    );
    const picked = keywordsOf(sentence).split(" ").filter(Boolean).slice(0, 5);
    if (picked.length >= 3) {
      items.push({
        question: `What is said about ${picked.join(" ")}?`,
        evidence: sentence.slice(0, 60),
      });
    }
    if (items.length >= n) break;
  }
  return items;
}

/** Ids of chunks whose text contains `evidence` (case/whitespace-insensitive). */
export function relevantIds(chunks: Chunk[], evidence: string): Set<string> {
  const normalise = (t: string) => words(t.toLowerCase()).join(" ");
  const needle = normalise(evidence);
  return new Set(chunks.filter((c) => normalise(c.text).includes(needle)).map((c) => c.id));
}

/** Run every configuration over `qa` and aggregate retrieval + generation metrics. */
export async function evaluate(
  pipeline: RAGPipeline,
  qa: QAItem[],
  options: { ks: number[]; docIds?: string[] | null; source?: string },
): Promise<EvalReport> {
  const { ks, docIds = null, source = "provided" } = options;
  const chunks = pipeline.index.chunks.filter((c) => !docIds?.length || docIds.includes(c.doc_id));
  const labelled = qa
    .map((item) => ({ item, rel: relevantIds(chunks, item.evidence) }))
    .filter(({ rel }) => rel.size > 0);
  const topK = Math.max(...ks);
  const rows: EvalRow[] = [];
  for (const [name, overrides] of CONFIGS) {
    const settings = retrievalSettingsSchema.parse({ top_k: topK, doc_ids: docIds, ...overrides });
    const first = labelled[0];
    if (first) {
      // warm-up so lazy model loading does not distort the latency column
      await pipeline.query({ question: first.item.question, history: [], settings });
    }
    const recalls = new Map(ks.map((k) => [k, 0]));
    let precision = 0;
    let mrr = 0;
    let faith = 0;
    let relevance = 0;
    let elapsed = 0;
    for (const { item, rel } of labelled) {
      const result = await pipeline.query({ question: item.question, history: [], settings });
      const ids = result.trace.ranked_chunk_ids;
      for (const k of ks)
        recalls.set(k, (recalls.get(k) as number) + metrics.recallAtK(ids, rel, k));
      precision += metrics.precisionAtK(ids, rel, topK);
      mrr += metrics.reciprocalRank(ids, rel);
      faith += metrics.faithfulnessOverlap(
        result.answer,
        result.citations.map((c) => c.text),
      );
      relevance += metrics.answerRelevance(item.question, result.answer);
      elapsed += result.trace.total_ms;
    }
    const n = Math.max(labelled.length, 1);
    const r = (v: number, d: number) => Math.round((v / n) * 10 ** d) / 10 ** d;
    rows.push({
      config: name,
      recall: Object.fromEntries([...recalls].map(([k, v]) => [`@${k}`, r(v, 3)])),
      precision: r(precision, 3),
      mrr: r(mrr, 3),
      faithfulness: r(faith, 3),
      answer_relevance: r(relevance, 3),
      avg_ms: r(elapsed, 1),
    });
  }
  return {
    embedder: pipeline.embedder.name,
    questions: labelled.length,
    ks,
    precision_k: topK,
    source,
    rows,
  };
}
