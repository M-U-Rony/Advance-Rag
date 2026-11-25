/**
 * RAGPipeline: orchestrates indexing (offline) and querying (online).
 *
 * **Offline indexing**: PDF -> pages -> chunks (+ parents) -> embeddings -> vector store + BM25.
 *
 * **Online querying**: Retrieve -> Augment -> Generate, expanded with the advanced stages:
 *
 * ```
 * condense history -> query transform -> vector search \
 *                                     -> BM25 search    -> fusion -> rerank
 *     -> parent expansion -> contextual compression -> grounded prompt -> LLM
 * ```
 *
 * Every stage records its latency, parameters and hits in a {@link Trace} so the UI (and the
 * evaluation harness) can show exactly what happened.
 */
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { Settings } from "./config.js";
import type { Embedder } from "./embeddings/types.js";
import { ExtractiveGenerator, type LLMClient } from "./generation/llm.js";
import {
  NO_ANSWER,
  buildGroundedPrompt,
  promptTokenEstimate,
  type ContextItem,
} from "./generation/prompts.js";
import { chunkDocument, type ChunkingParams } from "./ingestion/chunking.js";
import { loadPdf } from "./ingestion/pdfLoader.js";
import { BM25Index } from "./retrieval/bm25.js";
import { compressText } from "./retrieval/compression.js";
import { MetadataFilter } from "./retrieval/filters.js";
import { reciprocalRankFusion, weightedFusion } from "./retrieval/fusion.js";
import {
  condenseWithHistory,
  hydeDocument,
  multiQuery,
  rewriteQuery,
} from "./retrieval/queryTransform.js";
import { CrossEncoderReranker, HeuristicReranker, type Reranker } from "./retrieval/rerank.js";
import { VectorStore } from "./retrieval/vectorStore.js";
import type {
  Citation,
  Hit,
  QueryRequest,
  QueryResponse,
  RetrievalSettings,
  Stage,
  Trace,
} from "./schemas.js";
import { DocumentStore, nowIso, type DocumentRecord } from "./store.js";
import { round, snippet, wordCount } from "./text.js";
import { chunkMetadata, type Chunk, type ChunkSet } from "./types.js";
import { InvalidInputError } from "./errors.js";

/** `[stage key, label]` in pipeline order; this is also the order the UI shows them in. */
export const STAGE_LABELS: readonly (readonly [string, string])[] = [
  ["transform", "Query transform"],
  ["vector", "Vector search"],
  ["bm25", "BM25 keyword search"],
  ["fusion", "Fusion"],
  ["rerank", "Rerank"],
  ["parent", "Parent expansion"],
  ["compression", "Contextual compression"],
  ["generate", "Generate"],
];

/** Immutable-by-convention snapshot of everything searchable. */
export class CorpusIndex {
  constructor(
    readonly chunks: Chunk[] = [],
    readonly byId: Map<string, Chunk> = new Map(),
    readonly parents: Map<string, Chunk> = new Map(),
    readonly vectors: VectorStore = new VectorStore(),
    readonly bm25: BM25Index = new BM25Index(),
  ) {}
}

/** Ranked `[chunk id, score]` lists from first-stage retrieval of one query. */
interface SearchResult {
  vector: [string, number][];
  keyword: [string, number][];
  fused: [string, number][];
  vectorMs: number;
  bm25Ms: number;
}

/** Output of the retrieval half of the pipeline. */
export interface Retrieval {
  trace: Trace;
  contexts: Chunk[];
  contextScores: number[];
  texts: string[];
  items: ContextItem[];
  system: string;
  user: string;
}

/** One server-sent event: `trace` first, then many `token`, then `done`. */
export type StreamEvent =
  | { event: "trace"; data: Trace }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: QueryResponse };

/** Tiny stopwatch helper. */
class Clock {
  private start = performance.now();

  /** Milliseconds since the previous lap (or construction). */
  lap(): number {
    const now = performance.now();
    const elapsed = now - this.start;
    this.start = now;
    return round(elapsed, 2);
  }
}

/** Serialises async critical sections so concurrent uploads cannot corrupt the index. */
class Mutex {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.tail.then(fn, fn);
    this.tail = result.catch(() => undefined);
    return result;
  }
}

function makeHit(chunk: Chunk, score: number, rank: number, isParent = false, limit = 220): Hit {
  return {
    chunk_id: chunk.id,
    filename: chunk.filename,
    page: chunk.page,
    rank,
    score: round(score, 5),
    snippet: snippet(chunk.text, limit),
    is_parent: isParent,
  };
}

function rankedHits(index: CorpusIndex, scored: [string, number][], limit?: number): Hit[] {
  return scored
    .slice(0, limit)
    .map(([id, score], i) => makeHit(index.byId.get(id) as Chunk, score, i + 1));
}

/** Document indexing and advanced question answering. */
export class RAGPipeline {
  readonly extractive = new ExtractiveGenerator();
  readonly heuristicReranker = new HeuristicReranker();
  readonly store: DocumentStore;
  readonly documents = new Map<string, DocumentRecord>();
  index = new CorpusIndex();
  private readonly mutex = new Mutex();
  private readonly cacheDir: string;

  constructor(
    readonly settings: Settings,
    readonly embedder: Embedder,
    readonly llm: LLMClient | null,
    readonly crossEncoder: CrossEncoderReranker,
    readonly embedderFallbackReason: string | null = null,
  ) {
    this.store = new DocumentStore(settings.dataDir);
    this.cacheDir = path.join(settings.dataDir, "index", embedder.name.replaceAll("/", "_"));
  }

  /** Load persisted documents and (re)build the indexes. Call once after construction. */
  async init(): Promise<this> {
    const chunkSets = new Map<string, ChunkSet>();
    for (const [record, chunkSet] of this.store.loadAll()) {
      this.documents.set(record.id, record);
      chunkSets.set(record.id, chunkSet);
    }
    let cached = new VectorStore();
    if (!this.embedder.needsFit && existsSync(path.join(this.cacheDir, "vectors.json"))) {
      try {
        cached = VectorStore.load(this.cacheDir);
      } catch (error) {
        console.warn("Ignoring unreadable vector cache", error);
      }
    }
    await this.rebuild(chunkSets, cached);
    return this;
  }

  // ------------------------------------------------------------------ indexing

  /** Recompute the searchable index from stored chunks, re-using cached vectors. */
  private async rebuild(chunkSets: Map<string, ChunkSet>, cached?: VectorStore): Promise<void> {
    const chunks = [...chunkSets.values()].flatMap((cs) => cs.chunks);
    const parents = new Map([...chunkSets.values()].flatMap((cs) => [...cs.parents]));
    const texts = chunks.map((c) => c.text);
    const vectors = new VectorStore();
    if (chunks.length > 0) {
      let rows: Float32Array[];
      if (this.embedder.needsFit) {
        this.embedder.fit(texts);
        rows = await this.embedder.embedDocuments(texts);
      } else {
        // Dense vectors are expensive: embed only the chunks we have not seen before.
        const known = cached ?? this.index.vectors;
        const rowOf = new Map(known.ids.map((id, i) => [id, i]));
        const missing = chunks.filter((c) => !rowOf.has(c.id));
        const fresh =
          missing.length > 0 ? await this.embedder.embedDocuments(missing.map((c) => c.text)) : [];
        const freshRow = new Map(missing.map((c, i) => [c.id, i]));
        rows = chunks.map((c) =>
          rowOf.has(c.id)
            ? known.row(rowOf.get(c.id) as number)
            : (fresh[freshRow.get(c.id) as number] as Float32Array),
        );
      }
      vectors.add(
        chunks.map((c) => c.id),
        rows,
        chunks.map(chunkMetadata),
      );
      if (!this.embedder.needsFit) vectors.save(this.cacheDir);
    }
    const bm25 = new BM25Index();
    bm25.index(texts);
    this.index = new CorpusIndex(
      chunks,
      new Map(chunks.map((c) => [c.id, c])),
      parents,
      vectors,
      bm25,
    );
  }

  private chunkSets(): Map<string, ChunkSet> {
    const sets = new Map<string, ChunkSet>(
      [...this.documents.keys()].map((id) => [id, { chunks: [], parents: new Map() }]),
    );
    for (const chunk of this.index.chunks) sets.get(chunk.doc_id)?.chunks.push(chunk);
    for (const [pid, parent] of this.index.parents)
      sets.get(parent.doc_id)?.parents.set(pid, parent);
    return sets;
  }

  /**
   * Parse, chunk, embed and index a PDF.
   * @throws InvalidInputError for unusable input.
   */
  async addDocument(
    filename: string,
    data: Uint8Array,
    params: ChunkingParams,
  ): Promise<DocumentRecord> {
    const pages = await loadPdf(data);
    const docId = randomUUID().replaceAll("-", "").slice(0, 10);
    const chunkSet = chunkDocument(pages, docId, filename, params);
    if (chunkSet.chunks.length === 0) {
      throw new InvalidInputError("No text chunks could be produced from this PDF.");
    }
    const record: DocumentRecord = {
      id: docId,
      filename,
      pages: pages.length,
      chunks: chunkSet.chunks.length,
      parent_chunks: chunkSet.parents.size,
      strategy: params.strategy,
      chunk_size: params.strategy === "parent_child" ? params.child_size : params.chunk_size,
      chunk_overlap: params.chunk_overlap,
      created_at: nowIso(),
    };
    return this.mutex.run(async () => {
      const sets = this.chunkSets();
      sets.set(docId, chunkSet);
      await this.rebuild(sets);
      this.documents.set(docId, record);
      this.store.save(record, chunkSet);
      return record;
    });
  }

  /** Remove a document from the registry and the indexes. */
  deleteDocument(docId: string): Promise<boolean> {
    return this.mutex.run(async () => {
      if (!this.documents.has(docId)) return false;
      const sets = this.chunkSets();
      sets.delete(docId);
      this.documents.delete(docId);
      this.store.delete(docId);
      await this.rebuild(sets);
      return true;
    });
  }

  /** Look up an indexed document by its file name. */
  findByFilename(filename: string): DocumentRecord | undefined {
    return [...this.documents.values()].find((d) => d.filename === filename);
  }

  /** Index a PDF from disk. */
  addPdfFile(file: string, params: ChunkingParams): Promise<DocumentRecord> {
    return this.addDocument(path.basename(file), readFileSync(file), params);
  }

  // ------------------------------------------------------------------ retrieval

  private metadataFilter(s: RetrievalSettings): MetadataFilter {
    return new MetadataFilter(
      s.doc_ids?.length ? new Set(s.doc_ids) : null,
      s.page_min,
      s.page_max,
    );
  }

  /** Run first-stage retrieval for one query (vector + BM25 + fusion), timing each part. */
  private async searchOne(
    index: CorpusIndex,
    vectorQuery: string,
    keywordQuery: string,
    s: RetrievalSettings,
    filter: MetadataFilter,
  ): Promise<SearchResult> {
    let vecHits: [string, number][] = [];
    let kwHits: [string, number][] = [];
    const clock = new Clock();
    if (s.retrieval_mode === "vector" || s.retrieval_mode === "hybrid") {
      const qvec = await this.embedder.embedQuery(vectorQuery);
      vecHits = index.vectors.search(qvec, s.candidates, s.similarity, filter);
    }
    const vectorMs = clock.lap();
    if (s.retrieval_mode === "bm25" || s.retrieval_mode === "hybrid") {
      const mask = filter.active ? index.chunks.map((c) => filter.matches(chunkMetadata(c))) : null;
      kwHits = index.bm25
        .search(keywordQuery, s.candidates, mask)
        .map(([i, score]) => [(index.chunks[i] as Chunk).id, score]);
    }
    const bm25Ms = clock.lap();
    let fused: [string, number][];
    if (s.retrieval_mode === "vector") fused = vecHits;
    else if (s.retrieval_mode === "bm25") fused = kwHits;
    else if (s.fusion === "rrf") {
      fused = reciprocalRankFusion([vecHits.map(([i]) => i), kwHits.map(([i]) => i)]);
    } else {
      fused = weightedFusion(new Map(vecHits), new Map(kwHits), s.alpha);
    }
    return {
      vector: vecHits,
      keyword: kwHits,
      fused: fused.slice(0, s.candidates),
      vectorMs,
      bm25Ms,
    };
  }

  private reranker(method: string): Reranker | null {
    if (method === "heuristic") return this.heuristicReranker;
    if (method === "cross_encoder") return this.crossEncoder;
    return null;
  }

  /** Run every stage up to (and including) prompt construction. */
  async retrieve(request: QueryRequest): Promise<Retrieval> {
    const s = request.settings;
    const index = this.index;
    const clock = new Clock();
    const total = new Clock();
    const history = request.history;
    const stages = new Map<string, Stage>(
      STAGE_LABELS.map(([key, label]) => [
        key,
        { key, label, ms: 0, skipped: true, info: {}, hits: [] },
      ]),
    );
    const stage = (key: string) => stages.get(key) as Stage;
    const trace: Trace = {
      question: request.question,
      condensed_query: request.question,
      rewritten_query: null,
      hyde_text: null,
      multi_queries: [],
      stages: [...stages.values()],
      ranked_chunk_ids: [],
      prompt_tokens: 0,
      total_ms: 0,
      embedder: this.embedder.name,
      llm: this.llm ? this.llm.name : this.extractive.name,
      reranker: null,
      warnings: [],
      settings: s,
    };
    const filter = this.metadataFilter(s);

    // 1. Query transformation ------------------------------------------------
    const query = await condenseWithHistory(request.question, history, this.llm);
    trace.condensed_query = query;
    let vectorQuery = query;
    let keywordQuery = query;
    let extraQueries: string[] = [];
    const condensed = query !== request.question;
    if (s.query_transform === "rewrite") {
      trace.rewritten_query = vectorQuery = keywordQuery = await rewriteQuery(query, this.llm);
    } else if (s.query_transform === "hyde") {
      trace.hyde_text = vectorQuery = await hydeDocument(query, this.llm);
    } else if (s.query_transform === "multi_query") {
      trace.multi_queries = extraQueries = await multiQuery(query, this.llm);
    }
    const transformStage = stage("transform");
    transformStage.skipped = s.query_transform === "none" && !condensed;
    transformStage.info = {
      transform: s.query_transform,
      condensed,
      uses_llm: this.llm !== null && (s.query_transform !== "none" || condensed),
    };
    transformStage.ms = clock.lap();

    // 2-4. First-stage retrieval + fusion ------------------------------------
    clock.lap();
    const primary = await this.searchOne(index, vectorQuery, keywordQuery, s, filter);
    let fused = primary.fused;
    if (extraQueries.length > 0) {
      const perQuery = [fused];
      for (const variant of extraQueries) {
        perQuery.push((await this.searchOne(index, variant, variant, s, filter)).fused);
      }
      fused = reciprocalRankFusion(perQuery.map((list) => list.map(([i]) => i))).slice(
        0,
        s.candidates,
      );
    }
    if (s.retrieval_mode === "vector" || s.retrieval_mode === "hybrid") {
      const st = stage("vector");
      st.skipped = false;
      st.ms = primary.vectorMs;
      st.info = { metric: s.similarity, embedder: this.embedder.name, k: s.candidates };
      st.hits = rankedHits(index, primary.vector);
    }
    if (s.retrieval_mode === "bm25" || s.retrieval_mode === "hybrid") {
      const st = stage("bm25");
      st.skipped = false;
      st.ms = primary.bm25Ms;
      st.info = { k1: index.bm25.k1, b: index.bm25.b, k: s.candidates };
      st.hits = rankedHits(index, primary.keyword);
    }
    clock.lap();
    const fusionStage = stage("fusion");
    fusionStage.skipped = s.retrieval_mode !== "hybrid" && extraQueries.length === 0;
    const method = extraQueries.length > 0 && s.retrieval_mode !== "hybrid" ? "rrf" : s.fusion;
    fusionStage.info = {
      method,
      alpha: method === "weighted" ? s.alpha : null,
      rrf_k: method === "rrf" ? 60 : null,
      queries: 1 + extraQueries.length,
    };
    fusionStage.hits = rankedHits(index, fused);
    fusionStage.ms = clock.lap();

    // 5. Rerank ----------------------------------------------------------------
    let ranked = fused;
    const reranker = this.reranker(s.rerank);
    if (reranker && fused.length > 0) {
      const candidates = fused.map(([id]): [string, string] => [
        id,
        (index.byId.get(id) as Chunk).text,
      ]);
      ranked = await reranker.rerank(query, candidates);
      trace.reranker = reranker.name;
      const st = stage("rerank");
      st.skipped = false;
      st.info = { reranker: reranker.name, candidates: candidates.length, kept: s.top_k };
      st.hits = rankedHits(index, ranked, s.top_k);
      if (reranker instanceof CrossEncoderReranker && reranker.failure) {
        trace.warnings.push(`Cross-encoder unavailable, used heuristic: ${reranker.failure}`);
      }
    }
    stage("rerank").ms = clock.lap();
    const selected = ranked.slice(0, s.top_k);
    trace.ranked_chunk_ids = selected.map(([id]) => id);

    // 6. Parent expansion ----------------------------------------------------
    let contexts: Chunk[] = selected.map(([id]) => index.byId.get(id) as Chunk);
    let scores = selected.map(([, score]) => score);
    if (s.parent_child && contexts.some((c) => c.parent_id)) {
      const expanded = new Map<string, [Chunk, number]>();
      contexts.forEach((chunk, i) => {
        const target = index.parents.get(chunk.parent_id ?? "") ?? chunk;
        if (!expanded.has(target.id)) expanded.set(target.id, [target, scores[i] as number]);
      });
      const st = stage("parent");
      st.skipped = false;
      st.info = { children: contexts.length, parents: expanded.size };
      contexts = [...expanded.values()].map(([c]) => c);
      scores = [...expanded.values()].map(([, sc]) => sc);
      st.hits = contexts.map((c, i) => makeHit(c, scores[i] as number, i + 1, true, 320));
    }
    stage("parent").ms = clock.lap();

    // 7. Contextual compression ------------------------------------------------
    let texts = contexts.map((c) => c.text);
    if (s.compression && contexts.length > 0) {
      const before = texts.reduce((n, t) => n + wordCount(t), 0);
      texts = texts.map((t) => compressText(query, t));
      const st = stage("compression");
      st.skipped = false;
      st.info = { words_before: before, words_after: texts.reduce((n, t) => n + wordCount(t), 0) };
      st.hits = contexts.map((c, i) => ({
        chunk_id: c.id,
        filename: c.filename,
        page: c.page,
        rank: i + 1,
        score: round(scores[i] as number, 5),
        snippet: snippet(texts[i] as string, 400),
        is_parent: index.parents.has(c.id),
      }));
    }
    stage("compression").ms = clock.lap();

    // 8. Prompt construction (generation happens in query()/queryStream()) -----
    const items: ContextItem[] = contexts.map((c, i) => ({
      index: i + 1,
      filename: c.filename,
      page: c.page,
      text: texts[i] as string,
    }));
    const [system, user] = buildGroundedPrompt(request.question, items, history);
    trace.prompt_tokens = promptTokenEstimate(system, user);
    trace.total_ms = total.lap();
    return { trace, contexts, contextScores: scores, texts, items, system, user };
  }

  // ------------------------------------------------------------------ generation

  private citations(r: Retrieval): Citation[] {
    return r.items.map((item, i) => {
      const chunk = r.contexts[i] as Chunk;
      return {
        index: item.index,
        chunk_id: chunk.id,
        doc_id: chunk.doc_id,
        filename: chunk.filename,
        page: chunk.page,
        score: round(r.contextScores[i] as number, 5),
        text: item.text,
      };
    });
  }

  private finish(r: Retrieval, answer: string, generateMs: number): QueryResponse {
    const st = r.trace.stages.find((x) => x.key === "generate") as Stage;
    st.skipped = false;
    st.ms = round(generateMs, 2);
    st.info = {
      llm: r.trace.llm,
      prompt_tokens: r.trace.prompt_tokens,
      context_passages: r.items.length,
    };
    r.trace.total_ms = round(
      r.trace.stages.reduce((sum, x) => sum + x.ms, 0),
      2,
    );
    return { answer, citations: this.citations(r), trace: r.trace };
  }

  /** Answer a question (non-streaming). */
  async query(request: QueryRequest): Promise<QueryResponse> {
    const r = await this.retrieve(request);
    const t0 = performance.now();
    let answer = "";
    for await (const token of this.generateTokens(r, request.question)) answer += token;
    return this.finish(r, answer, performance.now() - t0);
  }

  /** Yield `trace`, many `token`, then `done` events. */
  async *queryStream(request: QueryRequest): AsyncGenerator<StreamEvent> {
    const r = await this.retrieve(request);
    yield { event: "trace", data: structuredClone(r.trace) };
    const t0 = performance.now();
    const parts: string[] = [];
    for await (const token of this.generateTokens(r, request.question)) {
      parts.push(token);
      yield { event: "token", data: { text: token } };
    }
    yield { event: "done", data: this.finish(r, parts.join(""), performance.now() - t0) };
  }

  private async *generateTokens(r: Retrieval, question: string): AsyncGenerator<string> {
    if (r.items.length === 0) {
      yield NO_ANSWER;
      return;
    }
    if (this.llm) {
      let produced = false;
      try {
        for await (const token of this.llm.stream(r.system, r.user)) {
          produced = true;
          yield token;
        }
        return;
      } catch (error) {
        const name = error instanceof Error ? error.name : "Error";
        r.trace.warnings.push(`LLM call failed (${name}); used offline answer.`);
        console.warn("LLM call failed", error);
        if (produced) yield "\n\n";
      }
    }
    yield* this.extractive.stream(this.extractive.answer(question, r.items));
  }
}
