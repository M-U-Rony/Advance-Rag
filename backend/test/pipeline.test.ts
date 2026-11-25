import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { REPO_ROOT } from "../src/config.js";
import { NO_ANSWER } from "../src/generation/prompts.js";
import { chunkingParams } from "../src/ingestion/chunking.js";
import type { RAGPipeline } from "../src/pipeline.js";
import { type StreamEvent } from "../src/pipeline.js";
import type { TfidfEmbedder } from "../src/embeddings/tfidf.js";
import type { QueryResponse, RetrievalSettings } from "../src/schemas.js";
import {
  FakeLLM,
  brokenCrossEncoder,
  makeIndexed,
  makePdf,
  makePipeline,
  makeSettings,
  req,
} from "./helpers.js";

const stage = (r: QueryResponse, key: string) => r.trace.stages.find((s) => s.key === key)!;
const ask = (p: RAGPipeline, q: string, s: Partial<RetrievalSettings> = {}) => p.query(req(q, s));

describe("indexing", () => {
  it("registers document metadata", async () => {
    const pipeline = await makePipeline();
    const record = await pipeline.addDocument(
      "nature.pdf",
      await makePdf(),
      chunkingParams({ chunk_size: 30 }),
    );
    expect(record.pages).toBe(3);
    expect(record.chunks).toBeGreaterThanOrEqual(3);
    expect(record.id).toMatch(/^[0-9a-f]{10}$/);
    expect(record.created_at).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$/);
    expect(pipeline.documents.get(record.id)?.filename).toBe("nature.pdf");
    expect(pipeline.findByFilename("nature.pdf")).toEqual(record);
    expect(pipeline.findByFilename("nope.pdf")).toBeUndefined();
    expect(new Set(pipeline.index.chunks.map((c) => c.page))).toEqual(new Set([1, 2, 3]));
  });

  it("indexes the sample handbook from disk", async () => {
    const pipeline = await makePipeline();
    const record = await pipeline.addPdfFile(
      path.join(REPO_ROOT, "samples", "rag_handbook.pdf"),
      chunkingParams({ strategy: "parent_child" }),
    );
    expect(record).toMatchObject({
      pages: 6,
      strategy: "parent_child",
      chunk_size: 50,
      filename: "rag_handbook.pdf",
    });
    expect(record.parent_chunks).toBeGreaterThan(0);
  });

  it("serialises concurrent uploads", async () => {
    const pipeline = await makePipeline();
    const pdf = await makePdf();
    const params = chunkingParams({ chunk_size: 30 });
    const records = await Promise.all(
      [1, 2, 3].map((i) => pipeline.addDocument(`d${i}.pdf`, pdf, params)),
    );
    expect(pipeline.documents.size).toBe(3);
    expect(pipeline.index.chunks.length).toBe(records.reduce((n, r) => n + r.chunks, 0));
    expect(pipeline.index.vectors.size).toBe(pipeline.index.chunks.length);
  });
});

describe("answering", () => {
  it.each(["vector", "bm25", "hybrid"] as const)(
    "answers with a citation in %s mode",
    async (mode) => {
      for (const fusion of ["rrf", "weighted"] as const) {
        const pipeline = await makeIndexed();
        const r = await ask(pipeline, "What do chlorophyll and chloroplasts absorb?", {
          retrieval_mode: mode,
          fusion,
          rerank: "none",
        });
        expect(r.citations.length).toBeGreaterThan(0);
        expect(r.citations[0]).toMatchObject({ page: 1, filename: "nature.pdf" });
        expect(r.answer).toContain("[1]");
        expect(r.trace.embedder).toBe("tfidf");
        expect(r.trace.llm).toBe("Offline (extractive)");
        expect(stage(r, "vector").skipped).toBe(mode === "bm25");
        expect(stage(r, "bm25").skipped).toBe(mode === "vector");
        expect(stage(r, "fusion").skipped).toBe(mode !== "hybrid");
        expect(stage(r, "generate").skipped).toBe(false);
        expect(r.trace.prompt_tokens).toBeGreaterThan(0);
        expect(r.trace.total_ms).toBeGreaterThan(0);
      }
    },
  );

  it("records all eight stages in order", async () => {
    const r = await ask(await makeIndexed(), "What causes spring tides?", {
      retrieval_mode: "hybrid",
      fusion: "weighted",
      alpha: 0.4,
      top_k: 3,
    });
    expect(r.trace.stages.map((s) => s.key)).toEqual([
      "transform",
      "vector",
      "bm25",
      "fusion",
      "rerank",
      "parent",
      "compression",
      "generate",
    ]);
    expect(stage(r, "fusion").info).toEqual({
      method: "weighted",
      alpha: 0.4,
      rrf_k: null,
      queries: 1,
    });
    expect(r.citations[0]?.page).toBe(3);
  });

  it("refuses unanswerable questions", async () => {
    const r = await ask(await makeIndexed(), "Who won the football world cup in 1998?");
    expect(r.answer).toBe(NO_ANSWER);
  });

  it("answers with the refusal when nothing is indexed", async () => {
    const r = await ask(await makePipeline(), "anything");
    expect(r.citations).toEqual([]);
    expect(r.answer).toBe(NO_ANSWER);
  });

  it("reranks with the heuristic and records it", async () => {
    const r = await ask(await makeIndexed(), "How do tides depend on the Moon gravity?", {
      rerank: "heuristic",
      top_k: 2,
    });
    const st = stage(r, "rerank");
    expect(st.skipped).toBe(false);
    expect(st.info.reranker).toBe("heuristic");
    expect(st.hits).toHaveLength(2);
    expect(r.citations).toHaveLength(2);
    expect(r.citations[0]?.page).toBe(3);
  });

  it("never crashes when the cross-encoder is unavailable", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const r = await ask(await makeIndexed(), "volcano magma eruption", { rerank: "cross_encoder" });
    expect(r.citations[0]?.page).toBe(2);
    expect(r.trace.warnings.some((w) => w.includes("Cross-encoder unavailable"))).toBe(true);
    expect(r.trace.reranker).toContain("heuristic");
    vi.restoreAllMocks();
  });

  it("filters by document and page range", async () => {
    const pipeline = await makeIndexed();
    const other = await pipeline.addDocument(
      "other.pdf",
      await makePdf(),
      chunkingParams({ chunk_size: 30 }),
    );
    const natureId = [...pipeline.documents.values()].find((d) => d.filename === "nature.pdf")!.id;
    const names = (r: QueryResponse) => new Set(r.citations.map((c) => c.filename));
    expect(names(await ask(pipeline, "moon tides", { doc_ids: [natureId], top_k: 10 }))).toEqual(
      new Set(["nature.pdf"]),
    );
    expect(names(await ask(pipeline, "moon tides", { doc_ids: [other.id], top_k: 10 }))).toEqual(
      new Set(["other.pdf"]),
    );
    const pages = await ask(pipeline, "moon tides volcano", {
      page_min: 2,
      page_max: 2,
      top_k: 10,
    });
    expect(new Set(pages.citations.map((c) => c.page))).toEqual(new Set([2]));
    const none = await ask(pipeline, "moon", { doc_ids: ["missing"] });
    expect(none.citations).toEqual([]);
    expect(none.answer).toBe(NO_ANSWER);
  });

  it.each(["rewrite", "hyde", "multi_query"] as const)(
    "applies the %s transform offline",
    async (transform) => {
      const r = await ask(await makeIndexed(), "How do volcanoes erupt?", {
        query_transform: transform,
      });
      expect(stage(r, "transform").skipped).toBe(false);
      if (transform === "rewrite") expect(r.trace.rewritten_query).toBeTruthy();
      if (transform === "hyde") expect(r.trace.hyde_text?.split(" ").length).toBeGreaterThan(10);
      if (transform === "multi_query") {
        expect(r.trace.multi_queries.length).toBeGreaterThan(0);
        expect(stage(r, "fusion").info.queries).toBe(1 + r.trace.multi_queries.length);
        expect(stage(r, "fusion").skipped).toBe(false);
      }
      expect(r.citations.length).toBeGreaterThan(0);
    },
  );

  it("multi-query in vector mode fuses with RRF", async () => {
    const r = await ask(await makeIndexed(), "How do volcanoes erupt?", {
      query_transform: "multi_query",
      retrieval_mode: "vector",
      fusion: "weighted",
    });
    expect(stage(r, "fusion").info.method).toBe("rrf");
    expect(stage(r, "fusion").info.rrf_k).toBe(60);
  });

  it("uses the LLM for transforms and generation when present", async () => {
    const pipeline = await makePipeline(makeSettings(), { llm: new FakeLLM("volcano magma") });
    await pipeline.addDocument("n.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    const r = await ask(pipeline, "tell me things", { query_transform: "rewrite" });
    expect(r.trace.rewritten_query).toBe("volcano magma");
    expect(r.trace.llm).toBe("fake");
    expect(r.answer.startsWith("fake answer")).toBe(true);
    expect(stage(r, "transform").info.uses_llm).toBe(true);
  });

  it("falls back to the extractive answer when the LLM fails", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const pipeline = await makePipeline(makeSettings(), { llm: new FakeLLM("x", true) });
    await pipeline.addDocument("n.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    const r = await ask(pipeline, "What absorbs red and blue light?");
    expect(r.answer).toContain("[1]");
    expect(r.trace.warnings.some((w) => w.includes("LLM call failed"))).toBe(true);
    vi.restoreAllMocks();
  });

  it("keeps partial LLM output and appends the offline answer after a mid-stream failure", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const pipeline = await makePipeline(makeSettings(), {
      llm: new FakeLLM("x", false, "Partial"),
    });
    await pipeline.addDocument("n.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    const r = await ask(pipeline, "What absorbs red and blue light?");
    expect(r.answer.startsWith("Partial\n\n")).toBe(true);
    expect(r.answer).toContain("[1]");
    vi.restoreAllMocks();
  });

  it("condenses follow-up questions using the history", async () => {
    const r = await (
      await makeIndexed()
    ).query(
      req("What about its pull?", {}, [
        { role: "user", content: "Explain ocean tides" },
        { role: "assistant", content: "Tides come from the Moon." },
      ]),
    );
    expect(r.trace.condensed_query).toContain("ocean tides");
    expect(r.citations[0]?.page).toBe(3);
    expect(stage(r, "transform").info.condensed).toBe(true);
  });

  it("expands children to parents", async () => {
    const pipeline = await makePipeline();
    await pipeline.addDocument(
      "n.pdf",
      await makePdf(),
      chunkingParams({ strategy: "parent_child", parent_size: 60, child_size: 15 }),
    );
    const plain = await ask(pipeline, "What drives explosive eruptions?", { top_k: 1 });
    const expanded = await ask(pipeline, "What drives explosive eruptions?", {
      top_k: 1,
      parent_child: true,
    });
    const st = stage(expanded, "parent");
    expect(st.skipped).toBe(false);
    expect(st.info.parents).toBe(1);
    expect(st.hits[0]?.is_parent).toBe(true);
    expect((expanded.citations[0]?.text ?? "").split(" ").length).toBeGreaterThan(
      (plain.citations[0]?.text ?? "").split(" ").length,
    );
    expect(expanded.citations[0]?.chunk_id.split("-")[1]).toMatch(/^p/);
    expect(stage(plain, "parent").skipped).toBe(true);
  });

  it("ignores parent_child for flat chunks", async () => {
    const r = await ask(await makeIndexed(), "tides moon", { parent_child: true });
    expect(stage(r, "parent").skipped).toBe(true);
  });

  it("compression shrinks the context", async () => {
    const pipeline = await makeIndexed();
    const q = "What do volcanoes monitor seismic tremors?";
    const plain = await ask(pipeline, q, { top_k: 1 });
    const compressed = await ask(pipeline, q, { top_k: 1, compression: true });
    const st = stage(compressed, "compression");
    expect(st.skipped).toBe(false);
    expect(st.info.words_after as number).toBeLessThanOrEqual(st.info.words_before as number);
    expect((compressed.citations[0]?.text ?? "").length).toBeLessThanOrEqual(
      (plain.citations[0]?.text ?? "").length,
    );
  });

  it.each(["cosine", "dot", "l2"] as const)("supports the %s similarity metric", async (metric) => {
    const r = await ask(await makeIndexed(), "chlorophyll", {
      retrieval_mode: "vector",
      similarity: metric,
    });
    expect(stage(r, "vector").info.metric).toBe(metric);
    expect(r.citations[0]?.page).toBe(1);
  });
});

describe("streaming", () => {
  it("emits trace, tokens and done in order", async () => {
    const pipeline = await makeIndexed();
    const events: StreamEvent[] = [];
    for await (const e of pipeline.queryStream(req("What absorbs red and blue light?")))
      events.push(e);
    const names = events.map((e) => e.event);
    expect(names[0]).toBe("trace");
    expect(names.at(-1)).toBe("done");
    expect(names).toContain("token");
    const streamed = events.flatMap((e) => (e.event === "token" ? [e.data.text] : [])).join("");
    const done = events.at(-1) as Extract<StreamEvent, { event: "done" }>;
    expect(streamed).toBe(done.data.answer);
    expect(done.data.citations.length).toBeGreaterThan(0);
    // the trace event is a snapshot taken before generation finished
    const first = events[0] as Extract<StreamEvent, { event: "trace" }>;
    expect(first.data.stages.find((s) => s.key === "generate")?.skipped).toBe(true);
  });
});

describe("persistence", () => {
  it("round-trips through disk and supports delete", async () => {
    const settings = makeSettings();
    const first = await makePipeline(settings);
    const record = await first.addDocument(
      "n.pdf",
      await makePdf(),
      chunkingParams({ chunk_size: 30 }),
    );
    const reloaded = await makePipeline(settings);
    expect([...reloaded.documents.keys()]).toEqual([record.id]);
    expect((await ask(reloaded, "chlorophyll")).citations.length).toBeGreaterThan(0);
    expect(await reloaded.deleteDocument(record.id)).toBe(true);
    expect(await reloaded.deleteDocument(record.id)).toBe(false);
    expect((await makePipeline(settings)).documents.size).toBe(0);
    expect(reloaded.index.chunks).toHaveLength(0);
  });

  it("loads documents written by the stable on-disk format (snake_case fields)", async () => {
    const settings = makeSettings();
    const first = await makePipeline(settings);
    const record = await first.addDocument(
      "n.pdf",
      await makePdf(),
      chunkingParams({ chunk_size: 30 }),
    );
    const file = path.join(settings.dataDir, "docs", `${record.id}.json`);
    const stored = JSON.parse(readFileSync(file, "utf-8"));
    expect(Object.keys(stored)).toEqual(["record", "chunks", "parents"]);
    expect(Object.keys(stored.chunks[0])).toEqual([
      "id",
      "doc_id",
      "filename",
      "page",
      "position",
      "text",
      "heading",
      "parent_id",
    ]);
  });

  it("re-fits TF-IDF when documents change but keeps one vector per chunk", async () => {
    const pipeline = await makeIndexed();
    const emb = pipeline.embedder as TfidfEmbedder;
    const before = emb.dim;
    await pipeline.addDocument("extra.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    expect(emb.dim).toBe(before);
    expect(pipeline.index.vectors.size).toBe(pipeline.index.chunks.length);
  });

  it("exposes its collaborators", async () => {
    const pipeline = await makePipeline();
    expect(pipeline.crossEncoder.available).toBeNull();
    expect(brokenCrossEncoder().failure).toBeNull();
  });
});
