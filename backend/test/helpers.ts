/** Shared test helpers: an in-memory PDF builder, fake LLM / embedder, and pipeline factories. */
import { mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import PDFDocument from "pdfkit";
import { createApp } from "../src/app.js";
import { loadSettings, REPO_ROOT, type Settings } from "../src/config.js";
import { TfidfEmbedder } from "../src/embeddings/tfidf.js";
import type { Embedder } from "../src/embeddings/types.js";
import type { LLMClient } from "../src/generation/llm.js";
import { chunkingParams } from "../src/ingestion/chunking.js";
import { RAGPipeline } from "../src/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieval/rerank.js";
import {
  retrievalSettingsSchema,
  type QueryRequest,
  type RetrievalSettings,
} from "../src/schemas.js";
import { tokenize } from "../src/text.js";

export const PAGES: [heading: string, body: string][] = [
  [
    "Photosynthesis in Plants",
    "Photosynthesis converts sunlight into chemical energy. Chlorophyll inside the " +
      "chloroplasts absorbs red and blue light. The plant combines carbon dioxide and water " +
      "to produce glucose and oxygen. This process feeds almost every food chain on Earth.",
  ],
  [
    "Volcanic Eruptions",
    "A volcano erupts when magma rises through cracks in the crust. Pressure from dissolved " +
      "gases drives explosive eruptions. Basaltic lava flows slowly while rhyolitic magma " +
      "produces violent ash clouds. Monitoring seismic tremors helps scientists forecast events.",
  ],
  [
    "Ocean Tides",
    "Tides are caused mainly by the gravity of the Moon pulling on the oceans. Spring tides " +
      "occur when the Sun and Moon align, producing the largest tidal range. Neap tides happen " +
      "at quarter moons when the pull partially cancels. Coastal harbours schedule ships around " +
      "high tide.",
  ],
];

/** Build a small multi-page PDF in memory (one `[heading, body]` pair per page). */
export function makePdf(pages: [string, string][] = PAGES): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({ size: "A4", margin: 60 });
    const parts: Buffer[] = [];
    doc.on("data", (d: Buffer) => parts.push(d));
    doc.on("end", () => resolve(new Uint8Array(Buffer.concat(parts))));
    doc.on("error", reject);
    pages.forEach(([heading, body], i) => {
      if (i > 0) doc.addPage();
      if (heading) doc.font("Helvetica-Bold").fontSize(20).text(heading).moveDown(0.5);
      if (body) doc.font("Helvetica").fontSize(11).text(body);
    });
    doc.end();
  });
}

/** Deterministic stand-in for a real model. */
export class FakeLLM implements LLMClient {
  name = "fake";
  calls = 0;

  constructor(
    private readonly reply = "fake reply",
    private readonly fail = false,
    private readonly failAfter: string | null = null,
  ) {}

  complete(): Promise<string> {
    this.calls++;
    return this.fail ? Promise.reject(new Error("boom")) : Promise.resolve(this.reply);
  }

  async *stream(): AsyncGenerator<string> {
    if (this.failAfter) {
      yield this.failAfter;
      throw new Error("mid-stream failure");
    }
    if (this.fail) throw new Error("boom");
    yield "fake ";
    yield "answer [1]";
  }
}

/** Deterministic non-fitted "dense" embedder (hashed bag of words) that counts its work. */
export class HashEmbedder implements Embedder {
  readonly name = "hash-dense";
  readonly needsFit = false;
  embeddedTexts = 0;

  constructor(private readonly dim = 64) {}

  fit(): void {}

  private vec(text: string): Float32Array {
    const v = new Float32Array(this.dim);
    for (const token of tokenize(text, true)) {
      let h = 0;
      for (const ch of token) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
      v[h % this.dim] = (v[h % this.dim] as number) + 1;
    }
    const norm = Math.hypot(...v);
    return norm ? v.map((x) => x / norm) : v;
  }

  embedDocuments(texts: string[]): Promise<Float32Array[]> {
    this.embeddedTexts += texts.length;
    return Promise.resolve(texts.map((t) => this.vec(t)));
  }

  embedQuery(text: string): Promise<Float32Array> {
    return Promise.resolve(this.vec(text));
  }
}

/** Offline settings writing to a fresh temporary data dir. */
export function makeSettings(overrides: Partial<Settings> = {}): Settings {
  const dataDir = mkdtempSync(path.join(os.tmpdir(), "advance-rag-test-"));
  return loadSettings(
    { LLM_PROVIDER: "none", EMBEDDING_BACKEND: "tfidf" },
    {
      dataDir,
      modelCacheDir: path.join(dataDir, "models"),
      samplesDir: path.join(REPO_ROOT, "samples"),
      staticDir: path.join(dataDir, "no-static"),
      ...overrides,
    },
  );
}

/** A cross-encoder whose model can never be loaded (exercises the heuristic fallback). */
export function brokenCrossEncoder(): CrossEncoderReranker {
  return new CrossEncoderReranker(() => Promise.reject(new Error("model not found")));
}

/** Empty offline pipeline (TF-IDF, no LLM). */
export async function makePipeline(
  settings: Settings = makeSettings(),
  options: { embedder?: Embedder; llm?: LLMClient | null } = {},
): Promise<RAGPipeline> {
  return new RAGPipeline(
    settings,
    options.embedder ?? new TfidfEmbedder(),
    options.llm ?? null,
    brokenCrossEncoder(),
  ).init();
}

/** Pipeline with the test PDF indexed using small recursive chunks. */
export async function makeIndexed(): Promise<RAGPipeline> {
  const pipeline = await makePipeline();
  await pipeline.addDocument(
    "nature.pdf",
    await makePdf(),
    chunkingParams({ chunk_size: 30, chunk_overlap: 5 }),
  );
  return pipeline;
}

/** Build a query request with default settings overridden by `settings`. */
export function req(
  question: string,
  settings: Partial<RetrievalSettings> = {},
  history: QueryRequest["history"] = [],
): QueryRequest {
  return { question, history, settings: retrievalSettingsSchema.parse(settings) };
}

/** Fastify app on an offline pipeline whose cross-encoder never touches the network. */
export async function makeApp(overrides: Partial<Settings> = {}): Promise<FastifyInstance> {
  const settings = makeSettings(overrides);
  return createApp(settings, await makePipeline(settings));
}
