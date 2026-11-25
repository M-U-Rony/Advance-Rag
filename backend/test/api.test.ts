import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import { afterEach, describe, expect, it } from "vitest";
import type { Settings } from "../src/config.js";
import { formatSse } from "../src/api/routes.js";
import { makeApp, makePdf } from "./helpers.js";

const apps: FastifyInstance[] = [];
async function appWith(overrides: Partial<Settings> = {}): Promise<FastifyInstance> {
  const app = await makeApp(overrides);
  apps.push(app);
  return app;
}
afterEach(async () => {
  await Promise.all(apps.splice(0).map((a) => a.close()));
});

/** Encode a multipart/form-data body with the platform's own FormData implementation. */
async function multipart(
  file: { name: string; data: Uint8Array } | null,
  fields: Record<string, string> = {},
) {
  const form = new FormData();
  if (file) form.append("file", new Blob([file.data], { type: "application/pdf" }), file.name);
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  const response = new Response(form);
  return {
    payload: Buffer.from(await response.arrayBuffer()),
    headers: { "content-type": response.headers.get("content-type") as string },
  };
}

async function upload(app: FastifyInstance, fields: Record<string, string> = {}, pdf?: Uint8Array) {
  const body = await multipart({ name: "nature.pdf", data: pdf ?? (await makePdf()) }, fields);
  return app.inject({ method: "POST", url: "/api/documents", ...body });
}

const post = (app: FastifyInstance, url: string, payload: unknown) =>
  app.inject({ method: "POST", url, payload: payload as object });

describe("health and config", () => {
  it("reports status and the active backends", async () => {
    const app = await appWith();
    expect((await app.inject("/api/health")).json()).toEqual({ status: "ok", version: "1.0.0" });
    const cfg = (await app.inject("/api/config")).json();
    expect(cfg.llm).toEqual({ provider: "none", label: "Offline", model: null, offline: true });
    expect(cfg.embeddings).toEqual({
      requested: "tfidf",
      active: "tfidf",
      dense: false,
      fallback_reason: null,
    });
    expect(cfg.reranker).toEqual({
      model: "Xenova/ms-marco-MiniLM-L-6-v2",
      cross_encoder_available: null,
    });
    expect(cfg.sample_available).toBe(true);
  });

  it("reports a missing sample", async () => {
    const app = await appWith({ samplesDir: path.join(os.tmpdir(), "no-samples-here") });
    expect((await app.inject("/api/config")).json().sample_available).toBe(false);
    expect((await app.inject({ method: "POST", url: "/api/documents/sample" })).statusCode).toBe(
      404,
    );
  });
});

describe("documents", () => {
  it("uploads, lists and deletes", async () => {
    const app = await appWith();
    const res = await upload(app, { strategy: "fixed", chunk_size: "30", chunk_overlap: "5" });
    expect(res.statusCode).toBe(201);
    const doc = res.json();
    expect(doc).toMatchObject({ pages: 3, strategy: "fixed", filename: "nature.pdf" });
    expect(doc.chunks).toBeGreaterThan(0);
    expect(Object.keys(doc).sort()).toEqual([
      "chunk_overlap",
      "chunk_size",
      "chunks",
      "created_at",
      "filename",
      "id",
      "pages",
      "parent_chunks",
      "strategy",
    ]);
    expect((await app.inject("/api/documents")).json().map((d: { id: string }) => d.id)).toEqual([
      doc.id,
    ]);
    expect(
      (await app.inject({ method: "DELETE", url: `/api/documents/${doc.id}` })).statusCode,
    ).toBe(204);
    const again = await app.inject({ method: "DELETE", url: `/api/documents/${doc.id}` });
    expect(again.statusCode).toBe(404);
    expect(again.json()).toEqual({ detail: "Document not found" });
    expect((await app.inject("/api/documents")).json()).toEqual([]);
  });

  it("validates the upload", async () => {
    const app = await appWith();
    const bad = await app.inject({
      method: "POST",
      url: "/api/documents",
      ...(await multipart({ name: "x.pdf", data: new TextEncoder().encode("nope") })),
    });
    expect(bad.statusCode).toBe(422);
    expect(bad.json().detail).toMatch(/Could not read PDF/);
    expect((await upload(app, { chunk_size: "20", chunk_overlap: "30" })).statusCode).toBe(422);
    expect((await upload(app, { strategy: "magic" })).statusCode).toBe(422);
    const range = await upload(app, { chunk_size: "5" });
    expect(range.statusCode).toBe(422);
    expect(Array.isArray(range.json().detail)).toBe(true);
    const missing = await app.inject({
      method: "POST",
      url: "/api/documents",
      ...(await multipart(null, { strategy: "fixed" })),
    });
    expect(missing.statusCode).toBe(422);
    expect(missing.json().detail[0]).toMatchObject({
      loc: ["body", "file"],
      msg: "Field required",
    });
  });

  it("rejects non-multipart uploads", async () => {
    const app = await appWith();
    const res = await app.inject({ method: "POST", url: "/api/documents", payload: { a: 1 } });
    expect(res.statusCode).toBe(422);
  });

  it("enforces the upload size limit", async () => {
    const app = await appWith({ maxUploadMb: 0 });
    const res = await upload(app);
    expect(res.statusCode).toBe(413);
    expect(res.json().detail).toContain("limit");
  });

  it("ignores unknown extra files in the form", async () => {
    const app = await appWith();
    const form = new FormData();
    form.append("file", new Blob([await makePdf()]), "a.pdf");
    form.append("other", new Blob(["zzz"]), "b.txt");
    const response = new Response(form);
    const res = await app.inject({
      method: "POST",
      url: "/api/documents",
      payload: Buffer.from(await response.arrayBuffer()),
      headers: { "content-type": response.headers.get("content-type") as string },
    });
    expect(res.statusCode).toBe(201);
  });

  it("loads the sample once, with parent-child chunking", async () => {
    const app = await appWith();
    const first = await app.inject({ method: "POST", url: "/api/documents/sample" });
    expect(first.statusCode).toBe(201);
    expect(first.json()).toMatchObject({
      strategy: "parent_child",
      pages: 6,
      filename: "rag_handbook.pdf",
    });
    const again = (await app.inject({ method: "POST", url: "/api/documents/sample" })).json();
    expect(again.id).toBe(first.json().id);
    expect((await app.inject("/api/documents")).json()).toHaveLength(1);
  });
});

describe("query", () => {
  it("requires documents (409) after validating the body (422)", async () => {
    const app = await appWith();
    expect((await post(app, "/api/query", { question: "hi" })).statusCode).toBe(409);
    expect((await post(app, "/api/query/stream", { question: "hi" })).statusCode).toBe(409);
    expect((await post(app, "/api/evaluate", {})).statusCode).toBe(409);
    const res = await post(app, "/api/query", { question: "hi" });
    expect(res.json().detail).toContain("No documents indexed");
  });

  it("returns the answer, citations and a trace", async () => {
    const app = await appWith();
    await upload(app, { chunk_size: "30", chunk_overlap: "5" });
    const body = {
      question: "What causes spring tides?",
      history: [],
      settings: { retrieval_mode: "hybrid", fusion: "weighted", alpha: 0.4, top_k: 3 },
    };
    const res = await post(app, "/api/query", body);
    expect(res.statusCode).toBe(200);
    const data = res.json();
    expect(data.answer).toContain("[1]");
    expect(data.citations[0].page).toBe(3);
    expect(Object.keys(data.citations[0]).sort()).toEqual([
      "chunk_id",
      "doc_id",
      "filename",
      "index",
      "page",
      "score",
      "text",
    ]);
    expect(data.trace.stages.map((s: { key: string }) => s.key)).toEqual([
      "transform",
      "vector",
      "bm25",
      "fusion",
      "rerank",
      "parent",
      "compression",
      "generate",
    ]);
    expect(data.trace.settings.alpha).toBe(0.4);
    expect(Object.keys(data.trace.stages[1]).sort()).toEqual([
      "hits",
      "info",
      "key",
      "label",
      "ms",
      "skipped",
    ]);
    expect(Object.keys(data.trace).sort()).toEqual(
      [
        "condensed_query",
        "embedder",
        "hyde_text",
        "llm",
        "multi_queries",
        "prompt_tokens",
        "question",
        "ranked_chunk_ids",
        "reranker",
        "settings",
        "stages",
        "total_ms",
        "warnings",
        "rewritten_query",
      ].sort(),
    );
  });

  it("validates the request body", async () => {
    const app = await appWith();
    await upload(app);
    expect((await post(app, "/api/query", { question: "" })).statusCode).toBe(422);
    expect(
      (await post(app, "/api/query", { question: "x", settings: { retrieval_mode: "magic" } }))
        .statusCode,
    ).toBe(422);
    expect(
      (await post(app, "/api/query", { question: "x", settings: { alpha: 2 } })).statusCode,
    ).toBe(422);
    expect((await post(app, "/api/query", { question: "x".repeat(2001) })).statusCode).toBe(422);
    const malformed = await app.inject({
      method: "POST",
      url: "/api/query",
      payload: "{not json",
      headers: { "content-type": "application/json" },
    });
    expect(malformed.statusCode).toBe(422);
    expect(malformed.json()).toHaveProperty("detail");
  });

  it("accepts the settings the frontend sends (nulls included)", async () => {
    const app = await appWith();
    await upload(app, { chunk_size: "30", chunk_overlap: "5" });
    const settings = {
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
    const res = await post(app, "/api/query", { question: "volcano magma", history: [], settings });
    expect(res.statusCode).toBe(200);
    expect(res.json().trace.settings).toEqual(settings);
  });

  it("streams trace, token and done events", async () => {
    const app = await appWith();
    await upload(app, { chunk_size: "30", chunk_overlap: "5" });
    const res = await post(app, "/api/query/stream", { question: "volcano magma" });
    expect(res.headers["content-type"]).toMatch(/^text\/event-stream/);
    expect(res.headers["cache-control"]).toBe("no-cache");
    const events = res.body.split("\n\n").filter((e) => e.trim());
    const parse = (e: string) => ({
      name: /^event: (.+)$/m.exec(e)?.[1],
      data: JSON.parse(/^data: (.+)$/m.exec(e)?.[1] as string),
    });
    const parsed = events.map(parse);
    expect(parsed[0]?.name).toBe("trace");
    expect(parsed.at(-1)?.name).toBe("done");
    const streamed = parsed
      .filter((e) => e.name === "token")
      .map((e) => e.data.text)
      .join("");
    expect(streamed).toBe(parsed.at(-1)?.data.answer);
    expect(parsed.at(-1)?.data.trace.stages.at(-1).skipped).toBe(false);
  });

  it("escapes line separators inside SSE payloads", () => {
    const out = formatSse({ event: "token", data: { text: "a b c" } });
    expect(out).toBe('event: token\ndata: {"text":"a\\u2028b\\u2029c"}\n\n');
  });
});

describe("evaluate", () => {
  it("compares the four configurations on the sample", async () => {
    const app = await appWith();
    await app.inject({ method: "POST", url: "/api/documents/sample" });
    const res = await post(app, "/api/evaluate", { ks: [5, 1, 3, 3], max_questions: 12 });
    expect(res.statusCode).toBe(200);
    const report = res.json();
    expect(report).toMatchObject({
      questions: 12,
      source: "rag_handbook_qa.json",
      ks: [1, 3, 5],
      precision_k: 5,
    });
    expect(report.rows.map((r: { config: string }) => r.config)).toEqual([
      "vector",
      "bm25",
      "hybrid",
      "hybrid + rerank",
    ]);
    expect(
      report.rows.every(
        (r: { recall: Record<string, number> }) => r.recall["@5"]! >= 0 && r.recall["@5"]! <= 1,
      ),
    ).toBe(true);
  });

  it("auto-generates questions for unknown PDFs", async () => {
    const app = await appWith();
    await upload(app, { chunk_size: "30", chunk_overlap: "5" });
    const report = (await post(app, "/api/evaluate", { ks: [1, 3], max_questions: 3 })).json();
    expect(report.source).toBe("auto-generated");
    expect(report.questions).toBeGreaterThanOrEqual(1);
  });

  it("returns 422 when no questions can be built", async () => {
    const app = await appWith();
    const tiny = await makePdf([["Tiny", "Too short."]]);
    await upload(app, { chunk_size: "30", chunk_overlap: "5" }, tiny);
    const res = await post(app, "/api/evaluate", {});
    expect(res.statusCode).toBe(422);
    expect(res.json().detail).toContain("Could not build evaluation questions");
  });

  it("validates ks and respects doc_ids", async () => {
    const app = await appWith();
    const doc = (await upload(app, { chunk_size: "30", chunk_overlap: "5" })).json();
    expect((await post(app, "/api/evaluate", { ks: [] })).statusCode).toBe(422);
    expect((await post(app, "/api/evaluate", { ks: [99] })).statusCode).toBe(422);
    const ok = await post(app, "/api/evaluate", { ks: [1], doc_ids: [doc.id], max_questions: 2 });
    expect(ok.statusCode).toBe(200);
  });
});

describe("serving the frontend", () => {
  it("serves assets and falls back to index.html for client routes", async () => {
    const staticDir = mkdtempSync(path.join(os.tmpdir(), "static-"));
    mkdirSync(path.join(staticDir, "assets"));
    writeFileSync(path.join(staticDir, "index.html"), "<html>app</html>");
    writeFileSync(path.join(staticDir, "assets", "app.js"), "console.log(1)");
    writeFileSync(path.join(staticDir, "favicon.svg"), "<svg/>");
    const app = await appWith({ staticDir });
    expect((await app.inject("/")).body).toContain("app");
    expect((await app.inject("/assets/app.js")).body).toBe("console.log(1)");
    expect((await app.inject("/favicon.svg")).body).toBe("<svg/>");
    const route = await app.inject("/some/client/route");
    expect(route.statusCode).toBe(200);
    expect(route.body).toContain("app");
    expect((await app.inject("/api/health")).statusCode).toBe(200);
    expect((await app.inject("/api/nope")).statusCode).toBe(404);
    expect((await app.inject("/%2e%2e/%2e%2e/etc/passwd")).body).toContain("app");
    expect((await app.inject({ method: "POST", url: "/elsewhere" })).statusCode).toBe(404);
  });

  it("returns JSON 404s when there is no frontend build", async () => {
    const app = await appWith();
    const res = await app.inject("/");
    expect(res.statusCode).toBe(404);
    expect(res.json()).toEqual({ detail: "Not Found" });
  });

  it("sends CORS headers for the configured origins", async () => {
    const app = await appWith({ corsOrigins: ["http://localhost:5173"] });
    const res = await app.inject({
      method: "OPTIONS",
      url: "/api/query",
      headers: { origin: "http://localhost:5173", "access-control-request-method": "POST" },
    });
    expect(res.headers["access-control-allow-origin"]).toBe("http://localhost:5173");
  });
});
