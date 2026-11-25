/**
 * HTTP API routes (contract-compatible with the frontend in `frontend/src/lib/api.ts`).
 *
 * | Method | Path                    | Purpose                                              |
 * |--------|-------------------------|------------------------------------------------------|
 * | GET    | /api/health             | liveness probe                                       |
 * | GET    | /api/config             | active LLM / embedding / reranker backends           |
 * | GET    | /api/documents          | list indexed documents                               |
 * | POST   | /api/documents          | upload + index a PDF (multipart form)                |
 * | POST   | /api/documents/sample   | index the bundled sample PDF                         |
 * | DELETE | /api/documents/:id      | remove a document                                    |
 * | POST   | /api/query              | answer + citations + trace                           |
 * | POST   | /api/query/stream       | the same as server-sent events: trace, token*, done  |
 * | POST   | /api/evaluate           | recall@k comparison of the retrieval configurations  |
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { HttpError } from "../errors.js";
import { autoGenerateQa, evaluate, loadQa } from "../evaluation/evaluator.js";
import { providerLabel } from "../generation/llm.js";
import { STRATEGIES, validateChunkingParams, type ChunkingParams } from "../ingestion/chunking.js";
import type { RAGPipeline, StreamEvent } from "../pipeline.js";
import { evalRequestSchema, queryRequestSchema, type DocumentInfo } from "../schemas.js";
import { VERSION } from "../version.js";

export const SAMPLE_PDF = "rag_handbook.pdf";

/** Form fields of the upload endpoint (multipart values arrive as strings, hence `coerce`). */
const uploadFormSchema = z.object({
  strategy: z.enum(STRATEGIES).default("recursive"),
  chunk_size: z.coerce.number().int().min(10).max(2000).default(120),
  chunk_overlap: z.coerce.number().int().min(0).max(500).default(20),
  parent_size: z.coerce.number().int().min(20).max(4000).default(200),
  child_size: z.coerce.number().int().min(5).max(1000).default(50),
});

function requireDocuments(pipeline: RAGPipeline): void {
  if (pipeline.documents.size === 0) {
    throw new HttpError(409, "No documents indexed yet. Upload a PDF or load the sample.");
  }
}

/** Format one server-sent event. U+2028/9 are escaped because the browser parser treats them as line breaks. */
export function formatSse({ event, data }: StreamEvent): string {
  const json = JSON.stringify(data).replaceAll("\u2028", "\\u2028").replaceAll("\u2029", "\\u2029");
  return `event: ${event}\ndata: ${json}\n\n`;
}

/** Register every `/api/*` route on `app`. */
export function registerRoutes(app: FastifyInstance, pipeline: RAGPipeline): void {
  const info = (d: DocumentInfo): DocumentInfo => ({ ...d });

  app.get("/api/health", () => ({ status: "ok", version: VERSION }));

  app.get("/api/config", () => {
    const s = pipeline.settings;
    return {
      version: VERSION,
      llm: {
        provider: s.llmProvider,
        label: providerLabel(s),
        model: pipeline.llm ? s.llmModel || null : null,
        offline: pipeline.llm === null,
      },
      embeddings: {
        requested: s.embeddingBackend,
        active: pipeline.embedder.name,
        dense: !pipeline.embedder.needsFit,
        fallback_reason: pipeline.embedderFallbackReason,
      },
      reranker: {
        model: s.rerankerModel,
        cross_encoder_available: pipeline.crossEncoder.available,
      },
      sample_available: existsSync(path.join(s.samplesDir, SAMPLE_PDF)),
    };
  });

  app.get("/api/documents", () => [...pipeline.documents.values()].map(info));

  app.post("/api/documents", async (request, reply) => {
    if (!request.isMultipart()) {
      throw new z.ZodError([{ code: "custom", path: ["file"], message: "Field required" }]);
    }
    const fields: Record<string, string> = {};
    let file: { filename: string; data: Buffer } | null = null;
    for await (const part of request.parts()) {
      if (part.type === "file") {
        if (part.fieldname === "file") {
          file = { filename: part.filename, data: await part.toBuffer() };
        } else {
          await part.toBuffer(); // drain unexpected files
        }
      } else if (typeof part.value === "string") {
        fields[part.fieldname] = part.value;
      }
    }
    const form = uploadFormSchema.parse(fields);
    if (!file)
      throw new z.ZodError([{ code: "custom", path: ["file"], message: "Field required" }]);
    const params: ChunkingParams = { ...form };
    validateChunkingParams(params);
    const record = await pipeline.addDocument(file.filename || "document.pdf", file.data, params);
    return reply.code(201).send(info(record));
  });

  app.post("/api/documents/sample", async (_request, reply) => {
    const existing = pipeline.findByFilename(SAMPLE_PDF);
    if (existing) return reply.code(201).send(info(existing));
    const file = path.join(pipeline.settings.samplesDir, SAMPLE_PDF);
    if (!existsSync(file)) throw new HttpError(404, "Sample PDF is not available on this server");
    const record = await pipeline.addPdfFile(file, {
      strategy: "parent_child",
      chunk_size: 120,
      chunk_overlap: 20,
      parent_size: 200,
      child_size: 50,
    });
    return reply.code(201).send(info(record));
  });

  app.delete<{ Params: { docId: string } }>("/api/documents/:docId", async (request, reply) => {
    if (!(await pipeline.deleteDocument(request.params.docId))) {
      throw new HttpError(404, "Document not found");
    }
    return reply.code(204).send();
  });

  app.post("/api/query", async (request) => {
    const body = queryRequestSchema.parse(request.body);
    requireDocuments(pipeline);
    return pipeline.query(body);
  });

  app.post("/api/query/stream", (request, reply) => {
    const body = queryRequestSchema.parse(request.body);
    requireDocuments(pipeline);
    async function* events(): AsyncGenerator<string> {
      for await (const event of pipeline.queryStream(body)) yield formatSse(event);
    }
    return reply
      .header("Cache-Control", "no-cache")
      .header("X-Accel-Buffering", "no")
      .type("text/event-stream")
      .send(Readable.from(events()));
  });

  app.post("/api/evaluate", async (request) => {
    const body = evalRequestSchema.parse(request.body);
    requireDocuments(pipeline);
    const docs = [...pipeline.documents.values()].filter(
      (d) => !body.doc_ids?.length || body.doc_ids.includes(d.id),
    );
    const docIds = new Set(docs.map((d) => d.id));
    const chunks = pipeline.index.chunks.filter((c) => docIds.has(c.doc_id));
    const sampleQa = path.join(pipeline.settings.samplesDir, "rag_handbook_qa.json");
    let qa;
    let source: string;
    if (existsSync(sampleQa) && docs.some((d) => d.filename === SAMPLE_PDF)) {
      qa = loadQa(sampleQa);
      source = path.basename(sampleQa);
    } else {
      qa = autoGenerateQa(chunks, body.max_questions);
      source = "auto-generated";
    }
    if (qa.length === 0) {
      throw new HttpError(422, "Could not build evaluation questions from the indexed text");
    }
    return evaluate(pipeline, qa.slice(0, body.max_questions), {
      ks: [...new Set(body.ks)].sort((a, b) => a - b),
      docIds: docs.map((d) => d.id),
      source,
    });
  });
}
