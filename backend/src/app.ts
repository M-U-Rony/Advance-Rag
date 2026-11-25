/** Fastify application factory; also serves the built React frontend. */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyError, type FastifyInstance } from "fastify";
import { ZodError } from "zod";
import { registerRoutes } from "./api/routes.js";
import type { Settings } from "./config.js";
import { HttpError, InvalidInputError } from "./errors.js";
import type { RAGPipeline } from "./pipeline.js";
import { buildPipeline } from "./runtime.js";

/** FastAPI-style validation error entries: `{ type, loc, msg }`. */
function zodDetail(error: ZodError) {
  return error.issues.map((issue) => ({
    type: issue.code,
    loc: ["body", ...issue.path],
    msg: issue.message,
  }));
}

/**
 * Build the Fastify app. Pass `pipeline` in tests to reuse a prepared one; otherwise it is
 * built from `settings`. Errors are reported as `{ "detail": ... }`, like the original API.
 */
export async function createApp(
  settings: Settings,
  pipeline?: RAGPipeline,
): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });
  const rag = pipeline ?? (await buildPipeline(settings));

  await app.register(cors, {
    origin: settings.corsOrigins,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  });
  await app.register(multipart, {
    // busboy treats 0 as "no limit"; use at least 1 byte so a 0 MB limit rejects everything.
    limits: { fileSize: Math.max(settings.maxUploadMb * 1024 * 1024, 1), files: 5 },
  });

  app.setErrorHandler((error: FastifyError | Error, _request, reply) => {
    if (error instanceof ZodError) return reply.code(422).send({ detail: zodDetail(error) });
    if (error instanceof HttpError) return reply.code(error.status).send({ detail: error.detail });
    if (error instanceof InvalidInputError) return reply.code(422).send({ detail: error.message });
    const code = (error as FastifyError).code;
    if (code === "FST_REQ_FILE_TOO_LARGE") {
      return reply.code(413).send({ detail: `File exceeds ${settings.maxUploadMb} MB limit` });
    }
    const status = (error as FastifyError).statusCode ?? 500;
    if (status >= 500) {
      app.log.error(error);
      return reply.code(500).send({ detail: "Internal Server Error" });
    }
    // Malformed JSON / missing multipart body etc.: FastAPI reports these as 422.
    return reply.code(status === 400 ? 422 : status).send({ detail: error.message });
  });

  registerRoutes(app, rag);

  const index = path.join(settings.staticDir, "index.html");
  if (existsSync(index)) {
    const indexHtml = readFileSync(index);
    await app.register(fastifyStatic, { root: settings.staticDir });
    app.setNotFoundHandler((request, reply) => {
      if (request.method !== "GET" || request.url.startsWith("/api/")) {
        return reply.code(404).send({ detail: "Not Found" });
      }
      // Single-page app: unknown paths are client-side routes.
      return reply.type("text/html; charset=utf-8").send(indexHtml);
    });
  } else {
    app.setNotFoundHandler((_request, reply) => reply.code(404).send({ detail: "Not Found" }));
  }
  return app;
}
