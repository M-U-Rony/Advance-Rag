/** Application settings, loaded from environment variables (or a `.env` file) and validated with zod. */
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

/** Repository root: `backend/src` and `backend/dist` are both two levels below it. */
export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

/** Models exported for Transformers.js live under `Xenova/`; the old `BAAI/` id is mapped to it. */
const DEFAULT_EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5";

const settingsSchema = z.object({
  LLM_PROVIDER: z.enum(["none", "openai", "openai_compatible", "anthropic"]).default("none"),
  LLM_API_KEY: z.string().default(""),
  LLM_MODEL: z.string().default(""),
  LLM_BASE_URL: z.string().default(""),
  EMBEDDING_BACKEND: z.enum(["auto", "dense", "tfidf"]).default("auto"),
  EMBEDDING_MODEL: z.string().default(DEFAULT_EMBEDDING_MODEL),
  RERANKER_MODEL: z.string().default("Xenova/ms-marco-MiniLM-L-6-v2"),
  /** ONNX weight precision: `fp32` is exact, `q8` is ~4x smaller and faster. */
  MODEL_DTYPE: z.enum(["fp32", "fp16", "q8"]).default("fp32"),
  /** Where Transformers.js caches downloaded models (default: `<DATA_DIR>/models`). */
  MODEL_CACHE_DIR: z.string().default(""),
  DATA_DIR: z.string().default("./data"),
  STATIC_DIR: z.string().default(path.join(REPO_ROOT, "backend", "static")),
  SAMPLES_DIR: z.string().default(path.join(REPO_ROOT, "samples")),
  CORS_ORIGINS: z.string().default("http://localhost:5173,http://127.0.0.1:5173"),
  MAX_UPLOAD_MB: z.coerce.number().int().min(0).default(50),
  PORT: z.coerce.number().int().min(1).max(65535).default(8000),
  HOST: z.string().default("0.0.0.0"),
});

/** Validated runtime configuration (a camelCase view of the environment variables). */
export interface Settings {
  llmProvider: "none" | "openai" | "openai_compatible" | "anthropic";
  llmApiKey: string;
  llmModel: string;
  llmBaseUrl: string;
  embeddingBackend: "auto" | "dense" | "tfidf";
  embeddingModel: string;
  rerankerModel: string;
  modelDtype: "fp32" | "fp16" | "q8";
  modelCacheDir: string;
  dataDir: string;
  staticDir: string;
  samplesDir: string;
  corsOrigins: string[];
  maxUploadMb: number;
  port: number;
  host: string;
}

/**
 * Build {@link Settings} from an environment-like object. Empty values count as "unset" so a
 * line such as `LLM_MODEL=` in `.env.example` falls back to the default.
 * Pass `overrides` in tests to point at temporary directories.
 */
export function loadSettings(
  env: Record<string, string | undefined> = process.env,
  overrides: Partial<Settings> = {},
): Settings {
  const cleaned = Object.fromEntries(
    Object.entries(env).filter(([, v]) => v !== undefined && v !== ""),
  );
  const e = settingsSchema.parse(cleaned);
  const dataDir = path.resolve(e.DATA_DIR);
  const settings: Settings = {
    llmProvider: e.LLM_PROVIDER,
    llmApiKey: e.LLM_API_KEY,
    llmModel: e.LLM_MODEL,
    llmBaseUrl: e.LLM_BASE_URL,
    embeddingBackend: e.EMBEDDING_BACKEND,
    embeddingModel: e.EMBEDDING_MODEL.replace(/^BAAI\//, "Xenova/"),
    rerankerModel: e.RERANKER_MODEL,
    modelDtype: e.MODEL_DTYPE,
    modelCacheDir: e.MODEL_CACHE_DIR
      ? path.resolve(e.MODEL_CACHE_DIR)
      : path.join(dataDir, "models"),
    dataDir,
    staticDir: path.resolve(e.STATIC_DIR),
    samplesDir: path.resolve(e.SAMPLES_DIR),
    corsOrigins: e.CORS_ORIGINS.split(",")
      .map((o) => o.trim())
      .filter((o) => o.length > 0),
    maxUploadMb: e.MAX_UPLOAD_MB,
    port: e.PORT,
    host: e.HOST,
  };
  return { ...settings, ...overrides };
}

/** Load `.env` from the working directory or its parent (the repo root) if present. */
export function loadDotEnv(): void {
  for (const candidate of [".env", path.join("..", ".env")]) {
    if (existsSync(candidate)) {
      process.loadEnvFile(candidate);
      return;
    }
  }
}
