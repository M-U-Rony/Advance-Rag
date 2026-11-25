import path from "node:path";
import { describe, expect, it } from "vitest";
import { REPO_ROOT, loadSettings } from "../src/config.js";

describe("loadSettings", () => {
  it("has offline-friendly defaults", () => {
    const s = loadSettings({});
    expect(s).toMatchObject({
      llmProvider: "none",
      embeddingBackend: "auto",
      embeddingModel: "Xenova/bge-small-en-v1.5",
      rerankerModel: "Xenova/ms-marco-MiniLM-L-6-v2",
      maxUploadMb: 50,
      port: 8000,
      modelDtype: "fp32",
    });
    expect(s.corsOrigins).toEqual(["http://localhost:5173", "http://127.0.0.1:5173"]);
    expect(s.samplesDir).toBe(path.join(REPO_ROOT, "samples"));
    expect(s.staticDir).toBe(path.join(REPO_ROOT, "backend", "static"));
    expect(s.modelCacheDir).toBe(path.join(s.dataDir, "models"));
  });

  it("reads the documented environment variables", () => {
    const s = loadSettings({
      LLM_PROVIDER: "anthropic",
      LLM_API_KEY: "k",
      LLM_MODEL: "m",
      LLM_BASE_URL: "http://x",
      EMBEDDING_BACKEND: "dense",
      EMBEDDING_MODEL: "BAAI/bge-small-en-v1.5",
      DATA_DIR: "/tmp/rag",
      CORS_ORIGINS: " http://a , ,http://b ",
      MAX_UPLOAD_MB: "7",
      MODEL_CACHE_DIR: "/tmp/models",
    });
    expect(s).toMatchObject({
      llmProvider: "anthropic",
      llmApiKey: "k",
      llmModel: "m",
      llmBaseUrl: "http://x",
      embeddingBackend: "dense",
      embeddingModel: "Xenova/bge-small-en-v1.5",
      dataDir: "/tmp/rag",
      maxUploadMb: 7,
      modelCacheDir: "/tmp/models",
    });
    expect(s.corsOrigins).toEqual(["http://a", "http://b"]);
  });

  it("treats empty values as unset (as in .env.example)", () => {
    const s = loadSettings({ LLM_MODEL: "", EMBEDDING_BACKEND: "", MAX_UPLOAD_MB: "" });
    expect(s.llmModel).toBe("");
    expect(s.embeddingBackend).toBe("auto");
    expect(s.maxUploadMb).toBe(50);
  });

  it("rejects invalid values", () => {
    expect(() => loadSettings({ LLM_PROVIDER: "skynet" })).toThrow();
    expect(() => loadSettings({ MAX_UPLOAD_MB: "lots" })).toThrow();
  });

  it("applies overrides last", () => {
    expect(loadSettings({}, { port: 1234 }).port).toBe(1234);
  });
});

describe("loadDotEnv", () => {
  it("loads a .env file from the working directory without overriding real variables", async () => {
    const { mkdtempSync, writeFileSync } = await import("node:fs");
    const os = await import("node:os");
    const { loadDotEnv } = await import("../src/config.js");
    const dir = mkdtempSync(path.join(os.tmpdir(), "dotenv-"));
    writeFileSync(
      path.join(dir, ".env"),
      "ADVANCE_RAG_TEST_A=from-file\nADVANCE_RAG_TEST_B=from-file\n",
    );
    const cwd = process.cwd();
    process.env.ADVANCE_RAG_TEST_B = "from-env";
    try {
      process.chdir(dir);
      loadDotEnv();
      expect(process.env.ADVANCE_RAG_TEST_A).toBe("from-file");
      expect(process.env.ADVANCE_RAG_TEST_B).toBe("from-env");
      process.chdir(os.tmpdir());
      loadDotEnv(); // no .env anywhere: must not throw
    } finally {
      process.chdir(cwd);
      delete process.env.ADVANCE_RAG_TEST_A;
      delete process.env.ADVANCE_RAG_TEST_B;
    }
  });
});
