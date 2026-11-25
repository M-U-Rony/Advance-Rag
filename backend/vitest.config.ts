import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    testTimeout: 30_000,
    env: { EMBEDDING_BACKEND: "tfidf", LLM_PROVIDER: "none" },
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      // The CLI entry points only wire things together and are exercised manually / in CI smoke runs.
      exclude: ["src/server.ts", "src/evaluation/runEval.ts", "src/downloadModels.ts"],
      reporter: ["text", "lcov", "json-summary"],
    },
  },
});
