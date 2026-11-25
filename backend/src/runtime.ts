/** Application container: builds the pipeline from settings. */
import type { Settings } from "./config.js";
import { createEmbedder } from "./embeddings/dense.js";
import { createLlm } from "./generation/llm.js";
import { RAGPipeline } from "./pipeline.js";
import { CrossEncoderReranker } from "./retrieval/rerank.js";

/** Replaceable parts of the pipeline (tests inject a cross-encoder that never touches the network). */
export interface PipelineParts {
  crossEncoder?: CrossEncoderReranker;
}

/** Create embedder, LLM client and pipeline according to `settings`. */
export async function buildPipeline(
  settings: Settings,
  parts: PipelineParts = {},
): Promise<RAGPipeline> {
  const { embedder, fallbackReason } = await createEmbedder(settings);
  const pipeline = new RAGPipeline(
    settings,
    embedder,
    createLlm(settings),
    parts.crossEncoder ?? CrossEncoderReranker.fromSettings(settings),
    fallbackReason,
  );
  return pipeline.init();
}
