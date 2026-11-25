/**
 * Pre-download the dense embedding model and the cross-encoder into `MODEL_CACHE_DIR`.
 *
 * Used by the Dockerfile (`node dist/downloadModels.js`) and available locally as
 * `npm run models`. It never fails: without network the app simply falls back to TF-IDF and
 * the heuristic reranker, so a failed download must not break an image build.
 */
import { loadDotEnv, loadSettings } from "./config.js";
import { DenseEmbedder } from "./embeddings/dense.js";
import { loadTransformersScorer } from "./retrieval/rerank.js";

loadDotEnv();
const settings = loadSettings();
try {
  await DenseEmbedder.load(settings);
  await loadTransformersScorer(settings);
  console.log(`models cached in ${settings.modelCacheDir}`);
} catch (error) {
  console.log("model pre-download skipped:", error instanceof Error ? error.message : error);
}
