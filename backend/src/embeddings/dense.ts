/**
 * Dense neural embeddings via Transformers.js (ONNX runtime, no PyTorch needed).
 *
 * A dense *bi-encoder* (default `Xenova/bge-small-en-v1.5`, 384 dimensions) encodes
 * queries and passages independently into the same semantic space, so "automobile" can
 * match "car". The model is downloaded lazily on first use; if it is unavailable
 * (offline machine) {@link createEmbedder} transparently falls back to TF-IDF.
 */
import type { Settings } from "../config.js";
import { TfidfEmbedder } from "./tfidf.js";
import type { Embedder } from "./types.js";

const BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const BATCH_SIZE = 16;

/** Turns texts into one vector per text; the only part that touches the neural network. */
export type Extractor = (texts: string[]) => Promise<Float32Array[]>;

/** Wrapper around a feature-extraction model (or any {@link Extractor}). */
export class DenseEmbedder implements Embedder {
  readonly needsFit = false;
  readonly name: string;
  private readonly queryPrefix: string;

  constructor(
    modelName: string,
    private readonly extract: Extractor,
  ) {
    this.name = modelName.split("/").pop() as string;
    this.queryPrefix = modelName.toLowerCase().includes("bge") ? BGE_QUERY_PREFIX : "";
  }

  /** Download (first run) and load the ONNX model. Throws if it cannot be loaded. */
  /* v8 ignore start */
  static async load(settings: Settings): Promise<DenseEmbedder> {
    const { pipeline, env } = await import("@huggingface/transformers");
    env.cacheDir = settings.modelCacheDir;
    const extractor = await pipeline("feature-extraction", settings.embeddingModel, {
      dtype: settings.modelDtype,
    });
    const extract: Extractor = async (texts) => {
      const rows: Float32Array[] = [];
      for (let i = 0; i < texts.length; i += BATCH_SIZE) {
        // BGE models use the [CLS] token as the sentence vector; vectors are L2-normalised.
        const output = await extractor(texts.slice(i, i + BATCH_SIZE), {
          pooling: "cls",
          normalize: true,
        });
        const dim = output.dims[output.dims.length - 1] as number;
        const data = output.data as Float32Array;
        for (let r = 0; r < data.length / dim; r++) rows.push(data.slice(r * dim, (r + 1) * dim));
      }
      return rows;
    };
    return new DenseEmbedder(settings.embeddingModel, extract);
  }
  /* v8 ignore stop */

  /** No-op: dense models are pre-trained. */
  fit(): void {}

  /** Embed passages, one row per text. */
  embedDocuments(texts: string[]): Promise<Float32Array[]> {
    return texts.length === 0 ? Promise.resolve([]) : this.extract(texts);
  }

  /** Embed a query (with the BGE retrieval instruction prefix). */
  async embedQuery(text: string): Promise<Float32Array> {
    const [row] = await this.extract([this.queryPrefix + text]);
    return row as Float32Array;
  }
}

/** Result of {@link createEmbedder}: `fallbackReason` is null unless dense was requested but failed. */
export interface EmbedderChoice {
  embedder: Embedder;
  fallbackReason: string | null;
}

/**
 * Build the configured embedder. `load` is injectable so tests can simulate a missing model.
 */
export async function createEmbedder(
  settings: Settings,
  load: (s: Settings) => Promise<Embedder> = (s) => DenseEmbedder.load(s),
): Promise<EmbedderChoice> {
  if (settings.embeddingBackend === "tfidf") {
    return { embedder: new TfidfEmbedder(), fallbackReason: null };
  }
  try {
    return { embedder: await load(settings), fallbackReason: null };
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    const reason = `${err.name}: ${err.message.slice(0, 160)}`;
    console.warn(`Dense embedder unavailable (${reason}); falling back to TF-IDF`);
    return { embedder: new TfidfEmbedder(), fallbackReason: reason };
  }
}
