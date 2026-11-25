/**
 * Reranking: a precise second pass over the retrieval candidates.
 *
 * First-stage retrieval uses a **bi-encoder** (or BM25): query and passage are encoded
 * *separately*, so passages can be pre-computed and searched in milliseconds, but the model
 * never sees query and passage together. A **cross-encoder** reads `[query, passage]` jointly
 * and outputs a relevance score, which is far more accurate but too slow to run on the whole
 * corpus. The standard recipe is therefore: retrieve ~30 candidates cheaply, rerank them with
 * the cross-encoder, keep the best 5.
 *
 * Two rerankers are provided: {@link CrossEncoderReranker} (Transformers.js ONNX model,
 * lazy-loaded) and {@link HeuristicReranker}, a dependency-free fallback using term overlap,
 * bigram matches and an early-position boost.
 */
import type { Settings } from "../config.js";
import { contentTerms, pairwise, tokenize } from "../text.js";

/** A candidate to rerank: `[chunk id, text]`. */
export type RerankItem = [id: string, text: string];

/** Scores `[id, text]` candidates against a query. */
export interface Reranker {
  name: string;
  /** Return all ids with relevance scores, best first. */
  rerank(query: string, items: RerankItem[]): Promise<[string, number][]>;
}

function bigramSet(tokens: string[]): Set<string> {
  return new Set(pairwise(tokens).map((p) => p.join(" ")));
}

function overlapCount<T>(a: Set<T>, b: Set<T>): number {
  let n = 0;
  for (const x of a) if (b.has(x)) n++;
  return n;
}

/** Lexical reranker: stop-word filtered term overlap + bigram matches + early position. */
export class HeuristicReranker implements Reranker {
  name = "heuristic";

  /** Relevance in roughly [0, 1] of `text` for `query`. */
  score(query: string, text: string): number {
    const qTerms = contentTerms(query);
    if (qTerms.length === 0) return 0;
    const docTokens = tokenize(text);
    const docTerms = new Set(docTokens);
    const uniqueQ = new Set(qTerms);
    const overlap = overlapCount(uniqueQ, docTerms) / uniqueQ.size;
    const qBigrams = bigramSet(tokenize(query));
    const bigramHits =
      qBigrams.size > 0 ? overlapCount(qBigrams, bigramSet(docTokens)) / qBigrams.size : 0;
    const head = new Set(docTokens.slice(0, Math.max(Math.floor(docTokens.length / 5), 10)));
    const early = overlapCount(uniqueQ, head) / uniqueQ.size;
    return 0.6 * overlap + 0.3 * bigramHits + 0.1 * early;
  }

  /** Score and sort candidates (ties keep the incoming order). */
  rerank(query: string, items: RerankItem[]): Promise<[string, number][]> {
    const scored = items.map(([id, text]): [string, number] => [id, this.score(query, text)]);
    return Promise.resolve(scored.sort((a, b) => b[1] - a[1]));
  }
}

/** Scores every passage against the query with a cross-encoder model. */
export type CrossEncoderScorer = (query: string, passages: string[]) => Promise<number[]>;

/** Loads the model and returns its scorer (rejects if the model cannot be loaded). */
export type ScorerLoader = () => Promise<CrossEncoderScorer>;

/** Transformers.js cross-encoder (`ms-marco-MiniLM-L-6-v2`) with a heuristic fallback. */
export class CrossEncoderReranker implements Reranker {
  name = "cross_encoder";
  private scorer: CrossEncoderScorer | null = null;
  private failed: string | null = null;
  private loading: Promise<CrossEncoderScorer | null> | null = null;
  private readonly fallback = new HeuristicReranker();

  constructor(private readonly load: ScorerLoader) {}

  /** Build a reranker that loads `settings.rerankerModel` from Hugging Face on first use. */
  static fromSettings(settings: Settings): CrossEncoderReranker {
    return new CrossEncoderReranker(() => loadTransformersScorer(settings));
  }

  /** `null` until first use, then whether the model loaded. */
  get available(): boolean | null {
    if (this.scorer) return true;
    return this.failed ? false : null;
  }

  /** Why the model could not be loaded, if it could not. */
  get failure(): string | null {
    return this.failed;
  }

  private ensureLoaded(): Promise<CrossEncoderScorer | null> {
    if (this.scorer || this.failed) return Promise.resolve(this.scorer);
    this.loading ??= this.load()
      .then((scorer) => (this.scorer = scorer))
      .catch((error: unknown) => {
        const err = error instanceof Error ? error : new Error(String(error));
        this.failed = `${err.name}: ${err.message.slice(0, 160)}`;
        console.warn(`Cross-encoder unavailable (${err.message}); using heuristic reranker`);
        return null;
      });
    return this.loading;
  }

  /** Cross-encode `(query, passage)` pairs; falls back to the heuristic if unavailable. */
  async rerank(query: string, items: RerankItem[]): Promise<[string, number][]> {
    const scorer = await this.ensureLoaded();
    if (!scorer || items.length === 0) {
      this.name = "heuristic (cross-encoder unavailable)";
      return this.fallback.rerank(query, items);
    }
    this.name = "cross_encoder";
    const scores = await scorer(
      query,
      items.map(([, text]) => text),
    );
    const scored = items.map(([id], i): [string, number] => [id, scores[i] as number]);
    return scored.sort((a, b) => b[1] - a[1]);
  }
}

/* v8 ignore start */
/** Load the Hugging Face sequence-classification model behind the cross-encoder. */
export async function loadTransformersScorer(settings: Settings): Promise<CrossEncoderScorer> {
  const { AutoTokenizer, AutoModelForSequenceClassification, env } =
    await import("@huggingface/transformers");
  env.cacheDir = settings.modelCacheDir;
  const tokenizer = await AutoTokenizer.from_pretrained(settings.rerankerModel);
  const model = await AutoModelForSequenceClassification.from_pretrained(settings.rerankerModel, {
    dtype: settings.modelDtype,
  });
  return async (query, passages) => {
    const scores: number[] = [];
    for (let i = 0; i < passages.length; i += 16) {
      const batch = passages.slice(i, i + 16);
      const inputs = tokenizer(new Array<string>(batch.length).fill(query), {
        text_pair: batch,
        padding: true,
        truncation: true,
      });
      const { logits } = (await model(inputs)) as { logits: { data: Float32Array } };
      scores.push(...Array.from(logits.data));
    }
    return scores;
  };
}
/* v8 ignore stop */
