/**
 * TF-IDF embeddings implemented from scratch.
 *
 * TF-IDF is the classic *sparse* baseline: each dimension is a vocabulary term and its
 * value is how often the term occurs in the text (term frequency, TF) scaled by how rare
 * the term is across the corpus (inverse document frequency, IDF):
 *
 * ```
 * idf(t)   = ln((n + 1) / (df(t) + 1)) + 1        // smoothed, always > 0
 * tfidf    = tf(t, text) * idf(t)                 // then L2-normalised
 * ```
 *
 * It only matches *words*, not meaning ("car" will not match "automobile"), which is
 * precisely the gap dense embeddings close. It needs no model download, so it is also
 * the offline fallback for the whole system.
 */
import { tokenize } from "../text.js";
import type { Embedder } from "./types.js";

/** Corpus-fitted TF-IDF vectoriser (uni-grams, L2-normalised output). */
export class TfidfEmbedder implements Embedder {
  readonly name = "tfidf";
  readonly needsFit = true;
  vocab = new Map<string, number>();
  idf = new Float32Array(0);

  constructor(private readonly maxFeatures = 20000) {}

  /** Vocabulary size. */
  get dim(): number {
    return this.vocab.size;
  }

  /** Build the vocabulary and IDF weights from `texts`. */
  fit(texts: string[]): void {
    const n = texts.length;
    const df = new Map<string, number>();
    for (const text of texts) {
      for (const term of new Set(tokenize(text))) df.set(term, (df.get(term) ?? 0) + 1);
    }
    const terms = [...df.keys()]
      .sort((a, b) => (df.get(b) as number) - (df.get(a) as number) || compare(a, b))
      .slice(0, this.maxFeatures)
      .sort(compare);
    this.vocab = new Map(terms.map((t, i) => [t, i]));
    this.idf = Float32Array.from(terms, (t) => Math.log((n + 1) / ((df.get(t) as number) + 1)) + 1);
  }

  private vectorize(text: string): Float32Array {
    const vec = new Float32Array(this.vocab.size);
    const counts = new Map<string, number>();
    for (const token of tokenize(text)) counts.set(token, (counts.get(token) ?? 0) + 1);
    for (const [term, count] of counts) {
      const idx = this.vocab.get(term);
      if (idx !== undefined) vec[idx] = count * (this.idf[idx] as number);
    }
    let sumSquares = 0;
    for (const v of vec) sumSquares += v * v;
    const norm = Math.sqrt(sumSquares);
    if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] = (vec[i] as number) / norm;
    return vec;
  }

  /** Embed passages; every row has one entry per vocabulary term. */
  embedDocuments(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(texts.map((t) => this.vectorize(t)));
  }

  /** Embed a query with the fitted vocabulary (unknown terms are ignored). */
  embedQuery(text: string): Promise<Float32Array> {
    return Promise.resolve(this.vectorize(text));
  }
}

function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
