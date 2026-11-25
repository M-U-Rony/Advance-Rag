/**
 * BM25 keyword ranking implemented from scratch (Okapi BM25, Lucene-style IDF).
 *
 * BM25 scores a document `d` for query terms `q_i` as:
 *
 * ```
 * score(d, q) = sum_i  idf(q_i) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
 * idf(t)      = ln( (N - df + 0.5) / (df + 0.5) + 1 )
 * ```
 *
 * - `k1` (1.2) controls term-frequency saturation: the 10th occurrence of a word adds far
 *   less than the 1st.
 * - `b` (0.75) controls length normalisation: long documents are penalised so they cannot
 *   win merely by containing more words.
 *
 * Unlike embeddings BM25 needs exact word matches, which makes it excellent for rare
 * identifiers, names and numbers, and weak on paraphrases. Hybrid search combines both.
 */
import { tokenize } from "../text.js";

/** In-memory BM25 index over a list of texts. */
export class BM25Index {
  docLen = new Float64Array(0);
  avgdl = 0;
  nDocs = 0;
  idf = new Map<string, number>();
  /** term -> list of `[docIndex, termFrequency]` */
  postings = new Map<string, [number, number][]>();

  constructor(
    readonly k1 = 1.2,
    readonly b = 0.75,
  ) {}

  /** (Re)build the index from `texts`; document ids are their list positions. */
  index(texts: string[]): void {
    this.nDocs = texts.length;
    this.postings = new Map();
    const lengths: number[] = [];
    texts.forEach((text, docId) => {
      const tokens = tokenize(text, true);
      lengths.push(tokens.length);
      const tf = new Map<string, number>();
      for (const token of tokens) tf.set(token, (tf.get(token) ?? 0) + 1);
      for (const [term, count] of tf) {
        const list = this.postings.get(term);
        if (list) list.push([docId, count]);
        else this.postings.set(term, [[docId, count]]);
      }
    });
    this.docLen = Float64Array.from(lengths);
    this.avgdl = lengths.length > 0 ? lengths.reduce((a, c) => a + c, 0) / lengths.length : 0;
    this.idf = new Map(
      [...this.postings].map(([term, list]) => [
        term,
        Math.log((this.nDocs - list.length + 0.5) / (list.length + 0.5) + 1),
      ]),
    );
  }

  /** BM25 score of every indexed document for `query`. */
  score(query: string): Float64Array {
    const scores = new Float64Array(this.nDocs);
    if (this.nDocs === 0 || this.avgdl === 0) return scores;
    for (const term of new Set(tokenize(query, true))) {
      for (const [docId, tf] of this.postings.get(term) ?? []) {
        const norm =
          this.k1 * (1 - this.b + (this.b * (this.docLen[docId] as number)) / this.avgdl);
        scores[docId] =
          (scores[docId] as number) +
          ((this.idf.get(term) as number) * tf * (this.k1 + 1)) / (tf + norm);
      }
    }
    return scores;
  }

  /**
   * Top-`k` `[docIndex, score]` pairs with positive score.
   * `allowed` is an optional boolean mask implementing metadata pre-filtering.
   */
  search(query: string, k: number, allowed?: readonly boolean[] | null): [number, number][] {
    const scores = this.score(query);
    if (allowed) allowed.forEach((ok, i) => !ok && (scores[i] = 0));
    return topK(scores, k).filter(([, s]) => s > 0);
  }
}

/** Indices of the `k` highest scores, best first; ties keep the lower index first. */
export function topK(scores: ArrayLike<number>, k: number): [number, number][] {
  const order = Array.from({ length: scores.length }, (_, i) => i);
  order.sort((a, b) => (scores[b] as number) - (scores[a] as number) || a - b);
  return order.slice(0, Math.max(k, 0)).map((i) => [i, scores[i] as number]);
}
