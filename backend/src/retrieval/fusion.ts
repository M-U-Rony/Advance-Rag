/**
 * Fusing several ranked lists into one: the heart of hybrid search.
 *
 * Vector scores (cosine in [-1, 1]) and BM25 scores (unbounded) live on different scales, so
 * they cannot simply be added. Two standard solutions are implemented:
 *
 * - **Weighted (alpha)**: min-max normalise each score list to [0, 1] then blend
 *   `alpha * vector + (1 - alpha) * keyword` (alpha=1 pure vector, alpha=0 pure keyword).
 * - **Reciprocal Rank Fusion (RRF)**: ignore scores, use only ranks:
 *   `score(d) = sum over lists of 1 / (k + rank(d))` with k=60. Robust, parameter-light and
 *   the default in most production systems.
 */

export const RRF_K = 60;

/** Fuse ranked id lists (best first) with RRF; returns `[id, score]` best first. */
export function reciprocalRankFusion(rankings: string[][], k = RRF_K): [string, number][] {
  const scores = new Map<string, number>();
  for (const ranking of rankings) {
    ranking.forEach((item, i) => {
      scores.set(item, (scores.get(item) ?? 0) + 1 / (k + i + 1));
    });
  }
  return [...scores].sort((a, b) => b[1] - a[1]);
}

/** Scale scores to [0, 1]; a constant list maps to 1.0 (or 0.0 when all are zero). */
export function minMaxNormalize(scores: ReadonlyMap<string, number>): Map<string, number> {
  if (scores.size === 0) return new Map();
  const values = [...scores.values()];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (hi === lo) return new Map([...scores.keys()].map((key) => [key, hi > 0 ? 1 : 0]));
  return new Map([...scores].map(([key, value]) => [key, (value - lo) / (hi - lo)]));
}

/** Blend normalised vector and keyword scores: `alpha*vector + (1-alpha)*keyword`. */
export function weightedFusion(
  vectorScores: ReadonlyMap<string, number>,
  keywordScores: ReadonlyMap<string, number>,
  alpha: number,
): [string, number][] {
  if (!(alpha >= 0 && alpha <= 1)) throw new RangeError("alpha must be between 0 and 1");
  const vec = minMaxNormalize(vectorScores);
  const kw = minMaxNormalize(keywordScores);
  const keys = new Set([...vec.keys(), ...kw.keys()]);
  const fused: [string, number][] = [...keys].map((key) => [
    key,
    alpha * (vec.get(key) ?? 0) + (1 - alpha) * (kw.get(key) ?? 0),
  ]);
  return fused.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}
