/**
 * Contextual compression: keep only the sentences of a chunk that matter for the query.
 *
 * Retrieved chunks often contain a relevant sentence surrounded by filler. Dropping the
 * filler cuts prompt tokens (cost and latency) and reduces the "lost in the middle" effect.
 * This implementation is extractive: sentences are kept when they share informative terms
 * with the query; at least the best sentence is always kept.
 */
import { contentTerms, pairwise, splitSentences } from "../text.js";

/** Fraction of query content terms present in each sentence. */
export function sentenceScores(query: string, sentences: string[]): number[] {
  const qTerms = new Set(contentTerms(query));
  if (qTerms.size === 0) return sentences.map(() => 0);
  return sentences.map((s) => {
    const sTerms = new Set(contentTerms(s));
    let shared = 0;
    for (const t of qTerms) if (sTerms.has(t)) shared++;
    return shared / qTerms.size;
  });
}

/** Fraction of the query's adjacent content-word pairs that also occur in `sentence`. */
export function bigramOverlap(query: string, sentence: string): number {
  const qPairs = new Set(pairwise(contentTerms(query)).map((p) => p.join("\u0000")));
  if (qPairs.size === 0) return 0;
  const sPairs = new Set(pairwise(contentTerms(sentence)).map((p) => p.join("\u0000")));
  let shared = 0;
  for (const p of qPairs) if (sPairs.has(p)) shared++;
  return shared / qPairs.size;
}

/** Return the query-relevant sentences of `text` in their original order. */
export function compressText(
  query: string,
  text: string,
  minScore = 0.15,
  maxSentences = 4,
): string {
  const sentences = splitSentences(text);
  if (sentences.length <= 1) return text;
  const scores = sentenceScores(query, sentences);
  const ranked = sentences
    .map((_, i) => i)
    .sort((a, b) => (scores[b] as number) - (scores[a] as number) || a - b);
  let keep = ranked.slice(0, maxSentences).filter((i) => (scores[i] as number) >= minScore);
  if (keep.length === 0) keep = [ranked[0] as number];
  return keep
    .sort((a, b) => a - b)
    .map((i) => sentences[i])
    .join(" ");
}
