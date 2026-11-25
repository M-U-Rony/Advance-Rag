/**
 * Shared text utilities: tokenisation, stop words, sentence splitting.
 *
 * Every lexical component (TF-IDF, BM25, the heuristic reranker, contextual
 * compression, the extractive generator) shares one tokenizer so that they all
 * agree on what a "term" is.
 */

const STOPWORD_TEXT = `
a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what when where which while who
whom why will with would you your tell explain describe give please
`;

/** Words that carry no topic information ("the", "of", "what"...). */
export const STOPWORDS: ReadonlySet<string> = new Set(STOPWORD_TEXT.split(/\s+/).filter(Boolean));

const TOKEN_RE = /[a-z0-9]+(?:[-'][a-z0-9]+)*/g;
const SENTENCE_RE = /(?<=[.!?])\s+(?=[A-Z0-9"'([])/;

/** Split on runs of whitespace, dropping empty pieces. */
export function words(text: string): string[] {
  return text.split(/\s+/).filter((w) => w.length > 0);
}

/** Lowercase `text` and split it into alphanumeric terms (keeping `k-means` and `it's` whole). */
export function tokenize(text: string, removeStopwords = false): string[] {
  const tokens = text.toLowerCase().match(TOKEN_RE) ?? [];
  return removeStopwords ? tokens.filter((t) => !STOPWORDS.has(t)) : tokens;
}

/** Tokens of `text` without stop words (the informative terms). */
export function contentTerms(text: string): string[] {
  return tokenize(text, true);
}

/** Split `text` into sentences using punctuation followed by a capital letter or digit. */
export function splitSentences(text: string): string[] {
  const flat = text.replace(/\s+/g, " ").trim();
  if (!flat) return [];
  return flat
    .split(SENTENCE_RE)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** Number of whitespace separated words. */
export function wordCount(text: string): number {
  return words(text).length;
}

/** Rough LLM token estimate (about 1.3 tokens per word for English). */
export function estimateTokens(text: string): number {
  return Math.trunc(wordCount(text) * 1.3) + 1;
}

/** Single-line preview of `text` truncated to `limit` characters. */
export function snippet(text: string, limit = 220): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length <= limit ? flat : flat.slice(0, limit - 1).trimEnd() + "…";
}

/** Round to `digits` decimal places. */
export function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/** Consecutive pairs of an array: `[a, b, c]` -> `[[a, b], [b, c]]`. */
export function pairwise<T>(items: readonly T[]): [T, T][] {
  const pairs: [T, T][] = [];
  for (let i = 0; i + 1 < items.length; i++) pairs.push([items[i] as T, items[i + 1] as T]);
  return pairs;
}
