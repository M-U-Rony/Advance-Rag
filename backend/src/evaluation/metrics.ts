/**
 * Evaluation metrics for retrieval and generation.
 *
 * Retrieval metrics compare the ranked chunk ids returned by a pipeline against the set of
 * chunk ids known to contain the answer:
 *
 * - **Recall@k**    - did at least one relevant chunk appear in the top k? (hit-rate form)
 * - **Precision@k** - what fraction of the top k are relevant?
 * - **MRR**         - mean of 1/rank of the first relevant chunk.
 *
 * Generation metrics (cheap, deterministic approximations of LLM-judge metrics):
 *
 * - **Faithfulness**     - fraction of answer claims (sentences) whose content words overlap the
 *   retrieved context by at least 50%: a low score signals hallucination.
 * - **Answer relevance** - share of question keywords that the answer addresses.
 *
 * An optional LLM judge ({@link llmFaithfulness}) can replace the overlap heuristic.
 */
import type { LLMClient } from "../generation/llm.js";
import { NO_ANSWER } from "../generation/prompts.js";
import { contentTerms, splitSentences } from "../text.js";

export const FAITHFULNESS_THRESHOLD = 0.5;

/** 1.0 if any of the top-`k` retrieved ids is relevant, else 0.0. */
export function recallAtK(retrieved: string[], relevant: ReadonlySet<string>, k: number): number {
  return relevant.size > 0 && retrieved.slice(0, k).some((r) => relevant.has(r)) ? 1 : 0;
}

/** Fraction of the top-`k` retrieved ids that are relevant. */
export function precisionAtK(
  retrieved: string[],
  relevant: ReadonlySet<string>,
  k: number,
): number {
  const top = retrieved.slice(0, k);
  return top.length > 0 ? top.filter((r) => relevant.has(r)).length / top.length : 0;
}

/** `1 / rank` of the first relevant id (0.0 if none retrieved). */
export function reciprocalRank(retrieved: string[], relevant: ReadonlySet<string>): number {
  const i = retrieved.findIndex((r) => relevant.has(r));
  return i === -1 ? 0 : 1 / (i + 1);
}

/** Fraction of answer sentences supported by the context (word overlap >= `threshold`). */
export function faithfulnessOverlap(
  answer: string,
  contexts: string[],
  threshold = FAITHFULNESS_THRESHOLD,
): number {
  if (answer.trim() === NO_ANSWER) return 1;
  const supportTerms = new Set(contexts.flatMap((c) => contentTerms(c)));
  const claims = splitSentences(answer).filter((s) => contentTerms(s).length > 0);
  if (claims.length === 0) return 0;
  const supported = claims.filter((claim) => {
    const terms = new Set(contentTerms(claim));
    let shared = 0;
    for (const t of terms) if (supportTerms.has(t)) shared++;
    return shared / terms.size >= threshold;
  });
  return supported.length / claims.length;
}

/** Share of the question's content words that also appear in the answer. */
export function answerRelevance(question: string, answer: string): number {
  const qTerms = new Set(contentTerms(question));
  if (qTerms.size === 0 || answer.trim() === NO_ANSWER) return 0;
  const aTerms = new Set(contentTerms(answer));
  let shared = 0;
  for (const t of qTerms) if (aTerms.has(t)) shared++;
  return shared / qTerms.size;
}

/** Optional LLM judge: ask the model to rate how well the context supports the answer (0-1). */
export async function llmFaithfulness(
  answer: string,
  contexts: string[],
  llm: LLMClient,
): Promise<number> {
  const reply = await llm.complete(
    "You are a strict evaluator. Reply with a single number between 0 and 1: the fraction of " +
      "claims in the ANSWER that are supported by the CONTEXT.",
    "CONTEXT:\n" + contexts.join("\n---\n") + `\n\nANSWER:\n${answer}`,
    8,
  );
  const value = Number.parseFloat(reply.trim().split(/\s+/)[0] ?? "");
  return Number.isNaN(value) ? 0 : Math.min(Math.max(value, 0), 1);
}
