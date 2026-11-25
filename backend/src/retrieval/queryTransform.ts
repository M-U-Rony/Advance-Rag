/**
 * Query transformation: improve the query *before* searching.
 *
 * User questions are often a poor search key: they are conversational, ambiguous, or phrased
 * differently from the documents. Four techniques are implemented; each uses the LLM when one
 * is configured and a deterministic heuristic otherwise, so everything works offline.
 *
 * - **Condense with history** - turn a follow-up ("what about its cost?") into a standalone query.
 * - **Rewrite** - reformulate the question into a keyword-rich search query.
 * - **HyDE** (Hypothetical Document Embeddings) - have the LLM write a *hypothetical answer* and
 *   embed that instead of the question: an answer looks more like the documents than a question does.
 * - **Multi-query** - generate several paraphrases, retrieve for each, and fuse the lists with RRF.
 */
import type { LLMClient } from "../generation/llm.js";
import { CONDENSE_SYSTEM, capitalize, recentHistory, type Turn } from "../generation/prompts.js";
import { contentTerms } from "../text.js";

const QUESTION_WORDS =
  /^(what|which|who|whom|when|where|why|how|is|are|was|were|do|does|did|can|could|should)\b(\s+(?:is|are|was|were|does|did|do|the|a|an)\b)*\s*/i;
const FOLLOW_UP_HINT =
  /\b(it|its|they|them|their|this|that|these|those|he|she|there|former|latter|the same)\b/i;

/** Remove duplicates, keeping the first occurrence. */
function unique<T>(items: T[]): T[] {
  return [...new Set(items)];
}

/** Strip question scaffolding and stop words, leaving the topical keywords. */
export function keywordsOf(question: string): string {
  const stripped = question.trim().replace(/\?+$/, "").replace(QUESTION_WORDS, "");
  const terms = contentTerms(stripped);
  return unique(terms.length > 0 ? terms : contentTerms(question)).join(" ");
}

async function llmText(
  llm: LLMClient,
  system: string,
  user: string,
  maxTokens = 200,
): Promise<string | null> {
  try {
    const out = (await llm.complete(system, user, maxTokens)).trim();
    return out || null;
  } catch {
    return null;
  }
}

/** Make a follow-up question standalone using the last three exchanges. */
export async function condenseWithHistory(
  question: string,
  history: Turn[],
  llm: LLMClient | null,
): Promise<string> {
  const turns = recentHistory(history);
  if (turns.length === 0) return question;
  if (llm) {
    const convo = turns.map((t) => `${capitalize(t.role)}: ${t.content}`).join("\n");
    const rewritten = await llmText(llm, CONDENSE_SYSTEM, `${convo}\n\nFollow-up: ${question}`);
    if (rewritten) return rewritten.trim().replace(/^"+|"+$/g, "");
  }
  if (FOLLOW_UP_HINT.test(question) || contentTerms(question).length < 3) {
    const lastUser = [...turns].reverse().find((t) => t.role === "user")?.content ?? "";
    const topic = keywordsOf(lastUser);
    if (topic) return `${question.replace(/\?+$/, "").trim()} (${topic})`;
  }
  return question;
}

/** Rewrite a question into a concise, keyword-rich search query. */
export async function rewriteQuery(question: string, llm: LLMClient | null): Promise<string> {
  if (llm) {
    const out = await llmText(
      llm,
      "Rewrite the question as a concise search query with the most informative keywords " +
        "and synonyms. Output only the query.",
      question,
      60,
    );
    if (out) return (out.split("\n")[0] as string).trim().replace(/^"+|"+$/g, "");
  }
  return keywordsOf(question) || question;
}

/** Hypothetical Document Embeddings: a made-up answer used as the search vector. */
export async function hydeDocument(question: string, llm: LLMClient | null): Promise<string> {
  if (llm) {
    const out = await llmText(
      llm,
      "Write a short, factual passage (3-4 sentences) that would answer the question, " +
        "as it might appear in a technical document. Do not mention that it is hypothetical.",
      question,
      180,
    );
    if (out) return out;
  }
  const topic = keywordsOf(question) || question;
  const capitalised = topic.charAt(0).toUpperCase() + topic.slice(1).toLowerCase();
  return (
    `${capitalised} is described in the document. The main points about ${topic} ` +
    `are explained in detail, including how ${topic} works and why it matters.`
  );
}

/** Generate `n` alternative phrasings of `question` (original not included). */
export async function multiQuery(
  question: string,
  llm: LLMClient | null,
  n = 3,
): Promise<string[]> {
  if (llm) {
    const out = await llmText(
      llm,
      `Generate ${n} different search queries that capture other ways to ask the question. ` +
        "One per line, no numbering.",
      question,
      150,
    );
    if (out) {
      const variants = out
        .split("\n")
        .map((ln) => ln.replace(/^[\d.\-*)\s]+/, "").trim())
        .filter((ln) => ln && ln.toLowerCase() !== question.toLowerCase());
      if (variants.length > 0) return variants.slice(0, n);
    }
  }
  const topic = keywordsOf(question) || question;
  const terms = topic.split(/\s+/);
  const candidates = [
    topic,
    `definition and explanation of ${topic}`,
    `how does ${topic} work`,
    terms.length > 1 ? [...terms].reverse().join(" ") : `${topic} overview`,
  ];
  return unique(candidates)
    .filter((c) => c.toLowerCase() !== question.toLowerCase())
    .slice(0, n);
}
