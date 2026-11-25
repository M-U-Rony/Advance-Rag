/**
 * Prompt construction: turning retrieved context into a grounded prompt.
 *
 * Grounding means the LLM may only use the supplied context, every source is labelled
 * `[Source i]` so the answer can cite `[i]`, and an explicit refusal sentence is defined
 * for questions the context cannot answer (the main defence against hallucination).
 */
import { estimateTokens } from "../text.js";

export const NO_ANSWER =
  "I don't have enough information in the provided documents to answer that.";

export const SYSTEM_PROMPT =
  "You are a careful assistant that answers questions about the user's documents.\n" +
  "Rules:\n" +
  "1. Use ONLY the information inside the numbered context sources below. Do not use " +
  "outside knowledge.\n" +
  "2. Cite the sources that support each statement with bracketed numbers such as [1] or " +
  "[2][3].\n" +
  `3. If the context does not contain the answer, reply exactly: "${NO_ANSWER}"\n` +
  "4. Be concise and precise.";

export const CONDENSE_SYSTEM =
  "Rewrite the user's follow-up question as a single standalone question that can be " +
  "understood without the conversation. Resolve pronouns and references. Output only the " +
  "rewritten question.";

export const MAX_HISTORY_EXCHANGES = 3;

/** One context passage handed to the LLM. */
export interface ContextItem {
  index: number;
  filename: string;
  page: number;
  text: string;
}

/** One chat message. */
export interface Turn {
  role: string;
  content: string;
}

/** Keep the last `exchanges` user/assistant pairs (2 messages each). */
export function recentHistory(history: Turn[], exchanges = MAX_HISTORY_EXCHANGES): Turn[] {
  return history.slice(-2 * exchanges);
}

/** `"user"` -> `"User"`, used to label conversation turns. */
export function capitalize(role: string): string {
  return role.charAt(0).toUpperCase() + role.slice(1).toLowerCase();
}

/** Render labelled sources: `[Source 1] (file.pdf, p. 3)` followed by the passage. */
export function formatContext(items: ContextItem[]): string {
  return items
    .map((c) => `[Source ${c.index}] (${c.filename}, p. ${c.page})\n${c.text}`)
    .join("\n\n");
}

/** Return `[systemPrompt, userPrompt]` for the answer-generation call. */
export function buildGroundedPrompt(
  question: string,
  items: ContextItem[],
  history: Turn[] = [],
): [system: string, user: string] {
  const parts = [
    items.length > 0 ? `Context:\n${formatContext(items)}` : "Context:\n(no relevant passages)",
  ];
  const turns = recentHistory(history);
  if (turns.length > 0) {
    const convo = turns.map((t) => `${capitalize(t.role)}: ${t.content}`).join("\n");
    parts.push(`Conversation so far:\n${convo}`);
  }
  parts.push(`Question: ${question}\nAnswer:`);
  return [SYSTEM_PROMPT, parts.join("\n\n")];
}

/** Approximate token count of the full prompt. */
export function promptTokenEstimate(system: string, user: string): number {
  return estimateTokens(system) + estimateTokens(user);
}
