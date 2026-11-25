/**
 * LLM abstraction: OpenAI-compatible APIs, Anthropic, and an offline extractive fallback.
 *
 * `openai_compatible` works with anything that speaks the OpenAI chat-completions protocol:
 * OpenAI, Groq, Google Gemini (OpenAI-compat endpoint), Ollama, vLLM, LM Studio...
 * When no provider is configured the {@link ExtractiveGenerator} composes an answer from the
 * most query-relevant sentences of the retrieved context, so the whole demo works offline.
 */
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import type { Settings } from "../config.js";
import { bigramOverlap, sentenceScores } from "../retrieval/compression.js";
import { splitSentences } from "../text.js";
import { NO_ANSWER, type ContextItem } from "./prompts.js";

/** Minimal chat-completion interface used by the pipeline. */
export interface LLMClient {
  name: string;
  /** Return the full completion. */
  complete(system: string, user: string, maxTokens?: number): Promise<string>;
  /** Yield completion text fragments as they arrive. */
  stream(system: string, user: string, maxTokens?: number): AsyncIterable<string>;
}

/** Chat client for any OpenAI-style endpoint. */
export class OpenAICompatibleClient implements LLMClient {
  private readonly client: OpenAI;

  constructor(
    apiKey: string,
    private readonly model: string,
    baseUrl: string | null,
    readonly name: string,
    client?: OpenAI,
  ) {
    this.client =
      client ?? new OpenAI({ apiKey: apiKey || "not-needed", baseURL: baseUrl || undefined });
  }

  private messages(system: string, user: string) {
    return [
      { role: "system" as const, content: system },
      { role: "user" as const, content: user },
    ];
  }

  /** Single non-streaming completion. */
  async complete(system: string, user: string, maxTokens = 600): Promise<string> {
    const resp = await this.client.chat.completions.create({
      model: this.model,
      messages: this.messages(system, user),
      max_tokens: maxTokens,
      temperature: 0.2,
    });
    return (resp.choices[0]?.message.content ?? "").trim();
  }

  /** Streaming completion. */
  async *stream(system: string, user: string, maxTokens = 800): AsyncGenerator<string> {
    const events = await this.client.chat.completions.create({
      model: this.model,
      messages: this.messages(system, user),
      max_tokens: maxTokens,
      temperature: 0.2,
      stream: true,
    });
    for await (const event of events) {
      const text = event.choices[0]?.delta.content;
      if (text) yield text;
    }
  }
}

/** Chat client for the Anthropic Messages API. */
export class AnthropicClient implements LLMClient {
  readonly name = "Anthropic";
  private readonly client: Anthropic;

  constructor(
    apiKey: string,
    private readonly model: string,
    client?: Anthropic,
  ) {
    this.client = client ?? new Anthropic({ apiKey });
  }

  /** Single non-streaming completion. */
  async complete(system: string, user: string, maxTokens = 600): Promise<string> {
    const resp = await this.client.messages.create({
      model: this.model,
      system,
      messages: [{ role: "user", content: user }],
      max_tokens: maxTokens,
    });
    return resp.content
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("")
      .trim();
  }

  /** Streaming completion. */
  async *stream(system: string, user: string, maxTokens = 800): AsyncGenerator<string> {
    const events = await this.client.messages.create({
      model: this.model,
      system,
      messages: [{ role: "user", content: user }],
      max_tokens: maxTokens,
      stream: true,
    });
    for await (const event of events) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        yield event.delta.text;
      }
    }
  }
}

/** Human-friendly provider name for the UI badge. */
export function providerLabel(settings: Pick<Settings, "llmProvider" | "llmBaseUrl">): string {
  if (settings.llmProvider === "none") return "Offline";
  if (settings.llmProvider === "anthropic") return "Anthropic";
  if (settings.llmProvider === "openai") return "OpenAI";
  const url = settings.llmBaseUrl.toLowerCase();
  const known: [string, string][] = [
    ["groq", "Groq"],
    ["generativelanguage", "Gemini"],
    ["localhost", "Ollama"],
    ["127.0.0.1", "Ollama"],
    ["openai.com", "OpenAI"],
  ];
  return known.find(([needle]) => url.includes(needle))?.[1] ?? "OpenAI-compatible";
}

/** Build the configured LLM client, or `null` to use the offline extractive generator. */
export function createLlm(settings: Settings): LLMClient | null {
  if (settings.llmProvider === "none") return null;
  if (settings.llmProvider === "anthropic") {
    return new AnthropicClient(settings.llmApiKey, settings.llmModel || "claude-3-5-haiku-latest");
  }
  const baseUrl =
    settings.llmBaseUrl || (settings.llmProvider === "openai" ? "https://api.openai.com/v1" : "");
  return new OpenAICompatibleClient(
    settings.llmApiKey,
    settings.llmModel || "gpt-4o-mini",
    baseUrl,
    providerLabel(settings),
  );
}

interface Candidate {
  score: number;
  index: number;
  pos: number;
  sentence: string;
}

/**
 * Offline "generator": answers by quoting the most query-relevant context sentences.
 *
 * It is deterministic, needs no network, and still produces cited, grounded answers; it
 * cannot paraphrase or reason across passages like a real LLM.
 */
export class ExtractiveGenerator {
  readonly name = "Offline (extractive)";

  constructor(
    private readonly maxSentences = 3,
    private readonly minScore = 0.3,
  ) {}

  /** Compose a cited answer, or the standard refusal when nothing is relevant. */
  answer(question: string, items: ContextItem[]): string {
    const candidates: Candidate[] = [];
    for (const item of items) {
      const sentences = splitSentences(item.text);
      const scores = sentenceScores(question, sentences);
      sentences.forEach((sentence, pos) => {
        const score = scores[pos] as number;
        if (score >= this.minScore) {
          // prefer passages the retriever ranked first, and phrase (bigram) matches
          const bonus = 0.3 * bigramOverlap(question, sentence) - 0.05 * item.index;
          candidates.push({ score: score + bonus, index: item.index, pos, sentence });
        }
      });
    }
    if (candidates.length === 0) return NO_ANSWER;
    const best = candidates.sort((a, b) => b.score - a.score).slice(0, this.maxSentences);
    best.sort((a, b) => a.index - b.index || a.pos - b.pos);
    return best.map((c) => `${c.sentence.trimEnd()} [${c.index}]`).join(" ");
  }

  /** Yield an already-composed answer word by word (to exercise the streaming path). */
  *stream(text: string): Generator<string> {
    yield* text.match(/\S+\s*/g) ?? [];
  }
}
