import { describe, expect, it, vi } from "vitest";
import { loadSettings } from "../src/config.js";
import {
  AnthropicClient,
  createLlm,
  ExtractiveGenerator,
  OpenAICompatibleClient,
  providerLabel,
} from "../src/generation/llm.js";
import {
  NO_ANSWER,
  buildGroundedPrompt,
  formatContext,
  promptTokenEstimate,
  recentHistory,
  type ContextItem,
  type Turn,
} from "../src/generation/prompts.js";
import {
  condenseWithHistory,
  hydeDocument,
  keywordsOf,
  multiQuery,
  rewriteQuery,
} from "../src/retrieval/queryTransform.js";
import { FakeLLM } from "./helpers.js";

const ITEMS: ContextItem[] = [
  {
    index: 1,
    filename: "a.pdf",
    page: 2,
    text: "RRF uses the constant k equal to 60. It ignores raw scores.",
  },
  { index: 2, filename: "b.pdf", page: 7, text: "Bananas are yellow." },
];

describe("prompts", () => {
  it("labels sources", () => {
    const rendered = formatContext(ITEMS);
    expect(rendered.startsWith("[Source 1] (a.pdf, p. 2)\nRRF uses")).toBe(true);
    expect(rendered).toContain("[Source 2] (b.pdf, p. 7)");
  });

  it("contains the rules and the question", () => {
    const [system, user] = buildGroundedPrompt("What is k?", ITEMS);
    expect(system).toContain("ONLY");
    expect(system).toContain(NO_ANSWER);
    expect(system).toContain("[1]");
    expect(user).toContain("Question: What is k?");
    expect(user).toContain("[Source 1]");
    expect(promptTokenEstimate(system, user)).toBeGreaterThan(20);
  });

  it("handles empty context and history", () => {
    const [, user] = buildGroundedPrompt(
      "q",
      [],
      [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ],
    );
    expect(user).toContain("(no relevant passages)");
    expect(user).toContain("User: hi");
    expect(user).toContain("Assistant: hello");
  });

  it("keeps the last three exchanges", () => {
    const turns: Turn[] = Array.from({ length: 10 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: String(i),
    }));
    expect(recentHistory(turns).map((t) => t.content)).toEqual(["4", "5", "6", "7", "8", "9"]);
  });
});

describe("ExtractiveGenerator", () => {
  it("cites the relevant sentence", () => {
    const answer = new ExtractiveGenerator().answer("What constant does RRF use?", ITEMS);
    expect(answer).toContain("constant k equal to 60");
    expect(answer.endsWith("[1]")).toBe(true);
    expect(answer).not.toContain("Bananas");
  });

  it("refuses when nothing is relevant", () => {
    expect(new ExtractiveGenerator().answer("Who won the football match?", ITEMS)).toBe(NO_ANSWER);
  });

  it("prefers phrase matches and earlier passages, and orders the output by source", () => {
    const items: ContextItem[] = [
      {
        index: 1,
        filename: "a",
        page: 1,
        text: "Rank fusion merges lists. Fusion of rank is nice.",
      },
      { index: 2, filename: "b", page: 1, text: "Rank fusion merges lists." },
    ];
    const answer = new ExtractiveGenerator(2).answer("rank fusion merges", items);
    expect(answer).toBe("Rank fusion merges lists. [1] Rank fusion merges lists. [2]");
  });

  it("streams word by word and reassembles", () => {
    const gen = new ExtractiveGenerator();
    const text = "Alpha beta gamma [1].";
    expect([...gen.stream(text)]).toEqual(["Alpha ", "beta ", "gamma ", "[1]."]);
    expect([...gen.stream(text)].join("")).toBe(text);
    expect([...gen.stream("")]).toEqual([]);
  });
});

describe("providers", () => {
  it.each([
    ["none", "", "Offline"],
    ["anthropic", "", "Anthropic"],
    ["openai", "", "OpenAI"],
    ["openai_compatible", "https://api.groq.com/openai/v1", "Groq"],
    ["openai_compatible", "https://generativelanguage.googleapis.com/v1beta/openai", "Gemini"],
    ["openai_compatible", "http://localhost:11434/v1", "Ollama"],
    ["openai_compatible", "http://127.0.0.1:11434/v1", "Ollama"],
    ["openai_compatible", "https://api.openai.com/v1", "OpenAI"],
    ["openai_compatible", "https://example.com/v1", "OpenAI-compatible"],
  ] as const)("%s %s -> %s", (llmProvider, llmBaseUrl, label) => {
    expect(providerLabel({ llmProvider, llmBaseUrl })).toBe(label);
  });

  it("creates the configured client", () => {
    expect(createLlm(loadSettings({ LLM_PROVIDER: "none" }))).toBeNull();
    const groq = createLlm(
      loadSettings({
        LLM_PROVIDER: "openai_compatible",
        LLM_API_KEY: "k",
        LLM_MODEL: "llama-3.3-70b-versatile",
        LLM_BASE_URL: "https://api.groq.com/openai/v1",
      }),
    );
    expect(groq?.name).toBe("Groq");
    expect(createLlm(loadSettings({ LLM_PROVIDER: "anthropic", LLM_API_KEY: "k" }))?.name).toBe(
      "Anthropic",
    );
    expect(createLlm(loadSettings({ LLM_PROVIDER: "openai", LLM_API_KEY: "k" }))?.name).toBe(
      "OpenAI",
    );
  });

  it("OpenAICompatibleClient talks to the chat-completions API", async () => {
    const calls: Record<string, unknown>[] = [];
    const chunk = (content: string | null) => ({ choices: [{ delta: { content } }] });
    const create = vi.fn((params: Record<string, unknown>) => {
      calls.push(params);
      if (params.stream) {
        return (async function* () {
          yield chunk("Hel");
          yield chunk(null);
          yield chunk("lo");
          yield { choices: [] };
        })();
      }
      return Promise.resolve({ choices: [{ message: { content: " hi " } }] });
    });
    const fake = { chat: { completions: { create } } };
    const client = new OpenAICompatibleClient(
      "k",
      "m",
      "http://localhost:1/v1",
      "Test",
      fake as never,
    );
    expect(await client.complete("sys", "user")).toBe("hi");
    let streamed = "";
    for await (const t of client.stream("sys", "user")) streamed += t;
    expect(streamed).toBe("Hello");
    expect((calls[0]?.messages as unknown[])[0]).toEqual({ role: "system", content: "sys" });
    expect(calls[0]?.model).toBe("m");
  });

  it("OpenAICompatibleClient tolerates empty completions", async () => {
    const fake = { chat: { completions: { create: () => Promise.resolve({ choices: [] }) } } };
    const client = new OpenAICompatibleClient("", "m", null, "Test", fake as never);
    expect(await client.complete("s", "u")).toBe("");
  });

  it("AnthropicClient talks to the messages API", async () => {
    const create = vi.fn((params: { stream?: boolean }) => {
      if (params.stream) {
        return (async function* () {
          yield { type: "message_start" };
          yield { type: "content_block_delta", delta: { type: "text_delta", text: "a" } };
          yield { type: "content_block_delta", delta: { type: "input_json_delta" } };
          yield { type: "content_block_delta", delta: { type: "text_delta", text: "b" } };
        })();
      }
      return Promise.resolve({ content: [{ type: "text", text: "ok " }, { type: "tool_use" }] });
    });
    const client = new AnthropicClient("k", "claude-test", { messages: { create } } as never);
    expect(await client.complete("s", "u")).toBe("ok");
    let streamed = "";
    for await (const t of client.stream("s", "u")) streamed += t;
    expect(streamed).toBe("ab");
  });
});

describe("query transforms", () => {
  it("keywordsOf strips question scaffolding", () => {
    expect(keywordsOf("What is the value of k in reciprocal rank fusion?")).toBe(
      "value k reciprocal rank fusion",
    );
    expect(keywordsOf("What is it?")).toBe("");
  });

  it("condense is the identity without history", async () => {
    expect(await condenseWithHistory("What is RRF?", [], null)).toBe("What is RRF?");
  });

  it("condense heuristic adds the topic for follow-ups", async () => {
    const history: Turn[] = [
      { role: "user", content: "Explain reciprocal rank fusion" },
      { role: "assistant", content: "It merges..." },
    ];
    const out = await condenseWithHistory("What about its constant?", history, null);
    expect(out).toContain("reciprocal rank fusion");
    expect(out.startsWith("What about its constant")).toBe(true);
  });

  it("condense uses the LLM, and falls back when it fails", async () => {
    const history: Turn[] = [
      { role: "user", content: "Explain BM25" },
      { role: "assistant", content: "A ranking function" },
    ];
    expect(
      await condenseWithHistory("and its k1?", history, new FakeLLM('"What is k1 in BM25?"')),
    ).toBe("What is k1 in BM25?");
    expect(await condenseWithHistory("and its k1?", history, new FakeLLM("fail", true))).toContain(
      "bm25",
    );
  });

  it("leaves standalone questions alone and ignores empty topics", async () => {
    const history: Turn[] = [
      { role: "user", content: "Explain BM25" },
      { role: "assistant", content: "A ranking function" },
    ];
    const q = "How does reciprocal rank fusion combine ranked lists";
    expect(await condenseWithHistory(q, history, null)).toBe(q);
    expect(
      await condenseWithHistory("and it?", [{ role: "user", content: "What is it?" }], null),
    ).toBe("and it?");
  });

  it("rewrite, HyDE and multi-query work offline", async () => {
    const q = "How does hybrid search work?";
    expect(await rewriteQuery(q, null)).toBe("hybrid search work");
    const hyde = await hydeDocument(q, null);
    expect(hyde.toLowerCase()).toContain("hybrid search work");
    expect(hyde.split(/\s+/).length).toBeGreaterThan(15);
    expect(hyde.startsWith("Hybrid search work is described")).toBe(true);
    const variants = await multiQuery(q, null, 3);
    expect(variants.length).toBeGreaterThanOrEqual(1);
    expect(variants.length).toBeLessThanOrEqual(3);
    expect(variants).not.toContain(q);
    expect(await multiQuery("Rockets?", null, 4)).toContain("rockets overview");
  });

  it("uses the LLM when available", async () => {
    expect(await rewriteQuery("q?", new FakeLLM('"rewritten keywords"\nextra'))).toBe(
      "rewritten keywords",
    );
    expect(await hydeDocument("q?", new FakeLLM("Hypothetical answer."))).toBe(
      "Hypothetical answer.",
    );
    expect(
      await multiQuery("q?", new FakeLLM("1. first\n2. second\n- third\n4. fourth\n\nq?")),
    ).toEqual(["first", "second", "third"]);
  });

  it("falls back when the LLM fails or answers nothing", async () => {
    expect(await rewriteQuery("What is BM25?", new FakeLLM("", true))).toBe("bm25");
    expect(
      (await hydeDocument("What is BM25?", new FakeLLM("", true))).toLowerCase().startsWith("bm25"),
    ).toBe(true);
    expect(await rewriteQuery("What is BM25?", new FakeLLM("   "))).toBe("bm25");
    expect(
      (await multiQuery("What is BM25?", new FakeLLM("What is BM25?"))).length,
    ).toBeGreaterThan(0);
  });
});
