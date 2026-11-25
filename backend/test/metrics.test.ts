import { describe, expect, it } from "vitest";
import {
  evaluate,
  autoGenerateQa,
  relevantIds,
  loadQa,
  type QAItem,
} from "../src/evaluation/evaluator.js";
import * as metrics from "../src/evaluation/metrics.js";
import { formatReport } from "../src/evaluation/cli.js";
import { NO_ANSWER } from "../src/generation/prompts.js";
import { REPO_ROOT } from "../src/config.js";
import { FakeLLM, makeIndexed } from "./helpers.js";
import path from "node:path";

describe("retrieval metrics", () => {
  it("computes recall, precision and MRR", () => {
    const retrieved = ["a", "b", "c", "d"];
    const relevant = new Set(["c", "z"]);
    expect(metrics.recallAtK(retrieved, relevant, 2)).toBe(0);
    expect(metrics.recallAtK(retrieved, relevant, 3)).toBe(1);
    expect(metrics.precisionAtK(retrieved, relevant, 4)).toBe(0.25);
    expect(metrics.reciprocalRank(retrieved, relevant)).toBeCloseTo(1 / 3);
    expect(metrics.reciprocalRank(["x"], relevant)).toBe(0);
    expect(metrics.recallAtK([], new Set(), 3)).toBe(0);
    expect(metrics.precisionAtK([], relevant, 3)).toBe(0);
  });
});

describe("generation metrics", () => {
  const context = ["Cosine similarity measures the angle between two vectors."];
  const grounded = "Cosine similarity measures the angle between vectors.";
  const hallucinated = "The Eiffel Tower was completed in 1889 by Gustave Eiffel.";

  it("faithfulness uses a 50% content-word overlap threshold", () => {
    expect(metrics.faithfulnessOverlap(grounded, context)).toBe(1);
    expect(metrics.faithfulnessOverlap(hallucinated, context)).toBe(0);
    expect(metrics.faithfulnessOverlap(`${grounded} ${hallucinated}`, context)).toBe(0.5);
    expect(metrics.faithfulnessOverlap(NO_ANSWER, context)).toBe(1);
    expect(metrics.faithfulnessOverlap("", context)).toBe(0);
  });

  it("answer relevance is the share of question keywords in the answer", () => {
    const q = "What is cosine similarity?";
    expect(metrics.answerRelevance(q, "Cosine similarity is an angle.")).toBe(1);
    expect(metrics.answerRelevance(q, "Bananas.")).toBe(0);
    expect(metrics.answerRelevance(q, NO_ANSWER)).toBe(0);
    expect(metrics.answerRelevance("the of", "anything")).toBe(0);
  });

  it("LLM judge output is clamped and parsed", async () => {
    expect(await metrics.llmFaithfulness("a", ["b"], new FakeLLM("0.75"))).toBe(0.75);
    expect(await metrics.llmFaithfulness("a", ["b"], new FakeLLM("9"))).toBe(1);
    expect(await metrics.llmFaithfulness("a", ["b"], new FakeLLM("not a number"))).toBe(0);
    expect(await metrics.llmFaithfulness("a", ["b"], new FakeLLM(""))).toBe(0);
  });
});

describe("evaluator", () => {
  it("finds relevant chunks and generates questions", async () => {
    const chunks = (await makeIndexed()).index.chunks;
    expect(relevantIds(chunks, "Chlorophyll   inside the chloroplasts").size).toBeGreaterThan(0);
    expect(relevantIds(chunks, "dinosaurs").size).toBe(0);
    const qa = autoGenerateQa(chunks, 3);
    expect(qa.length).toBeGreaterThan(0);
    expect(qa.every((item) => relevantIds(chunks, item.evidence).size > 0)).toBe(true);
    expect(autoGenerateQa(chunks, 3)).toEqual(qa); // deterministic
  });

  it("loads the sample QA file", () => {
    const qa = loadQa(path.join(REPO_ROOT, "samples", "rag_handbook_qa.json"));
    expect(qa.length).toBeGreaterThanOrEqual(20);
    expect(qa[0]).toHaveProperty("evidence");
  });

  it("reports the four configurations", async () => {
    const pipeline = await makeIndexed();
    const qa: QAItem[] = [
      {
        question: "What absorbs red and blue light in plants?",
        evidence: "absorbs red and blue light",
      },
      { question: "What causes spring tides?", evidence: "Sun and Moon align" },
      { question: "Unanswerable question about nothing", evidence: "no such evidence anywhere" },
    ];
    const report = await evaluate(pipeline, qa, { ks: [1, 3] });
    expect(report.questions).toBe(2);
    expect(report.rows.map((r) => r.config)).toEqual([
      "vector",
      "bm25",
      "hybrid",
      "hybrid + rerank",
    ]);
    for (const row of report.rows) {
      expect(Object.keys(row.recall)).toEqual(["@1", "@3"]);
      expect(row.recall["@3"] as number).toBeGreaterThanOrEqual(row.recall["@1"] as number);
      expect(row.faithfulness).toBeGreaterThanOrEqual(0);
      expect(row.faithfulness).toBeLessThanOrEqual(1);
    }
    expect(report.rows.at(-1)?.recall["@3"]).toBe(1);
    expect(formatReport(report)).toContain("| hybrid + rerank |");
  });

  it("copes with no labelled questions", async () => {
    const report = await evaluate(await makeIndexed(), [{ question: "x", evidence: "nope" }], {
      ks: [1],
    });
    expect(report.questions).toBe(0);
    expect(report.rows[0]?.mrr).toBe(0);
  });
});
