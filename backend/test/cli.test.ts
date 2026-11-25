import { mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { REPO_ROOT } from "../src/config.js";
import { formatReport, runCli } from "../src/evaluation/cli.js";
import { brokenCrossEncoder, makePdf } from "./helpers.js";

const offline = { crossEncoder: brokenCrossEncoder() };

afterEach(() => vi.restoreAllMocks());

describe("evaluation CLI", () => {
  it("prints the comparison table for the sample handbook", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const report = await runCli(
      [path.join(REPO_ROOT, "samples", "rag_handbook.pdf"), "--ks", "1,3,5"],
      offline,
    );
    expect(report.questions).toBeGreaterThanOrEqual(20);
    expect(report.rows).toHaveLength(4);
    const printed = log.mock.calls[0]?.[0] as string;
    expect(printed).toContain("Embedder: tfidf");
    expect(printed).toContain("QA source: rag_handbook_qa.json");
    expect(printed).toContain(
      "| Config | Recall@1 | Recall@3 | Recall@5 | P@5 | MRR | Faithful | AnsRel | ms |",
    );
    expect(printed.split("\n")).toHaveLength(8);
  });

  it("auto-generates questions when there is no QA file", async () => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    const dir = mkdtempSync(path.join(os.tmpdir(), "eval-"));
    const pdf = path.join(dir, "nature.pdf");
    writeFileSync(pdf, await makePdf());
    const report = await runCli([pdf, "--strategy", "fixed", "--auto-questions", "3"], offline);
    expect(report.source).toBe("auto-generated");
  });

  it("uses an explicit --qa file", async () => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    const dir = mkdtempSync(path.join(os.tmpdir(), "eval-"));
    const pdf = path.join(dir, "nature.pdf");
    const qa = path.join(dir, "custom.json");
    writeFileSync(pdf, await makePdf());
    writeFileSync(
      qa,
      JSON.stringify([{ question: "What causes spring tides?", evidence: "Sun and Moon align" }]),
    );
    const report = await runCli([pdf, "--qa", qa, "--ks", "1"], offline);
    expect(report).toMatchObject({ source: "custom.json", questions: 1 });
  });

  it("validates its arguments", async () => {
    await expect(runCli([], offline)).rejects.toThrow(/usage/);
    await expect(runCli(["x.pdf", "--strategy", "bogus"], offline)).rejects.toThrow(/--strategy/);
  });

  it("formats a Markdown table", () => {
    const table = formatReport({
      embedder: "e",
      questions: 2,
      ks: [1, 3],
      precision_k: 3,
      source: "s",
      rows: [
        {
          config: "vector",
          recall: { "@1": 0.5, "@3": 1 },
          precision: 0.33333,
          mrr: 0.75,
          faithfulness: 1,
          answer_relevance: 0.5,
          avg_ms: 12.34,
        },
      ],
    });
    expect(table).toContain("|---|---:|---:|---:|---:|---:|---:|---:|");
    expect(table).toContain("| vector | 0.50 | 1.00 | 0.33 | 0.75 | 1.00 | 0.50 | 12.3 |");
  });
});
