/**
 * CLI: compare vector / BM25 / hybrid / hybrid+rerank retrieval on a PDF.
 *
 * Usage (from `backend/`):
 *
 * ```
 * npm run eval -- ../samples/rag_handbook.pdf              # uses <stem>_qa.json next to it
 * npm run eval -- my.pdf --qa my_qa.json --ks 1,3,5
 * EMBEDDING_BACKEND=tfidf npm run eval -- my.pdf           # force the offline baseline
 * ```
 */
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { loadDotEnv, loadSettings } from "../config.js";
import { STRATEGIES, chunkingParams, type Strategy } from "../ingestion/chunking.js";
import { buildPipeline, type PipelineParts } from "../runtime.js";
import type { EvalReport } from "../schemas.js";
import { autoGenerateQa, evaluate, loadQa, type QAItem } from "./evaluator.js";

/** Render the report as a Markdown table. */
export function formatReport(report: EvalReport): string {
  const header = [
    "Config",
    ...report.ks.map((k) => `Recall@${k}`),
    `P@${report.precision_k}`,
    "MRR",
    "Faithful",
    "AnsRel",
    "ms",
  ];
  const lines = [
    `Embedder: ${report.embedder} | questions: ${report.questions} | QA source: ${report.source}`,
    "",
    `| ${header.join(" | ")} |`,
    `|${["---", ...header.slice(1).map(() => "---:")].join("|")}|`,
  ];
  for (const row of report.rows) {
    const cells = [
      row.config,
      ...report.ks.map((k) => (row.recall[`@${k}`] as number).toFixed(2)),
      row.precision.toFixed(2),
      row.mrr.toFixed(2),
      row.faithfulness.toFixed(2),
      row.answer_relevance.toFixed(2),
      row.avg_ms.toFixed(1),
    ];
    lines.push(`| ${cells.join(" | ")} |`);
  }
  return lines.join("\n");
}

/** Run the evaluation CLI with `argv` (default: the process arguments). */
export async function runCli(
  argv: string[] = process.argv.slice(2),
  parts: PipelineParts = {},
): Promise<EvalReport> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      qa: { type: "string" },
      ks: { type: "string", default: "1,3,5" },
      strategy: { type: "string", default: "recursive" },
      "auto-questions": { type: "string", default: "20" },
    },
  });
  const pdf = positionals[0];
  if (!pdf) throw new Error("usage: npm run eval -- <file.pdf> [--qa file.json] [--ks 1,3,5]");
  if (!STRATEGIES.includes(values.strategy as Strategy)) {
    throw new Error(`--strategy must be one of ${STRATEGIES.join(", ")}`);
  }
  const ks = (values.ks as string).split(/[\s,]+/).map(Number);

  loadDotEnv();
  const tmp = mkdtempSync(path.join(os.tmpdir(), "advance-rag-eval-"));
  try {
    // Keep the model cache of the normal data dir; only the document store is temporary.
    const pipeline = await buildPipeline({ ...loadSettings(), dataDir: tmp }, parts);
    await pipeline.addPdfFile(pdf, chunkingParams({ strategy: values.strategy as Strategy }));
    const qaPath = values.qa ?? path.join(path.dirname(pdf), `${path.parse(pdf).name}_qa.json`);
    let qa: QAItem[];
    let source: string;
    if (existsSync(qaPath)) {
      qa = loadQa(qaPath);
      source = path.basename(qaPath);
    } else {
      qa = autoGenerateQa(pipeline.index.chunks, Number(values["auto-questions"]));
      source = "auto-generated";
    }
    const report = await evaluate(pipeline, qa, { ks, source });
    console.log(formatReport(report));
    return report;
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}
