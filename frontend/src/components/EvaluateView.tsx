import { BarChart3, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import type { EvalReport } from "../lib/types";
import { Badge } from "./ui";

const COLORS = ["#818cf8", "#34d399", "#fbbf24", "#f472b6"];

export default function EvaluateView({ hasDocuments }: { hasDocuments: boolean }) {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.evaluate([1, 3, 5]));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  const chartData = report?.ks.map((k) => ({
    name: `Recall@${k}`,
    ...Object.fromEntries(report.rows.map((r) => [r.config, r.recall[`@${k}`]])),
  }));
  const bestMrr = report ? Math.max(...report.rows.map((r) => r.mrr)) : 0;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Retrieval evaluation</h2>
            <p className="mt-1 max-w-xl text-sm text-slate-400">
              Compares vector-only, BM25-only, hybrid (RRF) and hybrid + rerank on a labelled question set. A retrieved
              chunk counts as relevant if it contains the question&apos;s evidence phrase.
            </p>
          </div>
          <button
            onClick={() => void run()}
            disabled={!hasDocuments || loading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:brightness-110 disabled:opacity-40"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {loading ? "Evaluating..." : "Run evaluation"}
          </button>
        </div>

        {!hasDocuments && <p className="text-sm text-amber-300">Index a document (or load the sample) first.</p>}
        {error && (
          <p role="alert" className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </p>
        )}

        {!report && !loading && hasDocuments && (
          <div className="grid place-items-center rounded-2xl border border-dashed border-white/10 py-20 text-center">
            <BarChart3 className="mb-3 text-slate-600" size={32} />
            <p className="text-sm text-slate-400">Run the evaluation to see the Recall@k comparison.</p>
          </div>
        )}

        {report && chartData && (
          <>
            <div className="flex flex-wrap gap-2">
              <Badge tone="accent">{report.questions} questions</Badge>
              <Badge>embeddings: {report.embedder}</Badge>
              <Badge>QA set: {report.source}</Badge>
            </div>
            <div
              className="h-80 rounded-2xl border border-white/10 bg-white/[0.02] p-4"
              role="img"
              aria-label="Recall at k bar chart per retrieval configuration"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 1]} stroke="#64748b" tickLine={false} axisLine={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    contentStyle={{ background: "#1c1c2e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10 }}
                    labelStyle={{ color: "#e2e8f0" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {report.rows.map((r, i) => (
                    <Bar key={r.config} dataKey={r.config} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-white/10">
              <table className="w-full text-sm">
                <thead className="bg-white/[0.04] text-left text-xs uppercase tracking-wide text-slate-400">
                  <tr>
                    <th className="px-4 py-3">Configuration</th>
                    {report.ks.map((k) => (
                      <th key={k} className="px-3 py-3 text-right">
                        R@{k}
                      </th>
                    ))}
                    <th className="px-3 py-3 text-right">P@{report.precision_k}</th>
                    <th className="px-3 py-3 text-right">MRR</th>
                    <th className="px-3 py-3 text-right">Faithful</th>
                    <th className="px-3 py-3 text-right">Ans. rel.</th>
                    <th className="px-3 py-3 text-right">ms/query</th>
                  </tr>
                </thead>
                <tbody>
                  {report.rows.map((r, i) => (
                    <tr key={r.config} className="border-t border-white/5">
                      <td className="px-4 py-3 font-medium text-slate-100">
                        <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: COLORS[i % COLORS.length] }} />
                        {r.config}
                      </td>
                      {report.ks.map((k) => (
                        <td key={k} className="px-3 py-3 text-right font-mono text-slate-300">
                          {r.recall[`@${k}`].toFixed(2)}
                        </td>
                      ))}
                      <td className="px-3 py-3 text-right font-mono text-slate-300">{r.precision.toFixed(2)}</td>
                      <td
                        className={`px-3 py-3 text-right font-mono ${r.mrr === bestMrr ? "font-semibold text-emerald-300" : "text-slate-300"}`}
                      >
                        {r.mrr.toFixed(2)}
                      </td>
                      <td className="px-3 py-3 text-right font-mono text-slate-300">{r.faithfulness.toFixed(2)}</td>
                      <td className="px-3 py-3 text-right font-mono text-slate-300">{r.answer_relevance.toFixed(2)}</td>
                      <td className="px-3 py-3 text-right font-mono text-slate-400">{r.avg_ms.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
