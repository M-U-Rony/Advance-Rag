import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowDownUp,
  ChevronDown,
  GitMerge,
  Layers,
  type LucideIcon,
  MessageSquareText,
  Network,
  Scissors,
  Search,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import type { Stage, Trace } from "../lib/types";
import { Badge, cx } from "./ui";

const ICONS: Record<string, LucideIcon> = {
  transform: Sparkles,
  vector: Network,
  bm25: Search,
  fusion: GitMerge,
  rerank: ArrowDownUp,
  parent: Layers,
  compression: Scissors,
  generate: MessageSquareText,
};

const EXPLAIN: Record<string, string> = {
  transform: "Condense chat history and optionally rewrite, HyDE or multi-query the question.",
  vector: "Embed the query and find the nearest chunk vectors.",
  bm25: "Rank chunks by exact keyword statistics (BM25).",
  fusion: "Merge the vector and keyword rankings into one list.",
  rerank: "Re-score the candidates with a more accurate model and keep the best.",
  parent: "Swap matched child chunks for their larger parent passages.",
  compression: "Drop sentences that are irrelevant to the question.",
  generate: "Build the grounded prompt and stream the answer.",
};

export default function Inspector({ trace }: { trace: Trace | null }) {
  if (!trace) {
    return (
      <div className="grid h-full place-items-center p-8 text-center">
        <div>
          <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-white/5 text-slate-500">
            <Network size={22} />
          </div>
          <p className="text-sm font-medium text-slate-300">Pipeline inspector</p>
          <p className="mt-1 text-xs text-slate-500">Ask a question to see every retrieval stage, its latency and hits.</p>
        </div>
      </div>
    );
  }
  const maxMs = Math.max(...trace.stages.map((s) => s.ms), 0.01);
  return (
    <div className="space-y-4 p-5">
      <Summary trace={trace} />
      <TransformDetails trace={trace} />
      <ol className="relative space-y-3 pl-6">
        <span className="absolute bottom-3 left-[11px] top-3 w-px bg-gradient-to-b from-indigo-400/50 via-violet-400/30 to-transparent" />
        {trace.stages.map((stage, i) => (
          <StageCard key={stage.key} stage={stage} maxMs={maxMs} index={i} />
        ))}
      </ol>
    </div>
  );
}

function Summary({ trace }: { trace: Trace }) {
  return (
    <div className="space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <p className="text-sm font-medium leading-snug text-slate-100">{trace.question}</p>
      <div className="flex flex-wrap gap-1.5">
        <Badge tone="accent">{trace.total_ms.toFixed(0)} ms total</Badge>
        <Badge>~{trace.prompt_tokens} prompt tokens</Badge>
        <Badge>{trace.embedder}</Badge>
        <Badge>{trace.llm}</Badge>
        {trace.reranker && <Badge>{trace.reranker}</Badge>}
      </div>
      {trace.warnings.map((w) => (
        <p key={w} className="flex gap-2 text-xs text-amber-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {w}
        </p>
      ))}
    </div>
  );
}

function TransformDetails({ trace }: { trace: Trace }) {
  const rows: Array<[string, string]> = [];
  if (trace.condensed_query !== trace.question) rows.push(["Standalone query", trace.condensed_query]);
  if (trace.rewritten_query) rows.push(["Rewritten query", trace.rewritten_query]);
  if (trace.hyde_text) rows.push(["HyDE passage", trace.hyde_text]);
  trace.multi_queries.forEach((q, i) => rows.push([`Variant ${i + 1}`, q]));
  if (rows.length === 0) return null;
  return (
    <div className="space-y-2 rounded-xl border border-violet-400/20 bg-violet-400/5 p-4">
      {rows.map(([k, v]) => (
        <div key={k}>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-300">{k}</p>
          <p className="text-xs leading-relaxed text-slate-200">{v}</p>
        </div>
      ))}
    </div>
  );
}

function StageCard({ stage, maxMs, index }: { stage: Stage; maxMs: number; index: number }) {
  const [open, setOpen] = useState(false);
  const Icon = ICONS[stage.key] ?? Sparkles;
  const expandable = !stage.skipped && stage.hits.length > 0;
  const pct = stage.skipped ? 0 : Math.max((stage.ms / maxMs) * 100, 3);
  const infoEntries = Object.entries(stage.info).filter(([, v]) => v !== null && v !== undefined);

  return (
    <motion.li
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04 }}
      className="relative"
    >
      <span
        className={cx(
          "absolute -left-6 top-3 grid h-[22px] w-[22px] place-items-center rounded-full border",
          stage.skipped
            ? "border-white/10 bg-ink-800 text-slate-600"
            : "border-indigo-400/60 bg-indigo-500/20 text-indigo-200",
        )}
      >
        <Icon size={12} />
      </span>
      <div className={cx("rounded-xl border p-3 transition", stage.skipped ? "border-white/5 opacity-50" : "border-white/10 bg-white/[0.03]")}>
        <button
          type="button"
          disabled={!expandable}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={expandable ? open : undefined}
          className="flex w-full items-center justify-between gap-2 text-left disabled:cursor-default"
        >
          <span>
            <span className="block text-sm font-medium text-slate-100">{stage.label}</span>
            <span className="block text-[11px] text-slate-500">{stage.skipped ? "Skipped (off)" : EXPLAIN[stage.key]}</span>
          </span>
          <span className="flex shrink-0 items-center gap-2">
            {!stage.skipped && <span className="font-mono text-xs text-indigo-200">{stage.ms.toFixed(1)} ms</span>}
            {expandable && <ChevronDown size={14} className={cx("text-slate-500 transition", open && "rotate-180")} />}
          </span>
        </button>
        {!stage.skipped && (
          <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/5" role="presentation">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-violet-400"
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.6, delay: index * 0.04 }}
            />
          </div>
        )}
        {!stage.skipped && infoEntries.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {infoEntries.map(([k, v]) => (
              <span key={k} className="rounded-md bg-black/30 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                {k}: <span className="text-slate-200">{String(v)}</span>
              </span>
            ))}
          </div>
        )}
        <AnimatePresence initial={false}>
          {open && (
            <motion.ul
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-3 space-y-2 overflow-hidden"
            >
              {stage.hits.slice(0, 15).map((h) => (
                <li key={h.chunk_id + h.rank} className="rounded-lg bg-black/25 p-2.5">
                  <div className="mb-1 flex items-center justify-between gap-2 text-[11px]">
                    <span className="flex items-center gap-1.5 text-slate-300">
                      <span className="grid h-4 min-w-4 place-items-center rounded bg-indigo-400/20 px-1 font-semibold text-indigo-200">
                        {h.rank}
                      </span>
                      <span className="truncate">
                        {h.filename} p.{h.page}
                      </span>
                      {h.is_parent && <Badge tone="accent">parent</Badge>}
                    </span>
                    <span className="font-mono text-slate-400">{h.score.toFixed(4)}</span>
                  </div>
                  <p className="line-clamp-3 text-[11px] leading-relaxed text-slate-400">{h.snippet}</p>
                </li>
              ))}
            </motion.ul>
          )}
        </AnimatePresence>
      </div>
    </motion.li>
  );
}
