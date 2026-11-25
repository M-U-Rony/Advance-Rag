import { motion } from "framer-motion";
import { ArrowUp, Check, Copy, Sparkles, Square, User } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { STARTER_QUESTIONS } from "../lib/defaults";
import type { ChatMessage, Citation } from "../lib/types";
import { cx } from "./ui";

interface Props {
  messages: ChatMessage[];
  hasDocuments: boolean;
  busy: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onSend: (question: string) => void;
  onStop: () => void;
  onLoadSample: () => void;
  sampleAvailable: boolean;
}

export default function Chat(p: Props) {
  const [draft, setDraft] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [p.messages]);

  const submit = () => {
    const q = draft.trim();
    if (!q || p.busy) return;
    setDraft("");
    p.onSend(q);
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label="Chat">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8" role="log" aria-live="polite">
        <div className="mx-auto max-w-3xl space-y-6">
          {p.messages.length === 0 ? (
            <Welcome {...p} />
          ) : (
            p.messages.map((m) => (
              <MessageBubble key={m.id} message={m} selected={m.id === p.selectedId} onSelect={() => p.onSelect(m.id)} />
            ))
          )}
          <div ref={bottom} />
        </div>
      </div>

      <div className="border-t border-white/10 bg-ink-900/60 px-4 py-4 backdrop-blur sm:px-8">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-white/10 bg-ink-800/80 p-2 shadow-lg shadow-black/30 focus-within:border-indigo-400/50"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            aria-label="Ask a question about your documents"
            placeholder={p.hasDocuments ? "Ask anything about your documents..." : "Upload a PDF to start asking questions"}
            disabled={!p.hasDocuments}
            className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500 disabled:cursor-not-allowed"
          />
          {p.busy ? (
            <button
              type="button"
              onClick={p.onStop}
              aria-label="Stop generating"
              className="grid h-11 w-11 place-items-center rounded-xl bg-white/10 text-slate-100 transition hover:bg-white/20"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!draft.trim() || !p.hasDocuments}
              aria-label="Send question"
              className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-lg shadow-indigo-500/30 outline-none transition hover:brightness-110 focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:opacity-30 disabled:shadow-none"
            >
              <ArrowUp size={18} />
            </button>
          )}
        </form>
      </div>
    </section>
  );
}

function Welcome({ hasDocuments, onSend, onLoadSample, sampleAvailable }: Props) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="pt-10 text-center sm:pt-20">
      <div className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 shadow-xl shadow-indigo-500/30">
        <Sparkles className="text-white" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
        Ask your PDFs <span className="text-gradient">anything</span>
      </h2>
      <p className="mx-auto mt-3 max-w-lg text-sm text-slate-400">
        Hybrid search, reranking and query transformation, with every pipeline stage visible in the inspector on the
        right.
      </p>
      {hasDocuments ? (
        <div className="mx-auto mt-8 grid max-w-2xl gap-3 sm:grid-cols-2">
          {STARTER_QUESTIONS.map((q, i) => (
            <motion.button
              key={q}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 * i }}
              onClick={() => onSend(q)}
              className="rounded-xl border border-white/10 bg-white/[0.03] p-4 text-left text-sm text-slate-300 outline-none transition hover:border-indigo-400/50 hover:bg-indigo-400/10 focus-visible:ring-2 focus-visible:ring-indigo-400/70"
            >
              {q}
            </motion.button>
          ))}
        </div>
      ) : (
        sampleAvailable && (
          <button
            onClick={onLoadSample}
            className="mt-8 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:brightness-110"
          >
            Load the sample PDF to try it
          </button>
        )
      )}
    </motion.div>
  );
}

function MessageBubble({
  message,
  selected,
  onSelect,
}: {
  message: ChatMessage;
  selected: boolean;
  onSelect: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const text = useMemo(() => message.content.replace(/\[(\d+)\](?!\()/g, "[$1](cite:$1)"), [message.content]);

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <motion.article
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cx("flex gap-3", isUser && "flex-row-reverse")}
    >
      <div
        className={cx(
          "grid h-8 w-8 shrink-0 place-items-center rounded-lg",
          isUser ? "bg-white/10 text-slate-300" : "bg-gradient-to-br from-indigo-500 to-violet-500 text-white",
        )}
      >
        {isUser ? <User size={16} /> : <Sparkles size={16} />}
      </div>
      <div className={cx("min-w-0 max-w-[85%]", isUser && "text-right")}>
        <div
          onClick={!isUser ? onSelect : undefined}
          className={cx(
            "inline-block rounded-2xl px-4 py-3 text-left text-sm",
            isUser
              ? "bg-indigo-500/20 text-slate-100"
              : cx(
                  "border bg-ink-800/70 text-slate-200",
                  selected ? "border-indigo-400/40" : "border-white/10",
                  message.error && "border-rose-400/40 text-rose-200",
                ),
          )}
        >
          {isUser ? (
            message.content
          ) : (
            <div className={cx("prose-chat", message.streaming && "caret")}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                urlTransform={(u) => u}
                components={{
                  a: ({ href, children }) => {
                    if (href?.startsWith("cite:")) {
                      const n = Number(href.slice(5));
                      return <CitationChip index={n} citation={message.citations?.find((c) => c.index === n)} />;
                    }
                    return (
                      <a href={href} target="_blank" rel="noreferrer" className="text-indigo-300 underline">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {text || (message.streaming ? "Retrieving..." : "")}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && !message.streaming && message.content && (
          <div className="mt-1.5 flex items-center gap-3 text-[11px] text-slate-500">
            <button onClick={() => void copy()} className="flex items-center gap-1 transition hover:text-slate-200">
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? "Copied" : "Copy"}
            </button>
            {message.trace && (
              <button onClick={onSelect} className="transition hover:text-slate-200">
                {message.trace.total_ms.toFixed(0)} ms - view pipeline
              </button>
            )}
          </div>
        )}
      </div>
    </motion.article>
  );
}

function CitationChip({ index, citation }: { index: number; citation?: Citation }) {
  const [pinned, setPinned] = useState(false);
  return (
    <span className="group relative mx-0.5 inline-block align-baseline">
      <button
        type="button"
        onClick={() => setPinned((v) => !v)}
        aria-label={citation ? `Source ${index}: ${citation.filename}, page ${citation.page}` : `Source ${index}`}
        className="inline-grid h-[18px] min-w-[18px] place-items-center rounded-md bg-indigo-400/20 px-1 align-text-top text-[11px] font-semibold text-indigo-200 outline-none transition hover:bg-indigo-400/40 focus-visible:ring-2 focus-visible:ring-indigo-300"
      >
        {index}
      </button>
      {citation && (
        <span
          role="tooltip"
          className={cx(
            "absolute bottom-full left-1/2 z-40 mb-2 w-72 -translate-x-1/2 rounded-xl border border-white/10 bg-ink-700 p-3 text-left text-xs leading-relaxed text-slate-200 shadow-2xl group-focus-within:block group-hover:block",
            pinned ? "block" : "hidden",
          )}
        >
          <span className="mb-1 block font-semibold text-indigo-200">
            {citation.filename} - page {citation.page}
          </span>
          <span className="line-clamp-6 block text-slate-300">{citation.text}</span>
        </span>
      )}
    </span>
  );
}
