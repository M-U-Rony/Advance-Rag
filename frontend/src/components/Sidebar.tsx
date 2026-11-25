import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Cpu, FileText, FlaskConical, Layers, Loader2, Trash2, UploadCloud, X } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { api } from "../lib/api";
import { DEFAULT_CHUNKING } from "../lib/defaults";
import type { AppConfig, ChunkingOptions, DocumentInfo, Strategy } from "../lib/types";
import { Badge, Segmented, cx } from "./ui";

interface Props {
  documents: DocumentInfo[];
  config: AppConfig | null;
  onChanged: () => Promise<void>;
  onClose: () => void;
}

export default function Sidebar({ documents, config, onChanged, onClose }: Props) {
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<ChunkingOptions>(DEFAULT_CHUNKING);
  const [showOptions, setShowOptions] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      setError(null);
      for (const file of Array.from(files)) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          setError(`${file.name} is not a PDF`);
          continue;
        }
        try {
          setProgress(0);
          setIndexing(false);
          await api.upload(file, options, (f) => {
            setProgress(f);
            if (f >= 1) setIndexing(true);
          });
          await onChanged();
        } catch (e) {
          setError(e instanceof Error ? e.message : "Upload failed");
        } finally {
          setProgress(null);
          setIndexing(false);
        }
      }
    },
    [options, onChanged],
  );

  const loadSample = async () => {
    setError(null);
    setIndexing(true);
    try {
      await api.loadSample();
      await onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load sample");
    } finally {
      setIndexing(false);
    }
  };

  const remove = async (id: string) => {
    await api.remove(id);
    await onChanged();
  };

  const busy = progress !== null || indexing;

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r border-white/10 bg-ink-900/95 backdrop-blur">
      <div className="flex items-center justify-between px-5 pb-4 pt-5">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 shadow-lg shadow-indigo-500/30">
            <Layers size={18} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-none tracking-tight">
              Advance <span className="text-gradient">RAG</span>
            </h1>
            <p className="mt-1 text-[11px] text-slate-500">Inspectable retrieval pipeline</p>
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close sidebar"
          className="rounded-lg p-1.5 text-slate-400 hover:bg-white/5 lg:hidden"
        >
          <X size={18} />
        </button>
      </div>

      <div className="space-y-4 overflow-y-auto px-5 pb-4">
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload PDF"
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void upload(e.dataTransfer.files);
          }}
          className={cx(
            "group cursor-pointer rounded-2xl border border-dashed p-5 text-center outline-none transition focus-visible:ring-2 focus-visible:ring-indigo-400/70",
            dragging
              ? "border-indigo-400 bg-indigo-400/10"
              : "border-white/15 bg-white/[0.02] hover:border-indigo-400/60 hover:bg-indigo-400/5",
          )}
        >
          <input
            ref={input}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            hidden
            onChange={(e) => e.target.files && void upload(e.target.files)}
          />
          {busy ? (
            <div className="space-y-3">
              <Loader2 className="mx-auto animate-spin text-indigo-300" size={22} />
              <p className="text-xs text-slate-300">
                {indexing ? "Chunking and embedding..." : `Uploading ${Math.round((progress ?? 0) * 100)}%`}
              </p>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-violet-400"
                  animate={{ width: `${indexing ? 100 : (progress ?? 0) * 100}%` }}
                  transition={{ ease: "easeOut" }}
                />
              </div>
            </div>
          ) : (
            <>
              <UploadCloud className="mx-auto mb-2 text-indigo-300 transition group-hover:-translate-y-0.5" size={24} />
              <p className="text-sm font-medium text-slate-200">Drop PDFs here</p>
              <p className="text-xs text-slate-500">or click to browse</p>
            </>
          )}
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowOptions((v) => !v)}
            aria-expanded={showOptions}
            className="flex w-full items-center justify-between text-xs font-medium uppercase tracking-wide text-slate-400 hover:text-slate-200"
          >
            Chunking options
            <ChevronDown size={14} className={cx("transition", showOptions && "rotate-180")} />
          </button>
          <AnimatePresence initial={false}>
            {showOptions && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <ChunkingForm options={options} onChange={setOptions} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {config?.sample_available && (
          <button
            onClick={() => void loadSample()}
            disabled={busy}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-indigo-400/30 bg-indigo-400/10 px-3 py-2.5 text-sm font-medium text-indigo-100 outline-none transition hover:bg-indigo-400/20 focus-visible:ring-2 focus-visible:ring-indigo-400/70 disabled:opacity-50"
          >
            <FlaskConical size={16} /> Load sample PDF
          </button>
        )}

        {error && (
          <p role="alert" className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-xs text-rose-200">
            {error}
          </p>
        )}

        <div>
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            Documents ({documents.length})
          </h2>
          <ul className="space-y-2">
            <AnimatePresence initial={false}>
              {documents.map((d) => (
                <motion.li
                  key={d.id}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -16 }}
                  className="group flex items-start gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-3"
                >
                  <FileText size={18} className="mt-0.5 shrink-0 text-indigo-300" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-slate-100" title={d.filename}>
                      {d.filename}
                    </p>
                    <p className="mt-0.5 text-[11px] text-slate-500">
                      {d.pages} pages - {d.chunks} chunks
                      {d.parent_chunks > 0 && ` - ${d.parent_chunks} parents`}
                    </p>
                    <Badge tone="accent">{d.strategy.replace("_", "-")}</Badge>
                  </div>
                  <button
                    onClick={() => void remove(d.id)}
                    aria-label={`Delete ${d.filename}`}
                    className="rounded-lg p-1.5 text-slate-500 opacity-0 outline-none transition hover:bg-rose-400/10 hover:text-rose-300 focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    <Trash2 size={15} />
                  </button>
                </motion.li>
              ))}
            </AnimatePresence>
            {documents.length === 0 && (
              <li className="rounded-xl border border-white/5 px-3 py-4 text-center text-xs text-slate-500">
                No documents yet. Upload a PDF or load the sample.
              </li>
            )}
          </ul>
        </div>
      </div>

      <StatusFooter config={config} />
    </aside>
  );
}

function ChunkingForm({ options, onChange }: { options: ChunkingOptions; onChange: (o: ChunkingOptions) => void }) {
  const set = <K extends keyof ChunkingOptions>(k: K, v: ChunkingOptions[K]) => onChange({ ...options, [k]: v });
  const num = (k: keyof ChunkingOptions, label: string) => (
    <label className="block text-[11px] text-slate-400">
      {label}
      <input
        type="number"
        value={options[k] as number}
        min={5}
        onChange={(e) => set(k, Number(e.target.value) as never)}
        className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2 py-1.5 text-sm text-slate-100 outline-none focus:border-indigo-400/60"
      />
    </label>
  );
  const isPC = options.strategy === "parent_child";
  return (
    <div className="space-y-3 pt-3">
      <Segmented<Strategy>
        label="Chunking strategy"
        value={options.strategy}
        onChange={(v) => set("strategy", v)}
        options={[
          { value: "fixed", label: "Fixed" },
          { value: "recursive", label: "Recursive" },
          { value: "semantic", label: "Heading" },
          { value: "parent_child", label: "Parent" },
        ]}
      />
      <div className="grid grid-cols-2 gap-2">
        {isPC ? (
          <>
            {num("parent_size", "Parent words")}
            {num("child_size", "Child words")}
          </>
        ) : (
          <>
            {num("chunk_size", "Chunk words")}
            {options.strategy === "fixed" && num("chunk_overlap", "Overlap words")}
          </>
        )}
      </div>
    </div>
  );
}

function StatusFooter({ config }: { config: AppConfig | null }) {
  if (!config) return <div className="h-24 border-t border-white/10" />;
  const fallback = config.embeddings.fallback_reason;
  return (
    <div className="space-y-2 border-t border-white/10 px-5 py-4 text-xs">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-slate-400">
          <Cpu size={14} /> LLM
        </span>
        <Badge tone={config.llm.offline ? "warn" : "good"}>
          {config.llm.offline ? "Offline (extractive)" : `${config.llm.label}${config.llm.model ? ` - ${config.llm.model}` : ""}`}
        </Badge>
      </div>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-slate-400">
          <Layers size={14} /> Embeddings
        </span>
        <Badge tone={config.embeddings.dense ? "good" : "warn"}>
          {config.embeddings.dense ? config.embeddings.active : "TF-IDF"}
        </Badge>
      </div>
      {fallback && (
        <p className="text-[11px] leading-snug text-amber-300/80" title={fallback}>
          Dense model unavailable, using TF-IDF fallback.
        </p>
      )}
    </div>
  );
}
