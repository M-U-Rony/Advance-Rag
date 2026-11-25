import type { DocumentInfo, RetrievalSettings } from "../lib/types";
import { Field, Hint, Segmented, Slider, Toggle } from "./ui";

interface Props {
  settings: RetrievalSettings;
  documents: DocumentInfo[];
  onChange: (s: RetrievalSettings) => void;
  onReset: () => void;
}

export default function SettingsPanel({ settings: s, documents, onChange, onReset }: Props) {
  const set = <K extends keyof RetrievalSettings>(k: K, v: RetrievalSettings[K]) => onChange({ ...s, [k]: v });
  const usesVector = s.retrieval_mode !== "bm25";
  const toggleDoc = (id: string) => {
    const current = s.doc_ids ?? documents.map((d) => d.id);
    const next = current.includes(id) ? current.filter((x) => x !== id) : [...current, id];
    set("doc_ids", next.length === documents.length ? null : next);
  };

  return (
    <div className="space-y-6 p-5">
      <Field
        label="Retrieval mode"
        hint="Vector search matches meaning (embeddings). BM25 matches exact words. Hybrid runs both in parallel and fuses the rankings."
      >
        <Segmented
          label="Retrieval mode"
          value={s.retrieval_mode}
          onChange={(v) => set("retrieval_mode", v)}
          options={[
            { value: "vector", label: "Vector" },
            { value: "bm25", label: "BM25" },
            { value: "hybrid", label: "Hybrid" },
          ]}
        />
      </Field>

      {s.retrieval_mode === "hybrid" && (
        <>
          <Field
            label="Fusion"
            hint="Reciprocal Rank Fusion scores 1/(60+rank) and ignores raw scores. Weighted blends min-max normalised scores using alpha."
          >
            <Segmented
              label="Fusion method"
              value={s.fusion}
              onChange={(v) => set("fusion", v)}
              options={[
                { value: "rrf", label: "RRF (k=60)" },
                { value: "weighted", label: "Weighted" },
              ]}
            />
          </Field>
          {s.fusion === "weighted" && (
            <Field
              label="Alpha"
              hint="Weight of the vector score. 0 = keyword only, 1 = vector only."
              right={<span className="font-mono text-xs text-indigo-200">{s.alpha.toFixed(2)}</span>}
            >
              <Slider label="Alpha" value={s.alpha} min={0} max={1} step={0.05} onChange={(v) => set("alpha", v)} />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>keyword</span>
                <span>vector</span>
              </div>
            </Field>
          )}
        </>
      )}

      {usesVector && (
        <Field
          label="Similarity"
          hint="Cosine compares direction (length-invariant). Dot product also rewards magnitude. L2 is straight-line distance (smaller is closer)."
        >
          <Segmented
            label="Similarity metric"
            value={s.similarity}
            onChange={(v) => set("similarity", v)}
            options={[
              { value: "cosine", label: "Cosine" },
              { value: "dot", label: "Dot" },
              { value: "l2", label: "L2" },
            ]}
          />
        </Field>
      )}

      <Field
        label="Top-k"
        hint="How many passages are passed to the LLM as context."
        right={<span className="font-mono text-xs text-indigo-200">{s.top_k}</span>}
      >
        <Slider label="Top k" value={s.top_k} min={1} max={15} onChange={(v) => set("top_k", v)} />
      </Field>

      <Field
        label="Candidates"
        hint="How many passages first-stage retrieval fetches before reranking narrows them to top-k. Retrieve wide, rerank narrow."
        right={<span className="font-mono text-xs text-indigo-200">{s.candidates}</span>}
      >
        <Slider label="Candidates" value={s.candidates} min={5} max={100} step={5} onChange={(v) => set("candidates", v)} />
      </Field>

      <Field
        label="Rerank"
        hint="A second, more accurate pass. The cross-encoder reads query and passage together (slower, best quality). Heuristic uses term overlap, bigrams and position."
      >
        <Segmented
          label="Reranker"
          value={s.rerank}
          onChange={(v) => set("rerank", v)}
          options={[
            { value: "none", label: "Off" },
            { value: "heuristic", label: "Heuristic" },
            { value: "cross_encoder", label: "Cross-enc." },
          ]}
        />
      </Field>

      <Field
        label="Query transform"
        hint="Improve the query before searching. Rewrite: keyword-rich query. HyDE: embed a hypothetical answer. Multi-query: several paraphrases fused with RRF."
      >
        <Segmented
          label="Query transformation"
          value={s.query_transform}
          onChange={(v) => set("query_transform", v)}
          options={[
            { value: "none", label: "None" },
            { value: "rewrite", label: "Rewrite" },
            { value: "hyde", label: "HyDE" },
            { value: "multi_query", label: "Multi" },
          ]}
        />
      </Field>

      <div className="space-y-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
        <ToggleRow
          label="Parent-child expansion"
          hint="Search small child chunks for precision, then give the LLM their larger parent passage for context. Needs a document indexed with the Parent strategy."
          checked={s.parent_child}
          onChange={(v) => set("parent_child", v)}
        />
        <ToggleRow
          label="Contextual compression"
          hint="Keep only the sentences of each passage that relate to the query, cutting prompt tokens."
          checked={s.compression}
          onChange={(v) => set("compression", v)}
        />
      </div>

      <Field
        label="Scope filter"
        hint="Metadata pre-filtering: restrict the search to chosen documents and an inclusive page range BEFORE ranking."
      >
        <div className="space-y-1.5">
          {documents.map((d) => {
            const checked = s.doc_ids === null || s.doc_ids.includes(d.id);
            return (
              <label key={d.id} className="flex cursor-pointer items-center gap-2 text-xs text-slate-300">
                <input type="checkbox" checked={checked} onChange={() => toggleDoc(d.id)} className="accent-indigo-400" />
                <span className="truncate">{d.filename}</span>
              </label>
            );
          })}
          {documents.length === 0 && <p className="text-xs text-slate-500">No documents</p>}
        </div>
        <div className="grid grid-cols-2 gap-2 pt-1">
          <PageInput label="Page from" value={s.page_min} onChange={(v) => set("page_min", v)} />
          <PageInput label="Page to" value={s.page_max} onChange={(v) => set("page_max", v)} />
        </div>
      </Field>

      <button
        onClick={onReset}
        className="w-full rounded-lg border border-white/10 py-2 text-xs font-medium text-slate-400 transition hover:bg-white/5 hover:text-slate-200"
      >
        Reset to defaults
      </button>
    </div>
  );
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
        {label}
        <Hint text={hint} />
      </span>
      <Toggle checked={checked} onChange={onChange} label={label} />
    </div>
  );
}

function PageInput({ label, value, onChange }: { label: string; value: number | null; onChange: (v: number | null) => void }) {
  return (
    <label className="block text-[11px] text-slate-500">
      {label}
      <input
        type="number"
        min={1}
        value={value ?? ""}
        placeholder="any"
        onChange={(e) => onChange(e.target.value === "" ? null : Math.max(1, Number(e.target.value)))}
        className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2 py-1.5 text-sm text-slate-100 outline-none focus:border-indigo-400/60"
      />
    </label>
  );
}
