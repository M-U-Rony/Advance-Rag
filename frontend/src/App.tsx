import { BarChart3, Menu, MessageSquare, PanelRightClose, PanelRightOpen, RotateCcw } from "lucide-react";
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chat from "./components/Chat";
import Inspector from "./components/Inspector";
import SettingsPanel from "./components/SettingsPanel";
import Sidebar from "./components/Sidebar";
import { cx } from "./components/ui";
import { api, streamQuery } from "./lib/api";
import { DEFAULT_SETTINGS } from "./lib/defaults";
import type { AppConfig, ChatMessage, DocumentInfo, RetrievalSettings } from "./lib/types";

const EvaluateView = lazy(() => import("./components/EvaluateView"));

type View = "chat" | "evaluate";
type RightTab = "pipeline" | "settings";

const SETTINGS_KEY = "advance-rag:settings:v1";

function loadSettings(): RetrievalSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

let counter = 0;
const uid = () => `${Date.now()}-${counter++}`;

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [settings, setSettings] = useState<RetrievalSettings>(loadSettings);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>("chat");
  const [rightTab, setRightTab] = useState<RightTab>("pipeline");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const refreshDocuments = useCallback(async () => {
    const docs = await api.documents();
    setDocuments(docs);
    setSettings((s) => {
      if (!s.doc_ids) return s;
      const alive = s.doc_ids.filter((id) => docs.some((d) => d.id === id));
      return { ...s, doc_ids: alive.length > 0 ? alive : null };
    });
  }, []);

  useEffect(() => {
    void api.config().then(setConfig).catch(() => undefined);
    void refreshDocuments().catch(() => undefined);
  }, [refreshDocuments]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  const patch = (id: string, update: (m: ChatMessage) => ChatMessage) =>
    setMessages((all) => all.map((m) => (m.id === id ? update(m) : m)));

  const send = useCallback(
    async (question: string) => {
      const history = messages
        .filter((m) => !m.error && !m.streaming && m.content)
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));
      const answerId = uid();
      setMessages((all) => [
        ...all,
        { id: uid(), role: "user", content: question },
        { id: answerId, role: "assistant", content: "", streaming: true },
      ]);
      setSelectedId(answerId);
      setBusy(true);
      abort.current = new AbortController();
      const effective = {
        ...settings,
        doc_ids: settings.doc_ids && settings.doc_ids.length > 0 ? settings.doc_ids : null,
      };
      try {
        await streamQuery(
          question,
          history,
          effective,
          {
            onTrace: (trace) => patch(answerId, (m) => ({ ...m, trace })),
            onToken: (text) => patch(answerId, (m) => ({ ...m, content: m.content + text })),
            onDone: (res) =>
              patch(answerId, (m) => ({
                ...m,
                content: res.answer,
                citations: res.citations,
                trace: res.trace,
                streaming: false,
              })),
          },
          abort.current.signal,
        );
      } catch (e) {
        const stopped = e instanceof DOMException && e.name === "AbortError";
        patch(answerId, (m) => ({
          ...m,
          streaming: false,
          error: !stopped,
          content: stopped ? m.content : `Something went wrong: ${e instanceof Error ? e.message : "unknown error"}`,
        }));
      } finally {
        patch(answerId, (m) => ({ ...m, streaming: false }));
        setBusy(false);
      }
    },
    [messages, settings],
  );

  const loadSample = async () => {
    await api.loadSample();
    await refreshDocuments();
  };

  const selectedTrace = useMemo(() => {
    const explicit = messages.find((m) => m.id === selectedId)?.trace;
    if (explicit) return explicit;
    return [...messages].reverse().find((m) => m.trace)?.trace ?? null;
  }, [messages, selectedId]);

  return (
    <div className="bg-aurora flex h-full overflow-hidden text-slate-200">
      <div
        className={cx(
          "fixed inset-y-0 left-0 z-40 transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <Sidebar documents={documents} config={config} onChanged={refreshDocuments} onClose={() => setSidebarOpen(false)} />
      </div>
      {sidebarOpen && (
        <button
          aria-label="Close sidebar"
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-3 border-b border-white/10 bg-ink-900/60 px-4 py-3 backdrop-blur">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              className="rounded-lg p-2 text-slate-400 hover:bg-white/5 lg:hidden"
            >
              <Menu size={18} />
            </button>
            <nav className="flex gap-1 rounded-xl bg-black/30 p-1" aria-label="Views">
              {(
                [
                  ["chat", "Chat", MessageSquare],
                  ["evaluate", "Evaluate", BarChart3],
                ] as const
              ).map(([id, label, Icon]) => (
                <button
                  key={id}
                  onClick={() => setView(id)}
                  aria-current={view === id ? "page" : undefined}
                  className={cx(
                    "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-indigo-400/70",
                    view === id ? "bg-white/10 text-white" : "text-slate-400 hover:text-slate-200",
                  )}
                >
                  <Icon size={15} /> {label}
                </button>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <button
                onClick={() => {
                  abort.current?.abort();
                  setMessages([]);
                  setSelectedId(null);
                }}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-slate-400 transition hover:bg-white/5 hover:text-slate-200"
              >
                <RotateCcw size={14} /> New chat
              </button>
            )}
            <button
              onClick={() => setInspectorOpen((v) => !v)}
              aria-label={inspectorOpen ? "Hide pipeline inspector" : "Show pipeline inspector"}
              aria-pressed={inspectorOpen}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-white/5 hover:text-slate-100"
            >
              {inspectorOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          </div>
        </header>

        {view === "chat" ? (
          <Chat
            messages={messages}
            hasDocuments={documents.length > 0}
            busy={busy}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id);
              setRightTab("pipeline");
              setInspectorOpen(true);
            }}
            onSend={(q) => void send(q)}
            onStop={() => abort.current?.abort()}
            onLoadSample={() => void loadSample()}
            sampleAvailable={config?.sample_available ?? false}
          />
        ) : (
          <Suspense fallback={<div className="p-8 text-sm text-slate-500">Loading...</div>}>
            <EvaluateView hasDocuments={documents.length > 0} />
          </Suspense>
        )}
      </main>

      <aside
        aria-label="Pipeline inspector"
        className={cx(
          "z-30 flex h-full w-[24rem] max-w-full shrink-0 flex-col border-l border-white/10 bg-ink-900/95 backdrop-blur transition-all",
          "fixed inset-y-0 right-0 xl:static",
          inspectorOpen ? "translate-x-0" : "hidden",
        )}
      >
        <div className="flex gap-1 border-b border-white/10 p-3">
          {(
            [
              ["pipeline", "Pipeline"],
              ["settings", "Retrieval settings"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setRightTab(id)}
              aria-pressed={rightTab === id}
              className={cx(
                "flex-1 rounded-lg px-3 py-2 text-xs font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-indigo-400/70",
                rightTab === id ? "bg-white/10 text-white" : "text-slate-400 hover:text-slate-200",
              )}
            >
              {label}
            </button>
          ))}
          <button
            onClick={() => setInspectorOpen(false)}
            aria-label="Close inspector"
            className="rounded-lg p-2 text-slate-400 hover:bg-white/5 xl:hidden"
          >
            <PanelRightClose size={16} />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {rightTab === "pipeline" ? (
            <Inspector trace={selectedTrace} />
          ) : (
            <SettingsPanel
              settings={settings}
              documents={documents}
              onChange={setSettings}
              onReset={() => setSettings(DEFAULT_SETTINGS)}
            />
          )}
        </div>
      </aside>
    </div>
  );
}
