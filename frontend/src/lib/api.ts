import type {
  AppConfig,
  ChunkingOptions,
  Citation,
  DocumentInfo,
  EvalReport,
  QueryResponse,
  RetrievalSettings,
  Trace,
} from "./types";

export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  config: () => fetch("/api/config").then((r) => parse<AppConfig>(r)),
  documents: () => fetch("/api/documents").then((r) => parse<DocumentInfo[]>(r)),
  loadSample: () => fetch("/api/documents/sample", { method: "POST" }).then((r) => parse<DocumentInfo>(r)),
  remove: (id: string) => fetch(`/api/documents/${id}`, { method: "DELETE" }).then((r) => parse<void>(r)),
  evaluate: (ks: number[]) =>
    fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ks, max_questions: 30 }),
    }).then((r) => parse<EvalReport>(r)),

  /** Upload with progress reporting (fetch cannot report upload progress). */
  upload(file: File, options: ChunkingOptions, onProgress: (fraction: number) => void) {
    return new Promise<DocumentInfo>((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      Object.entries(options).forEach(([k, v]) => form.append(k, String(v)));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/documents");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        let body: { detail?: unknown } & Partial<DocumentInfo> = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as DocumentInfo);
        else reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(form);
    });
  },
};

export interface StreamHandlers {
  onTrace: (trace: Trace) => void;
  onToken: (text: string) => void;
  onDone: (response: QueryResponse) => void;
}

/** POST a question and consume the server-sent event stream. */
export async function streamQuery(
  question: string,
  history: HistoryTurn[],
  settings: RetrievalSettings,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history, settings }),
    signal,
  });
  if (!res.ok || !res.body) await parse<never>(res);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(block, handlers);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function dispatch(block: string, h: StreamHandlers): void {
  const event = /^event: (.+)$/m.exec(block)?.[1];
  const data = /^data: (.+)$/m.exec(block)?.[1];
  if (!event || !data) return;
  const payload = JSON.parse(data);
  if (event === "trace") h.onTrace(payload as Trace);
  else if (event === "token") h.onToken((payload as { text: string }).text);
  else if (event === "done") h.onDone(payload as QueryResponse);
}

export type { Citation };
