import { mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { createEmbedder, DenseEmbedder } from "../src/embeddings/dense.js";
import { TfidfEmbedder } from "../src/embeddings/tfidf.js";
import { MetadataFilter } from "../src/retrieval/filters.js";
import { VectorStore } from "../src/retrieval/vectorStore.js";
import { makeSettings } from "./helpers.js";

const dot = (a: Float32Array, b: Float32Array) =>
  a.reduce((s, x, i) => s + x * (b[i] as number), 0);

describe("TF-IDF embedder", () => {
  it("uses idf = ln((n+1)/(df+1)) + 1 and L2-normalises", async () => {
    const emb = new TfidfEmbedder();
    emb.fit(["cat dog", "cat bird"]);
    expect(emb.dim).toBe(3);
    const idx = (t: string) => emb.vocab.get(t) as number;
    expect(emb.idf[idx("cat")]).toBeCloseTo(1);
    expect(emb.idf[idx("dog")]).toBeCloseTo(Math.log(3 / 2) + 1);
    const vec = await emb.embedQuery("cat dog");
    expect(Math.hypot(...vec)).toBeCloseTo(1);
    const norm = Math.hypot(1, Math.log(1.5) + 1);
    expect(vec[idx("cat")]).toBeCloseTo(1 / norm);
    expect(vec[idx("dog")]).toBeCloseTo((Math.log(1.5) + 1) / norm);
  });

  it("ignores unknown terms and handles empty input", async () => {
    const emb = new TfidfEmbedder();
    emb.fit(["alpha beta"]);
    expect([...(await emb.embedQuery("zzz"))].every((x) => x === 0)).toBe(true);
    expect(await emb.embedDocuments([])).toEqual([]);
  });

  it("scores similar texts higher", async () => {
    const emb = new TfidfEmbedder();
    const docs = ["red apple pie recipe", "blue ocean waves", "apple pie with cream"];
    emb.fit(docs);
    const q = await emb.embedQuery("apple pie");
    const [a, b, c] = (await emb.embedDocuments(docs)).map((d) => dot(d, q)) as [
      number,
      number,
      number,
    ];
    expect(b).toBe(0);
    expect(a).toBeGreaterThan(0);
    expect(c).toBeGreaterThan(0);
  });

  it("caps the vocabulary at max features, keeping the most frequent terms", () => {
    const emb = new TfidfEmbedder(2);
    emb.fit(["a1 a1 b2", "a1 c3", "b2 zzz"]);
    expect([...emb.vocab.keys()].sort()).toEqual(["a1", "b2"]);
  });
});

describe("DenseEmbedder and createEmbedder", () => {
  it("adds the BGE query prefix only to queries", async () => {
    const seen: string[][] = [];
    const extract = (texts: string[]) => {
      seen.push(texts);
      return Promise.resolve(texts.map(() => new Float32Array([1, 0])));
    };
    const emb = new DenseEmbedder("Xenova/bge-small-en-v1.5", extract);
    expect(emb.name).toBe("bge-small-en-v1.5");
    expect(emb.needsFit).toBe(false);
    emb.fit();
    await emb.embedDocuments(["doc"]);
    await emb.embedQuery("question");
    expect(seen[0]).toEqual(["doc"]);
    expect(seen[1]?.[0]).toMatch(
      /^Represent this sentence for searching relevant passages: question$/,
    );
    expect(await emb.embedDocuments([])).toEqual([]);
    const plain = new DenseEmbedder("org/other-model", extract);
    await plain.embedQuery("q");
    expect(seen.at(-1)).toEqual(["q"]);
  });

  it("creates TF-IDF when requested", async () => {
    const { embedder, fallbackReason } = await createEmbedder(makeSettings());
    expect(embedder.name).toBe("tfidf");
    expect(fallbackReason).toBeNull();
  });

  it("falls back to TF-IDF when the dense model cannot be loaded", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const settings = makeSettings({ embeddingBackend: "dense", embeddingModel: "does-not/exist" });
    const { embedder, fallbackReason } = await createEmbedder(settings, () =>
      Promise.reject(new Error("could not load")),
    );
    expect(embedder.name).toBe("tfidf");
    expect(fallbackReason).toContain("could not load");
    warn.mockRestore();
  });

  it("returns the dense embedder when loading works", async () => {
    const fake = new DenseEmbedder("x/bge-test", () => Promise.resolve([]));
    const { embedder, fallbackReason } = await createEmbedder(
      makeSettings({ embeddingBackend: "auto" }),
      () => Promise.resolve(fake),
    );
    expect(embedder).toBe(fake);
    expect(fallbackReason).toBeNull();
  });
});

const ROWS = [new Float32Array([1, 0]), new Float32Array([0, 1]), new Float32Array([2, 0])];
const META = [
  { doc_id: "d1", page: 1, filename: "a" },
  { doc_id: "d1", page: 5, filename: "a" },
  { doc_id: "d2", page: 2, filename: "b" },
];
const Q = new Float32Array([1, 0]);

function store(): VectorStore {
  const s = new VectorStore();
  s.add(["a", "b", "c"], ROWS, META);
  return s;
}

describe("VectorStore", () => {
  it("scores with cosine, dot and L2", () => {
    const s = store();
    const scores = (metric: "cosine" | "dot" | "l2") => new Map(s.search(Q, 3, metric));
    expect(scores("cosine").get("a")).toBeCloseTo(1);
    expect(scores("cosine").get("b")).toBeCloseTo(0);
    expect(scores("cosine").get("c")).toBeCloseTo(1);
    expect(scores("dot").get("c")).toBeCloseTo(2);
    expect(scores("l2").get("b")).toBeCloseTo(-Math.sqrt(2));
    expect(scores("l2").get("c")).toBeCloseTo(-1);
  });

  it("orders results by the chosen metric", () => {
    const s = store();
    expect(s.search(Q, 3, "dot").map(([id]) => id)).toEqual(["c", "a", "b"]);
    expect(s.search(Q, 3, "l2").map(([id]) => id)).toEqual(["a", "c", "b"]);
    expect(s.search(Q, 0)).toEqual([]);
    expect(new VectorStore().search(Q, 3)).toEqual([]);
  });

  it("handles zero vectors under cosine", () => {
    const s = new VectorStore();
    s.add(["z"], [new Float32Array([0, 0])], [META[0] as (typeof META)[0]]);
    expect(s.search(new Float32Array([0, 0]), 1)).toEqual([["z", 0]]);
  });

  it("pre-filters by metadata", () => {
    const s = store();
    const ids = (f: MetadataFilter) => new Set(s.search(Q, 5, "cosine", f).map(([id]) => id));
    expect(ids(new MetadataFilter(new Set(["d1"])))).toEqual(new Set(["a", "b"]));
    expect(ids(new MetadataFilter(null, 2, 5))).toEqual(new Set(["b", "c"]));
    expect(ids(new MetadataFilter(new Set(["zzz"])))).toEqual(new Set());
  });

  it("validates inputs on add", () => {
    expect(() => new VectorStore().add(["a"], ROWS, [])).toThrow(/same length/);
    const s = store();
    expect(() =>
      s.add(["d"], [new Float32Array([1, 2, 3])], [META[0] as (typeof META)[0]]),
    ).toThrow(/dimension/);
    s.add([], [], []); // no-op
    expect(s.size).toBe(3);
  });

  it("round-trips through disk", () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "vs-"));
    const s = store();
    s.save(dir);
    const loaded = VectorStore.load(dir);
    expect(loaded.ids).toEqual(s.ids);
    expect(loaded.metadata).toEqual(s.metadata);
    expect([...loaded.matrix]).toEqual([...s.matrix]);
    expect(loaded.search(Q, 2)).toEqual(s.search(Q, 2));
    expect([...loaded.row(2)]).toEqual([2, 0]);
  });

  it("rejects a truncated cache file", () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "vs-"));
    store().save(dir);
    writeFileSync(path.join(dir, "vectors.bin"), Buffer.alloc(4));
    expect(() => VectorStore.load(dir)).toThrow(/corrupt/);
  });
});
