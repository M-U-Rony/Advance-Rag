/** Dense (non-fitted) embedders: vectors are cached and only new chunks are embedded. */
import { existsSync, writeFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { chunkingParams } from "../src/ingestion/chunking.js";
import { HashEmbedder, makePdf, makePipeline, makeSettings, req } from "./helpers.js";

describe("dense vector cache", () => {
  it("embeds incrementally and serves restarts from disk", async () => {
    const settings = makeSettings();
    const emb = new HashEmbedder();
    const pipe = await makePipeline(settings, { embedder: emb });
    const first = await pipe.addDocument(
      "a.pdf",
      await makePdf(),
      chunkingParams({ chunk_size: 30 }),
    );
    const afterFirst = emb.embeddedTexts;
    expect(afterFirst).toBe(first.chunks);

    const otherPdf = await makePdf([
      ["Bees", "Honey bees dance to communicate the location of nectar."],
    ]);
    const second = await pipe.addDocument("b.pdf", otherPdf, chunkingParams({ chunk_size: 30 }));
    expect(emb.embeddedTexts).toBe(afterFirst + second.chunks); // old vectors reused

    await pipe.deleteDocument(first.id);
    expect(emb.embeddedTexts).toBe(afterFirst + second.chunks); // deleting re-embeds nothing
    expect(pipe.index.vectors.size).toBe(second.chunks);

    const reloadedEmb = new HashEmbedder();
    const reloaded = await makePipeline(settings, { embedder: reloadedEmb });
    expect(reloadedEmb.embeddedTexts).toBe(0); // served from the persisted vector store
    const answer = await reloaded.query(req("How do bees communicate nectar location?"));
    expect(answer.citations[0]?.filename).toBe("b.pdf");
    expect(answer.trace.embedder).toBe("hash-dense");
  });

  it("ignores a corrupt cache and re-embeds", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const settings = makeSettings();
    const pipe = await makePipeline(settings, { embedder: new HashEmbedder() });
    await pipe.addDocument("a.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    writeFileSync(path.join(settings.dataDir, "index", "hash-dense", "vectors.bin"), "garbage");
    const emb = new HashEmbedder();
    const reloaded = await makePipeline(settings, { embedder: emb });
    expect(emb.embeddedTexts).toBe(reloaded.index.chunks.length);
    expect(emb.embeddedTexts).toBeGreaterThan(0);
    vi.restoreAllMocks();
  });

  it("uses a model-specific cache directory", async () => {
    const settings = makeSettings();
    const pipe = await makePipeline(settings, { embedder: new HashEmbedder() });
    await pipe.addDocument("a.pdf", await makePdf(), chunkingParams({ chunk_size: 30 }));
    expect(existsSync(path.join(settings.dataDir, "index", "hash-dense", "vectors.json"))).toBe(
      true,
    );
  });
});
