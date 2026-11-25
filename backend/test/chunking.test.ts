import { describe, expect, it } from "vitest";
import {
  chunkDocument,
  chunkingParams,
  fixedSizeChunks,
  recursiveChunks,
  semanticChunks,
  splitSections,
  validateChunkingParams,
  type Strategy,
} from "../src/ingestion/chunking.js";
import { wordCount, words } from "../src/text.js";
import type { Page } from "../src/types.js";

describe("fixed-size chunks", () => {
  it("slides a window with overlap", () => {
    const text = Array.from({ length: 10 }, (_, i) => `w${i}`).join(" ");
    expect(fixedSizeChunks(text, 4, 1)).toEqual(["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]);
  });

  it("handles short and empty text", () => {
    expect(fixedSizeChunks("a b c", 10, 2)).toEqual(["a b c"]);
    expect(fixedSizeChunks("", 10, 2)).toEqual([]);
  });

  it("always advances even if overlap >= size", () => {
    expect(fixedSizeChunks("a b c d", 2, 5)).toEqual(["a b", "b c", "c d"]);
  });
});

describe("recursive chunks", () => {
  it("respects size and paragraph boundaries", () => {
    const para1 = "Alpha one. Alpha two. Alpha three.";
    const para2 = "Beta one. Beta two.";
    const chunks = recursiveChunks(`${para1}\n\n${para2}`, 6);
    expect(chunks.every((c) => wordCount(c) <= 6)).toBe(true);
    expect(chunks.join(" ").replaceAll("  ", " ")).toBe(`${para1} ${para2}`);
  });

  it("splits an overlong sentence by words", () => {
    const sentence = Array(25).fill("word").join(" ") + ".";
    const chunks = recursiveChunks(sentence, 10);
    expect(chunks).toHaveLength(3);
    expect(chunks.every((c) => wordCount(c) <= 10)).toBe(true);
  });

  it("ignores blank paragraphs", () => {
    expect(recursiveChunks("\n\n  \n\n", 10)).toEqual([]);
  });
});

describe("sections and semantic chunks", () => {
  it("detects headings", () => {
    const text = "Intro Section\n\nBody of intro here.\n\nSecond Section\n\nBody two here.";
    expect(splitSections(text)).toEqual([
      ["Intro Section", "Body of intro here."],
      ["Second Section", "Body two here."],
    ]);
  });

  it("keeps text before the first heading and skips blank paragraphs", () => {
    expect(splitSections("Preamble text goes here.\n\n \n\nNext Heading\n\nBody.")).toEqual([
      [null, "Preamble text goes here."],
      ["Next Heading", "Body."],
    ]);
  });

  it("prefixes the heading onto each semantic chunk", () => {
    const text = "Intro Section\n\nBody of intro here.";
    expect(semanticChunks(text, 50)).toEqual([
      ["Intro Section", "Intro Section. Body of intro here."],
    ]);
    expect(semanticChunks("Plain body text only.", 50)).toEqual([[null, "Plain body text only."]]);
  });
});

function pages(): Page[] {
  const body = Array.from(
    { length: 30 },
    (_, i) => `Sentence number ${i} talks about topic ${i}.`,
  ).join(" ");
  return [
    { number: 1, text: "Title Here\n\n" + body },
    { number: 2, text: "Another Heading\n\nShort page text here." },
  ];
}

describe("chunkDocument", () => {
  it.each<Strategy>(["fixed", "recursive", "semantic"])(
    "%s keeps metadata consistent",
    (strategy) => {
      const result = chunkDocument(
        pages(),
        "d1",
        "f.pdf",
        chunkingParams({ strategy, chunk_size: 40, chunk_overlap: 5 }),
      );
      expect(result.parents.size).toBe(0);
      expect(result.chunks.length).toBeGreaterThan(2);
      expect(result.chunks.map((c) => c.position)).toEqual(result.chunks.map((_, i) => i));
      expect(new Set(result.chunks.map((c) => c.page))).toEqual(new Set([1, 2]));
      expect(result.chunks.every((c) => c.doc_id === "d1" && c.filename === "f.pdf")).toBe(true);
      expect(new Set(result.chunks.map((c) => c.id)).size).toBe(result.chunks.length);
    },
  );

  it("semantic chunks remember their heading", () => {
    const result = chunkDocument(pages(), "d", "f.pdf", chunkingParams({ strategy: "semantic" }));
    expect(result.chunks.map((c) => c.heading)).toContain("Another Heading");
  });

  it("parent-child maps every child to a parent on the same page", () => {
    const result = chunkDocument(
      pages(),
      "d1",
      "f.pdf",
      chunkingParams({ strategy: "parent_child", parent_size: 60, child_size: 15 }),
    );
    expect(result.parents.size).toBeGreaterThan(0);
    for (const child of result.chunks) {
      const parent = result.parents.get(child.parent_id ?? "");
      expect(parent).toBeDefined();
      expect(parent?.page).toBe(child.page);
      expect(wordCount(child.text)).toBeLessThanOrEqual(15);
      expect(parent?.text).toContain(words(child.text)[0] as string);
    }
    expect(
      Math.max(...[...result.parents.values()].map((p) => wordCount(p.text))),
    ).toBeLessThanOrEqual(60);
    expect(result.chunks.length).toBeGreaterThan(result.parents.size);
    expect(result.chunks[0]?.id).toBe("d1-c0");
    expect([...result.parents.keys()][0]).toBe("d1-p0");
  });
});

describe("validateChunkingParams", () => {
  it.each([
    { strategy: "nope" },
    { chunk_size: 3 },
    { chunk_overlap: 500 },
    { chunk_overlap: -1 },
    { parent_size: 10, child_size: 20 },
  ])("rejects %j", (bad) => {
    expect(() => validateChunkingParams(chunkingParams(bad as never))).toThrow();
  });

  it("accepts the defaults", () => {
    expect(() => validateChunkingParams(chunkingParams())).not.toThrow();
  });
});
