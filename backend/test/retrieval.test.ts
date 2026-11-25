import { describe, expect, it } from "vitest";
import { BM25Index } from "../src/retrieval/bm25.js";
import { bigramOverlap, compressText, sentenceScores } from "../src/retrieval/compression.js";
import { MetadataFilter } from "../src/retrieval/filters.js";
import { minMaxNormalize, reciprocalRankFusion, weightedFusion } from "../src/retrieval/fusion.js";
import { CrossEncoderReranker, HeuristicReranker } from "../src/retrieval/rerank.js";

function bm25Term(
  tf: number,
  dl: number,
  avgdl: number,
  df: number,
  n: number,
  k1 = 1.2,
  b = 0.75,
) {
  const idf = Math.log((n - df + 0.5) / (df + 0.5) + 1);
  return (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl));
}

describe("BM25", () => {
  it("matches hand-computed values", () => {
    const index = new BM25Index();
    index.index(["apple banana", "apple cherry", "banana banana cherry"]);
    expect(index.avgdl).toBeCloseTo(7 / 3);
    const apple = index.score("apple");
    expect(apple[0]).toBeCloseTo(bm25Term(1, 2, 7 / 3, 2, 3));
    expect(apple[1]).toBeCloseTo(bm25Term(1, 2, 7 / 3, 2, 3));
    expect(apple[2]).toBe(0);
    const banana = index.score("banana");
    expect(banana[2]).toBeCloseTo(bm25Term(2, 3, 7 / 3, 2, 3));
    expect(banana[2] as number).toBeGreaterThan(banana[0] as number);
  });

  it("normalises length and saturates term frequency", () => {
    const index = new BM25Index();
    index.index([
      "rocket " + "filler ".repeat(3),
      "rocket " + "filler ".repeat(40),
      "rocket rocket rocket rocket",
    ]);
    const s = index.score("rocket");
    expect(s[0] as number).toBeGreaterThan(s[1] as number);
    const gain12 = bm25Term(2, 4, 4, 1, 1) - bm25Term(1, 4, 4, 1, 1);
    const gain34 = bm25Term(4, 4, 4, 1, 1) - bm25Term(3, 4, 4, 1, 1);
    expect(gain12).toBeGreaterThan(gain34);
  });

  it("searches with an optional metadata mask", () => {
    const index = new BM25Index();
    index.index(["red fox jumps", "blue whale swims", "red whale rests"]);
    expect(index.search("red whale", 3)[0]?.[0]).toBe(2);
    expect(index.search("red whale", 3, [true, true, false]).map(([i]) => i)).not.toContain(2);
    expect(index.search("unknownterm", 3)).toEqual([]);
  });

  it("handles an empty index", () => {
    const index = new BM25Index();
    index.index([]);
    expect(index.search("anything", 3)).toEqual([]);
  });
});

describe("fusion", () => {
  it("RRF sums 1/(60+rank) over the lists", () => {
    const fused = new Map(
      reciprocalRankFusion([
        ["a", "b", "c"],
        ["b", "a", "d"],
      ]),
    );
    expect(fused.get("a")).toBeCloseTo(1 / 61 + 1 / 62);
    expect(fused.get("b")).toBeCloseTo(1 / 62 + 1 / 61);
    expect(fused.get("c")).toBeCloseTo(1 / 63);
    expect(reciprocalRankFusion([["x", "y"], ["y"]]).map(([id]) => id)).toEqual(["y", "x"]);
  });

  it("RRF accepts a custom k", () => {
    expect(new Map(reciprocalRankFusion([["a"]], 10)).get("a")).toBeCloseTo(1 / 11);
  });

  it("min-max normalises", () => {
    expect(
      minMaxNormalize(
        new Map([
          ["a", 2],
          ["b", 4],
          ["c", 6],
        ]),
      ),
    ).toEqual(
      new Map([
        ["a", 0],
        ["b", 0.5],
        ["c", 1],
      ]),
    );
    expect(minMaxNormalize(new Map([["a", 3]]))).toEqual(new Map([["a", 1]]));
    expect(
      minMaxNormalize(
        new Map([
          ["a", 0],
          ["b", 0],
        ]),
      ),
    ).toEqual(
      new Map([
        ["a", 0],
        ["b", 0],
      ]),
    );
    expect(minMaxNormalize(new Map())).toEqual(new Map());
  });

  it("weighted fusion honours the alpha extremes", () => {
    const vec = new Map([
      ["a", 1],
      ["b", 0.5],
    ]);
    const kw = new Map([
      ["a", 0],
      ["b", 10],
    ]);
    expect(weightedFusion(vec, kw, 1)[0]?.[0]).toBe("a");
    expect(weightedFusion(vec, kw, 0)[0]?.[0]).toBe("b");
    const blended = new Map(weightedFusion(vec, kw, 0.5));
    expect(blended.get("a")).toBeCloseTo(0.5);
    expect(blended.get("b")).toBeCloseTo(0.5);
  });

  it("weighted fusion breaks ties by id and rejects a bad alpha", () => {
    const fused = weightedFusion(
      new Map([
        ["b", 1],
        ["a", 1],
      ]),
      new Map(),
      1,
    );
    expect(fused.map(([id]) => id)).toEqual(["a", "b"]);
    expect(() => weightedFusion(new Map(), new Map(), 1.5)).toThrow(RangeError);
  });
});

describe("heuristic reranker", () => {
  it("prefers relevant text", async () => {
    const ranked = await new HeuristicReranker().rerank("how does reciprocal rank fusion work", [
      ["noise", "Bananas are yellow and grow in tropical climates."],
      ["hit", "Reciprocal rank fusion merges two ranked lists using ranks only."],
      ["partial", "A list of fruit names and ranks."],
    ]);
    expect(ranked[0]?.[0]).toBe("hit");
    expect(ranked[0]?.[1] as number).toBeGreaterThan(ranked.at(-1)?.[1] as number);
  });

  it("scores bigram matches higher and returns 0 for stop-word-only queries", () => {
    const r = new HeuristicReranker();
    expect(r.score("the of a", "anything at all")).toBe(0);
    expect(r.score("vector search", "vector search is fast")).toBeGreaterThan(
      r.score("vector search", "search the vector"),
    );
  });
});

describe("cross-encoder reranker", () => {
  it("falls back to the heuristic when the model cannot be loaded", async () => {
    const reranker = new CrossEncoderReranker(() => Promise.reject(new Error("no such model")));
    expect(reranker.available).toBeNull();
    const ranked = await reranker.rerank("fusion", [
      ["a", "unrelated text"],
      ["b", "rank fusion method"],
    ]);
    expect(ranked[0]?.[0]).toBe("b");
    expect(reranker.available).toBe(false);
    expect(reranker.failure).toContain("no such model");
    expect(reranker.name).toContain("heuristic");
    await reranker.rerank("fusion", []); // second call must not retry the load
  });

  it("uses the model scores once loaded", async () => {
    let loads = 0;
    const reranker = new CrossEncoderReranker(() => {
      loads++;
      return Promise.resolve((_q: string, passages: string[]) =>
        Promise.resolve(passages.map((p) => (p.includes("best") ? 9 : 1))),
      );
    });
    const ranked = await reranker.rerank("q", [
      ["a", "meh"],
      ["b", "the best one"],
    ]);
    expect(ranked).toEqual([
      ["b", 9],
      ["a", 1],
    ]);
    await reranker.rerank("q", [["a", "meh"]]);
    expect(loads).toBe(1);
    expect(reranker.available).toBe(true);
    expect(reranker.name).toBe("cross_encoder");
  });
});

describe("compression", () => {
  it("keeps the query-relevant sentences", () => {
    const text =
      "The museum opens at nine. Bananas are rich in potassium and fibre. " +
      "Tickets cost ten dollars. The cafe sells coffee.";
    expect(compressText("how much potassium do bananas have", text)).toBe(
      "Bananas are rich in potassium and fibre.",
    );
  });

  it("always keeps something", () => {
    expect(compressText("zebra", "First sentence here. Second sentence there.")).toBe(
      "First sentence here.",
    );
    expect(compressText("anything", "Single sentence.")).toBe("Single sentence.");
  });

  it("scores sentences and bigram overlap", () => {
    expect(sentenceScores("the of", ["anything"])).toEqual([0]);
    expect(
      sentenceScores("potassium bananas", ["Bananas are rich in potassium.", "Nothing."]),
    ).toEqual([1, 0]);
    expect(bigramOverlap("reciprocal rank fusion", "Reciprocal rank fusion is neat")).toBe(1);
    expect(bigramOverlap("single", "single")).toBe(0);
  });
});

describe("MetadataFilter", () => {
  const meta = { doc_id: "d1", page: 3, filename: "f.pdf" };

  it("is inactive by default and matches everything", () => {
    const filter = new MetadataFilter();
    expect(filter.active).toBe(false);
    expect(filter.matches(meta)).toBe(true);
  });

  it("filters by document and inclusive page range", () => {
    expect(new MetadataFilter(new Set(["d2"])).matches(meta)).toBe(false);
    expect(new MetadataFilter(new Set(["d1"])).matches(meta)).toBe(true);
    expect(new MetadataFilter(null, 4, null).matches(meta)).toBe(false);
    expect(new MetadataFilter(null, null, 2).matches(meta)).toBe(false);
    expect(new MetadataFilter(null, 3, 3).matches(meta)).toBe(true);
    expect(new MetadataFilter(new Set()).active).toBe(false);
  });
});
