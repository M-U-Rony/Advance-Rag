import { describe, expect, it } from "vitest";
import { cleanPageText, isHeading, loadPdf } from "../src/ingestion/pdfLoader.js";
import {
  contentTerms,
  estimateTokens,
  pairwise,
  round,
  snippet,
  splitSentences,
  tokenize,
  wordCount,
} from "../src/text.js";
import { makePdf } from "./helpers.js";

describe("text utilities", () => {
  it("tokenize lowercases and drops punctuation", () => {
    expect(tokenize("Hello, World! It's k-means 2024.")).toEqual([
      "hello",
      "world",
      "it's",
      "k-means",
      "2024",
    ]);
  });

  it("content_terms removes stop words", () => {
    expect(contentTerms("What is the capital of France")).toEqual(["capital", "france"]);
  });

  it("splits sentences on punctuation followed by a capital", () => {
    expect(splitSentences("One is here. Two is there! Is three? Yes.")).toEqual([
      "One is here.",
      "Two is there!",
      "Is three?",
      "Yes.",
    ]);
    expect(splitSentences("   ")).toEqual([]);
  });

  it("snippet truncates with an ellipsis", () => {
    expect(snippet("a ".repeat(300), 20).endsWith("…")).toBe(true);
    expect(snippet("short")).toBe("short");
  });

  it("estimates more tokens for more words", () => {
    expect(estimateTokens("one two three four")).toBeGreaterThan(estimateTokens("one two"));
    expect(estimateTokens("")).toBe(1);
    expect(wordCount("  a  b\nc ")).toBe(3);
  });

  it("round and pairwise behave as documented", () => {
    expect(round(0.123456, 3)).toBe(0.123);
    expect(pairwise([1, 2, 3])).toEqual([
      [1, 2],
      [2, 3],
    ]);
    expect(pairwise([1])).toEqual([]);
  });
});

describe("isHeading", () => {
  it.each([
    ["Reciprocal Rank Fusion", true],
    ["1.2 Chunking strategies", true],
    ["INTRODUCTION", true],
    ["This is a normal sentence.", false],
    ["a short line without a period but lowercase", false],
    ["Heading ending with colon:", false],
    ["", false],
    ["One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen", false],
  ])("%j -> %s", (line, expected) => {
    expect(isHeading(line)).toBe(expected);
  });
});

describe("cleanPageText", () => {
  it("restores paragraphs and de-hyphenates", () => {
    const raw =
      "Main Heading\n" +
      "This paragraph is long enough to fill an entire line of the page easily and\n" +
      "continues with a hyphen-\nated word and ends here.\n" +
      "Short end.\n" +
      "Next paragraph starts here.";
    const paragraphs = cleanPageText(raw).split("\n\n");
    expect(paragraphs[0]).toBe("Main Heading");
    expect(paragraphs[1]).toContain("hyphenated");
    expect(paragraphs.at(-1)).toBe("Next paragraph starts here.");
  });

  it("returns an empty string for blank pages and keeps single blank lines", () => {
    expect(cleanPageText("  \n \n")).toBe("");
    expect(cleanPageText("First line here\n\n\n\nSecond line here")).toBe(
      "First line here Second line here".replace(" Second", "\n\nSecond"),
    );
  });
});

describe("loadPdf", () => {
  it("returns numbered pages", async () => {
    const pages = await loadPdf(await makePdf());
    expect(pages.map((p) => p.number)).toEqual([1, 2, 3]);
    expect(pages[0]?.text).toContain("Photosynthesis");
    expect(pages[2]?.text).toContain("Moon");
  });

  it("rejects garbage", async () => {
    await expect(loadPdf(new TextEncoder().encode("not a pdf"))).rejects.toThrow(
      /Could not read PDF/,
    );
  });

  it("rejects PDFs without extractable text", async () => {
    await expect(loadPdf(await makePdf([["", ""]]))).rejects.toThrow(/no extractable text/);
  });
});
