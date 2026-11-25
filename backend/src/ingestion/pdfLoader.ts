/**
 * PDF loading: extract text page by page and clean it for chunking.
 *
 * Page numbers are preserved so every chunk can later be cited as `file.pdf, p. 3`.
 * The cleaning step restores paragraph structure that PDF extraction flattens, because
 * the recursive and heading-aware chunkers rely on blank lines between paragraphs.
 */
import { extractText } from "unpdf";
import { InvalidInputError } from "../errors.js";
import type { Page } from "../types.js";

const TERMINAL_PUNCTUATION = [".", "!", "?", ":", ";", ","];
const HEADING_MAX_WORDS = 12;

function isUpperCase(text: string): boolean {
  return text === text.toUpperCase() && text !== text.toLowerCase();
}

/** Heuristic heading detector: short, no trailing punctuation, capitalised or numbered. */
export function isHeading(line: string): boolean {
  const stripped = line.trim();
  const words = stripped.split(/\s+/).filter((w) => w.length > 0);
  if (
    words.length === 0 ||
    words.length > HEADING_MAX_WORDS ||
    TERMINAL_PUNCTUATION.some((p) => stripped.endsWith(p))
  ) {
    return false;
  }
  if (/^\d+(\.\d+)*[.)]?\s+\S/.test(stripped)) return true;
  if (isUpperCase(stripped) && stripped.length > 3) return true;
  const significant = words.filter((w) => w.length > 3);
  return (
    significant.length > 0 &&
    significant.every((w) => isUpperCase(w.charAt(0))) &&
    words.length <= 8
  );
}

/** Normalise whitespace, de-hyphenate, and re-insert paragraph breaks. */
export function cleanPageText(raw: string): string {
  const text = raw
    .replaceAll("\r", "")
    .replaceAll("­", "")
    .replace(/(\w)-\n(\w)/g, "$1$2");
  const lines = text.split("\n").map((ln) => ln.replace(/[ \t]+/g, " ").trim());
  const nonBlank = lines.filter((ln) => ln.length > 0);
  if (nonBlank.length === 0) return "";
  const longest = Math.max(...nonBlank.map((ln) => ln.length));

  // Pass 1: decide where paragraph breaks (empty strings) go.
  const out: string[] = [];
  const endsWithText = () => out.length > 0 && out[out.length - 1] !== "";
  for (const line of lines) {
    if (!line) {
      if (endsWithText()) out.push("");
      continue;
    }
    if (isHeading(line)) {
      if (endsWithText()) out.push("");
      out.push(line, "");
      continue;
    }
    out.push(line);
    if (/[.!?]$/.test(line) && line.length < 0.6 * longest) out.push("");
  }

  // Pass 2: join consecutive non-empty lines into paragraphs.
  const paragraphs: string[] = [];
  let current: string[] = [];
  for (const line of out) {
    if (line === "") {
      if (current.length > 0) {
        paragraphs.push(current.join(" "));
        current = [];
      }
    } else {
      current.push(line);
    }
  }
  if (current.length > 0) paragraphs.push(current.join(" "));
  return paragraphs.join("\n\n");
}

/**
 * Extract non-empty pages from PDF `data`.
 * @throws InvalidInputError if the bytes are not a readable PDF or contain no extractable text.
 */
export async function loadPdf(data: Uint8Array): Promise<Page[]> {
  let pages: Page[];
  try {
    // unpdf copies/loads the bytes itself; pass a fresh Uint8Array because pdf.js detaches its input.
    const { text } = await extractText(new Uint8Array(data), { mergePages: false });
    pages = text.map((t, i) => ({ number: i + 1, text: cleanPageText(t) }));
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new InvalidInputError(`Could not read PDF: ${reason}`);
  }
  pages = pages.filter((p) => p.text.length > 0);
  if (pages.length === 0) {
    throw new InvalidInputError("The PDF contains no extractable text (scanned PDFs need OCR).");
  }
  return pages;
}
