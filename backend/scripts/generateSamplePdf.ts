/**
 * Generate `samples/rag_handbook.pdf`: a short, real-content PDF about RAG itself.
 *
 * Run from `backend/`:  `npm run sample-pdf`
 */
import { createWriteStream, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import PDFDocument from "pdfkit";
import { PAGES, TITLE, type Block } from "./handbookContent.js";

const CM = 28.3465; // PDF points per centimetre

const STYLES: Record<Block[0], { font: string; size: number; before: number; after: number }> = {
  h1: { font: "Helvetica-Bold", size: 20, before: 0, after: 14 },
  h2: { font: "Helvetica-Bold", size: 14, before: 10, after: 6 },
  p: { font: "Helvetica", size: 11, before: 0, after: 8 },
};

/** Render the handbook to `target` (one PDF page per entry of {@link PAGES}). */
export async function build(target: string): Promise<void> {
  mkdirSync(path.dirname(target), { recursive: true });
  const doc = new PDFDocument({
    size: "A4",
    margin: 2.2 * CM,
    info: { Title: TITLE, Author: "Advance RAG" },
  });
  const out = createWriteStream(target);
  doc.pipe(out);
  PAGES.forEach((page, number) => {
    if (number > 0) doc.addPage();
    for (const [kind, text] of page) {
      const style = STYLES[kind];
      doc.font(style.font).fontSize(style.size);
      doc.moveDown(0).y += style.before;
      doc.text(text, { lineGap: kind === "p" ? 5 : 2, align: "left" });
      doc.y += style.after;
    }
  });
  doc.end();
  await new Promise<void>((resolve, reject) => {
    out.on("finish", resolve);
    out.on("error", reject);
  });
}

const here = path.dirname(fileURLToPath(import.meta.url));
const target = path.resolve(here, "../../samples/rag_handbook.pdf");
await build(target);
console.log(`wrote ${target}`);
