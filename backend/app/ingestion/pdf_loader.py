"""PDF loading: extract text page by page and clean it for chunking.

Page numbers are preserved so every chunk can later be cited as ``file.pdf, p. 3``.
The cleaning step restores paragraph structure that PDF extraction flattens, because
the recursive and heading-aware chunkers rely on blank lines between paragraphs.
"""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from app.models import Page

_TERMINAL = (".", "!", "?", ":", ";", ",")
_HEADING_MAX_WORDS = 12


def is_heading(line: str) -> bool:
    """Heuristic heading detector: short, no trailing punctuation, capitalised or numbered."""
    stripped = line.strip()
    words = stripped.split()
    if not words or len(words) > _HEADING_MAX_WORDS or stripped.endswith(_TERMINAL):
        return False
    if re.match(r"^\d+(\.\d+)*[.)]?\s+\S", stripped):
        return True
    if stripped.isupper() and len(stripped) > 3:
        return True
    significant = [w for w in words if len(w) > 3]
    return bool(significant) and all(w[0].isupper() for w in significant) and len(words) <= 8


def clean_page_text(raw: str) -> str:
    """Normalise whitespace, de-hyphenate, and re-insert paragraph breaks."""
    text = raw.replace("\r", "").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    nonblank = [ln for ln in lines if ln]
    if not nonblank:
        return ""
    longest = max(len(ln) for ln in nonblank)
    out: list[str] = []
    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if is_heading(line):
            if out and out[-1] != "":
                out.append("")
            out.extend([line, ""])
            continue
        out.append(line)
        if line.endswith((".", "!", "?")) and len(line) < 0.6 * longest:
            out.append("")
    paragraphs: list[str] = []
    current: list[str] = []
    for line in out:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def load_pdf(data: bytes) -> list[Page]:
    """Extract non-empty pages from PDF ``data``.

    Raises:
        ValueError: if the bytes are not a readable PDF or contain no extractable text.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [
            Page(number=i, text=clean_page_text(p.extract_text() or ""))
            for i, p in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:  # pypdf raises many exception types for malformed files
        raise ValueError(f"Could not read PDF: {exc}") from exc
    pages = [p for p in pages if p.text]
    if not pages:
        raise ValueError("The PDF contains no extractable text (scanned PDFs need OCR).")
    return pages
