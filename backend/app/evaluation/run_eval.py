"""CLI: compare vector / BM25 / hybrid / hybrid+rerank retrieval on a PDF.

Usage::

    python -m app.evaluation.run_eval ../samples/rag_handbook.pdf   # uses <stem>_qa.json next to it
    python -m app.evaluation.run_eval my.pdf --qa my_qa.json --ks 1 3 5
    EMBEDDING_BACKEND=tfidf python -m app.evaluation.run_eval my.pdf     # force offline baseline
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from app.config import Settings
from app.evaluation.evaluator import auto_generate_qa, evaluate, load_qa
from app.ingestion.chunking import STRATEGIES, ChunkingParams
from app.runtime import build_pipeline
from app.schemas import EvalReport


def format_report(report: EvalReport) -> str:
    """Render the report as a Markdown table."""
    recall_cols = [f"Recall@{k}" for k in report.ks]
    header = ["Config", *recall_cols, f"P@{report.precision_k}", "MRR", "Faithful", "AnsRel", "ms"]
    lines = [
        f"Embedder: {report.embedder} | questions: {report.questions} | QA source: {report.source}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for row in report.rows:
        cells = [
            row.config,
            *(f"{row.recall[f'@{k}']:.2f}" for k in report.ks),
            f"{row.precision:.2f}",
            f"{row.mrr:.2f}",
            f"{row.faithfulness:.2f}",
            f"{row.answer_relevance:.2f}",
            f"{row.avg_ms:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m app.evaluation.run_eval``."""
    parser = argparse.ArgumentParser(description="Recall@k comparison of retrieval configurations")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--qa", type=Path, help="JSON file with question/evidence pairs")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--auto-questions", type=int, default=20)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = build_pipeline(Settings(data_dir=Path(tmp)))
        pipeline.add_pdf_file(args.pdf, ChunkingParams(strategy=args.strategy))
        qa_path = args.qa or args.pdf.with_name(f"{args.pdf.stem}_qa.json")
        if qa_path.exists():
            qa, source = load_qa(qa_path), qa_path.name
        else:
            qa, source = (
                auto_generate_qa(pipeline.index.chunks, args.auto_questions),
                "auto-generated",
            )
        print(format_report(evaluate(pipeline, qa, ks=args.ks, source=source)))


if __name__ == "__main__":
    main()
