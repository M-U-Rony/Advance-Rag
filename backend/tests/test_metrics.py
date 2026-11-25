from __future__ import annotations

import pytest

from app.evaluation import metrics
from app.evaluation.evaluator import QAItem, auto_generate_qa, evaluate, relevant_ids
from app.generation.prompts import NO_ANSWER
from app.pipeline import RAGPipeline
from tests.fakes import FakeLLM


def test_recall_precision_mrr() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c", "z"}
    assert metrics.recall_at_k(retrieved, relevant, 2) == 0.0
    assert metrics.recall_at_k(retrieved, relevant, 3) == 1.0
    assert metrics.precision_at_k(retrieved, relevant, 4) == 0.25
    assert metrics.reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["x"], relevant) == 0.0
    assert metrics.recall_at_k([], set(), 3) == 0.0
    assert metrics.precision_at_k([], relevant, 3) == 0.0


def test_faithfulness_overlap_threshold() -> None:
    context = ["Cosine similarity measures the angle between two vectors."]
    grounded = "Cosine similarity measures the angle between vectors."
    hallucinated = "The Eiffel Tower was completed in 1889 by Gustave Eiffel."
    assert metrics.faithfulness_overlap(grounded, context) == 1.0
    assert metrics.faithfulness_overlap(hallucinated, context) == 0.0
    mixed = f"{grounded} {hallucinated}"
    assert metrics.faithfulness_overlap(mixed, context) == 0.5
    assert metrics.faithfulness_overlap(NO_ANSWER, context) == 1.0
    assert metrics.faithfulness_overlap("", context) == 0.0


def test_answer_relevance() -> None:
    assert (
        metrics.answer_relevance("What is cosine similarity?", "Cosine similarity is an angle.")
        == 1.0
    )
    assert metrics.answer_relevance("What is cosine similarity?", "Bananas.") == 0.0
    assert metrics.answer_relevance("What is cosine similarity?", NO_ANSWER) == 0.0


def test_llm_faithfulness_judge() -> None:
    assert metrics.llm_faithfulness("a", ["b"], FakeLLM("0.75")) == 0.75
    assert metrics.llm_faithfulness("a", ["b"], FakeLLM("9")) == 1.0
    assert metrics.llm_faithfulness("a", ["b"], FakeLLM("not a number")) == 0.0


def test_relevant_ids_and_auto_qa(indexed: RAGPipeline) -> None:
    chunks = indexed.index.chunks
    assert relevant_ids(chunks, "Chlorophyll   inside the chloroplasts")
    assert not relevant_ids(chunks, "dinosaurs")
    qa = auto_generate_qa(chunks, 3)
    assert qa
    assert all(relevant_ids(chunks, item.evidence) for item in qa)


def test_evaluate_reports_four_configs(indexed: RAGPipeline) -> None:
    qa = [
        QAItem("What absorbs red and blue light in plants?", "absorbs red and blue light"),
        QAItem("What causes spring tides?", "Sun and Moon align"),
        QAItem("Unanswerable question about nothing", "no such evidence anywhere"),
    ]
    report = evaluate(indexed, qa, ks=[1, 3])
    assert report.questions == 2  # the unlabelled question is dropped
    assert [r.config for r in report.rows] == ["vector", "bm25", "hybrid", "hybrid + rerank"]
    for row in report.rows:
        assert set(row.recall) == {"@1", "@3"}
        assert row.recall["@3"] >= row.recall["@1"]
        assert 0.0 <= row.faithfulness <= 1.0
    assert report.rows[-1].recall["@3"] == 1.0
