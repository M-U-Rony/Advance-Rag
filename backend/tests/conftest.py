"""Shared fixtures. Everything runs offline: TF-IDF embeddings and no LLM."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from app.config import REPO_ROOT, Settings
from app.embeddings.tfidf import TfidfEmbedder
from app.ingestion.chunking import ChunkingParams
from app.main import create_app
from app.pipeline import RAGPipeline

PAGES = [
    (
        "Photosynthesis in Plants",
        "Photosynthesis converts sunlight into chemical energy. Chlorophyll inside the "
        "chloroplasts absorbs red and blue light. The plant combines carbon dioxide and water "
        "to produce glucose and oxygen. This process feeds almost every food chain on Earth.",
    ),
    (
        "Volcanic Eruptions",
        "A volcano erupts when magma rises through cracks in the crust. Pressure from dissolved "
        "gases drives explosive eruptions. Basaltic lava flows slowly while rhyolitic magma "
        "produces violent ash clouds. Monitoring seismic tremors helps scientists forecast events.",
    ),
    (
        "Ocean Tides",
        "Tides are caused mainly by the gravity of the Moon pulling on the oceans. Spring tides "
        "occur when the Sun and Moon align, producing the largest tidal range. Neap tides happen "
        "at quarter moons when the pull partially cancels. Coastal harbours schedule ships around "
        "high tide.",
    ),
]


def make_pdf(pages: list[tuple[str, str]] = PAGES) -> bytes:
    """Build a small multi-page PDF in memory."""
    styles = getSampleStyleSheet()
    story: list = []
    for i, (heading, body) in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading1"]))
        story.append(Paragraph(body, styles["BodyText"]))
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4).build(story)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    """A 3-page PDF about photosynthesis, volcanoes and tides."""
    return make_pdf()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings writing to a temporary data dir."""
    return Settings(
        llm_provider="none",
        embedding_backend="tfidf",
        data_dir=tmp_path / "data",
        samples_dir=REPO_ROOT / "samples",
        static_dir=tmp_path / "no-static",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def pipeline(settings: Settings) -> RAGPipeline:
    """Empty offline pipeline."""
    return RAGPipeline(settings, TfidfEmbedder(), None)


@pytest.fixture
def indexed(pipeline: RAGPipeline, pdf_bytes: bytes) -> RAGPipeline:
    """Pipeline with the test PDF indexed using small recursive chunks."""
    pipeline.add_document("nature.pdf", pdf_bytes, ChunkingParams(chunk_size=30, chunk_overlap=5))
    return pipeline


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """HTTP client against a fresh offline app."""
    with TestClient(create_app(settings)) as c:
        yield c
