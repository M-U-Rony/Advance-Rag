from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def upload(client: TestClient, pdf: bytes, **form: str) -> dict:
    resp = client.post(
        "/api/documents", files={"file": ("nature.pdf", pdf, "application/pdf")}, data=form
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_health_and_config(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["llm"]["label"] == "Offline" and cfg["llm"]["offline"] is True
    assert cfg["embeddings"]["active"] == "tfidf" and cfg["embeddings"]["dense"] is False
    assert cfg["sample_available"] is True


def test_upload_list_delete(client: TestClient, pdf_bytes: bytes) -> None:
    doc = upload(client, pdf_bytes, strategy="fixed", chunk_size="30", chunk_overlap="5")
    assert doc["pages"] == 3 and doc["strategy"] == "fixed" and doc["chunks"] > 0
    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_upload_validation(client: TestClient, pdf_bytes: bytes) -> None:
    bad = client.post("/api/documents", files={"file": ("x.pdf", b"nope", "application/pdf")})
    assert bad.status_code == 422
    overlap = client.post(
        "/api/documents",
        files={"file": ("x.pdf", pdf_bytes, "application/pdf")},
        data={"chunk_size": "20", "chunk_overlap": "30"},
    )
    assert overlap.status_code == 422
    strategy = client.post(
        "/api/documents",
        files={"file": ("x.pdf", pdf_bytes, "application/pdf")},
        data={"strategy": "magic"},
    )
    assert strategy.status_code == 422


def test_upload_size_limit(settings: Settings, pdf_bytes: bytes) -> None:
    tiny = settings.model_copy(update={"max_upload_mb": 0})
    with TestClient(create_app(tiny)) as c:
        resp = c.post("/api/documents", files={"file": ("x.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 413


def test_query_requires_documents(client: TestClient) -> None:
    assert client.post("/api/query", json={"question": "hi"}).status_code == 409
    assert client.post("/api/query/stream", json={"question": "hi"}).status_code == 409
    assert client.post("/api/evaluate", json={}).status_code == 409


def test_query_returns_answer_citations_and_trace(client: TestClient, pdf_bytes: bytes) -> None:
    upload(client, pdf_bytes, chunk_size="30", chunk_overlap="5")
    body = {
        "question": "What causes spring tides?",
        "history": [],
        "settings": {"retrieval_mode": "hybrid", "fusion": "weighted", "alpha": 0.4, "top_k": 3},
    }
    data = client.post("/api/query", json=body).json()
    assert "[1]" in data["answer"]
    assert data["citations"][0]["page"] == 3
    keys = [s["key"] for s in data["trace"]["stages"]]
    assert keys == [
        "transform",
        "vector",
        "bm25",
        "fusion",
        "rerank",
        "parent",
        "compression",
        "generate",
    ]
    assert data["trace"]["settings"]["alpha"] == 0.4


def test_query_validation_errors(client: TestClient, pdf_bytes: bytes) -> None:
    upload(client, pdf_bytes)
    assert client.post("/api/query", json={"question": ""}).status_code == 422
    bad = {"question": "x", "settings": {"retrieval_mode": "magic"}}
    assert client.post("/api/query", json=bad).status_code == 422
    bad_alpha = {"question": "x", "settings": {"alpha": 2}}
    assert client.post("/api/query", json=bad_alpha).status_code == 422


def test_stream_endpoint_sse(client: TestClient, pdf_bytes: bytes) -> None:
    upload(client, pdf_bytes, chunk_size="30", chunk_overlap="5")
    with client.stream("POST", "/api/query/stream", json={"question": "volcano magma"}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())
    events = [e for e in raw.split("\n\n") if e.strip()]
    names = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert names[0] == "trace" and names[-1] == "done"
    done = json.loads(events[-1].split("\n")[1].removeprefix("data: "))
    streamed = "".join(
        json.loads(e.split("\n")[1].removeprefix("data: "))["text"]
        for e in events
        if e.startswith("event: token")
    )
    assert streamed == done["answer"]


def test_sample_load_and_evaluate(client: TestClient) -> None:
    first = client.post("/api/documents/sample")
    assert first.status_code == 201
    assert first.json()["strategy"] == "parent_child" and first.json()["pages"] == 6
    again = client.post("/api/documents/sample").json()
    assert again["id"] == first.json()["id"]
    assert len(client.get("/api/documents").json()) == 1
    report = client.post("/api/evaluate", json={"ks": [1, 3, 5], "max_questions": 12}).json()
    assert report["questions"] == 12 and report["source"] == "rag_handbook_qa.json"
    assert [r["config"] for r in report["rows"]] == ["vector", "bm25", "hybrid", "hybrid + rerank"]
    assert all(0 <= r["recall"]["@5"] <= 1 for r in report["rows"])


def test_evaluate_auto_generates_for_unknown_pdf(client: TestClient, pdf_bytes: bytes) -> None:
    upload(client, pdf_bytes, chunk_size="30", chunk_overlap="5")
    report = client.post("/api/evaluate", json={"ks": [1, 3], "max_questions": 3}).json()
    assert report["source"] == "auto-generated" and report["questions"] >= 1


def test_sample_missing_returns_404(settings: Settings, tmp_path) -> None:
    empty = settings.model_copy(update={"samples_dir": tmp_path / "none"})
    with TestClient(create_app(empty)) as c:
        assert c.post("/api/documents/sample").status_code == 404
        assert c.get("/api/config").json()["sample_available"] is False


def test_serves_built_frontend(settings: Settings, tmp_path) -> None:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>app</html>")
    (static / "assets" / "app.js").write_text("console.log(1)")
    (static / "favicon.svg").write_text("<svg/>")
    with TestClient(create_app(settings.model_copy(update={"static_dir": static}))) as c:
        assert "app" in c.get("/").text
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/favicon.svg").text == "<svg/>"
        assert "app" in c.get("/some/client/route").text
        assert c.get("/api/health").status_code == 200
        assert "app" in c.get("/../../etc/passwd").text
