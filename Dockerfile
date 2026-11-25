# syntax=docker/dockerfile:1.7

# ---- Stage 1: build the React frontend -------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime -------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    STATIC_DIR=/app/backend/static
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app/backend
COPY backend/pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Pre-download the dense embedding and cross-encoder models. Failures (e.g. an offline
# build machine) are tolerated: the app falls back to TF-IDF / heuristic reranking.
RUN python - <<'PY' || true
try:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    TextEmbedding("BAAI/bge-small-en-v1.5")
    TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    print("models cached")
except Exception as exc:
    print("model pre-download skipped:", exc)
PY

COPY backend/app ./app
COPY samples /app/samples
COPY --from=frontend /build/dist ./static

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /opt/fastembed /app
USER appuser

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
