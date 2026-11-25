.PHONY: install dev backend frontend test lint format eval build docker

PY := backend/.venv/bin

install:  ## Install backend (venv) and frontend dependencies
	cd backend && uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
	cd frontend && npm ci

backend:  ## Run the API with auto-reload on :8000
	cd backend && .venv/bin/uvicorn app.main:app --factory --reload --port 8000

frontend:  ## Run the Vite dev server on :5173 (proxies /api)
	cd frontend && npm run dev

dev:  ## Run backend and frontend together
	$(MAKE) -j2 backend frontend

test:  ## Backend tests with coverage
	cd backend && .venv/bin/pytest --cov=app --cov-report=term-missing

lint:  ## ruff + frontend typecheck/lint
	cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check .
	cd frontend && npm run typecheck && npm run lint

format:
	cd backend && .venv/bin/ruff check . --fix && .venv/bin/ruff format .

eval:  ## Recall@k comparison on the sample PDF
	cd backend && .venv/bin/python -m app.evaluation.run_eval ../samples/rag_handbook.pdf

build:  ## Build the frontend into backend/static for single-process serving
	cd frontend && npm run build && rm -rf ../backend/static && cp -r dist ../backend/static

docker:
	docker compose up --build
