.PHONY: install dev backend frontend test lint format eval build docker

install:  ## Install backend and frontend dependencies
	cd backend && npm ci
	cd frontend && npm ci

backend:  ## Run the API with auto-reload on :8000
	cd backend && npm run dev

frontend:  ## Run the Vite dev server on :5173 (proxies /api)
	cd frontend && npm run dev

dev:  ## Run backend and frontend together
	$(MAKE) -j2 backend frontend

test:  ## Backend tests with coverage
	cd backend && npm test

lint:  ## eslint + prettier + tsc for the backend, typecheck + eslint for the frontend
	cd backend && npm run lint && npm run format:check && npm run typecheck
	cd frontend && npm run typecheck && npm run lint

format:
	cd backend && npm run format

eval:  ## Recall@k comparison on the sample PDF
	cd backend && npm run eval -- ../samples/rag_handbook.pdf

build:  ## Build the frontend into backend/static for single-process serving
	cd frontend && npm run build && rm -rf ../backend/static && cp -r dist ../backend/static
	cd backend && npm run build

docker:
	docker compose up --build
