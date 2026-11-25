# syntax=docker/dockerfile:1.7

# ---- Stage 1: build the React frontend -------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: compile the TypeScript backend --------------------------------
# Debian ("slim"), not Alpine: onnxruntime-node ships glibc binaries only.
FROM node:22-slim AS backend
# The CPU runtime is bundled with onnxruntime-node; skip its optional CUDA download (~300 MB).
ENV ONNXRUNTIME_NODE_INSTALL=skip
WORKDIR /build
COPY backend/package.json backend/package-lock.json ./
RUN npm ci
COPY backend/tsconfig.json backend/tsconfig.build.json ./
COPY backend/src ./src
RUN npm run build && npm prune --omit=dev \
    && rm -rf node_modules/onnxruntime-node/bin/napi-v6/darwin node_modules/onnxruntime-node/bin/napi-v6/win32

# ---- Stage 3: runtime --------------------------------------------------------
FROM node:22-slim AS runtime
ENV NODE_ENV=production \
    DATA_DIR=/data \
    MODEL_CACHE_DIR=/opt/models \
    STATIC_DIR=/app/backend/static \
    SAMPLES_DIR=/app/samples
WORKDIR /app/backend
COPY --from=backend /build/node_modules ./node_modules
COPY --from=backend /build/dist ./dist
COPY backend/package.json ./
COPY samples /app/samples
COPY --from=frontend /build/dist ./static

# Pre-download the dense embedding and cross-encoder models. Failures (e.g. an offline
# build machine) are tolerated: the app falls back to TF-IDF / heuristic reranking.
RUN node dist/downloadModels.js || true

# Run as the unprivileged `node` user that ships with the base image.
RUN mkdir -p /data /opt/models && chown -R node:node /data /opt/models /app
USER node

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:8000/api/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"
CMD ["node", "dist/server.js"]
