# ── OSS-RAG FastAPI App ──────────────────────────────────────────────────────
# Multi-stage build: keeps final image lean by separating dependency install
# from the actual app layer.
#
# Build:  docker build -t oss-rag .
# Run:    docker run -p 8080:8080 --env-file .env oss-rag
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim AS base

# System deps needed by some Python packages (chromadb, onnxruntime, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker cache layer is reused when only code changes
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# Install the package itself (setup.py)
RUN pip install --no-cache-dir -e .

# ── Runtime config ────────────────────────────────────────────────────────────
# Cloud Run injects PORT env var (default 8080)
# 0.0.0.0 required — localhost won't be reachable from outside the container
EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
