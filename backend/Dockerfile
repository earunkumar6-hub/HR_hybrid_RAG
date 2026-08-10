# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------
# Backend image: FastAPI + LangChain + Chroma + NetworkX + NeMo Guardrails
# + DeepEval. Multi-stage build keeps the runtime image free of compilers
# and pip caches. Runs as a non-root user in both local and prod use.
# ----------------------------------------------------------------------

# ---- 1. builder: compile/collect Python deps into an isolated prefix ----
FROM python:3.12-slim AS builder

WORKDIR /build

# build-essential is needed for a few packages (e.g. chromadb's deps) that
# ship C extensions without a prebuilt wheel for every platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ---- 2. runtime: slim image, only what's needed to run the app ----
FROM python:3.12-slim AS runtime

# curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY backend/ .

# Data dirs must exist before volumes are mounted over them, and must be
# owned by the non-root user that ultimately writes to them.
RUN mkdir -p data/documents data/chroma data/graph \
    && chown -R appuser:appuser /app /home/appuser/.local

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# NOTE on concurrency: the knowledge-graph store (rag/kg_rag.py) and the
# Chroma embedding/vectorstore singletons are cached in-process and the KG
# graph is persisted as a single JSON file guarded by an in-process lock.
# That lock does NOT coordinate across multiple OS processes, so running
# more than one uvicorn worker (or more than one backend container replica
# writing to the same volume) can race if ingestion runs concurrently in
# more than one process (e.g. two replicas both starting up at once, or
# an admin hitting POST /reindex while another instance is still doing
# its startup ingest) and corrupt or lose graph updates. Keep --workers 1
# / a single replica unless you first move the KG store to something
# that supports real concurrent writers (e.g. a proper graph DB) and
# Chroma to server mode (chromadb's HttpClient against a shared chroma
# server) instead of the embedded persistent client used here. Reads
# (/query, /graph) scale fine behind a load balancer; it's concurrent
# ingestion writes that don't.
# --loop asyncio (instead of uvicorn's default, uvloop): DeepEval's
# synchronous metric.measure() internally uses nest_asyncio to run its
# LLM-judge calls inside whatever event loop is already running.
# nest_asyncio can only patch the standard library's asyncio loop, not
# uvloop's C-extension loop type — with the default uvloop-backed loop
# this fails with "Can't patch loop of type <class 'uvloop.Loop'>" the
# moment /evaluate (or the DeepEval regression suite, if run in-process)
# calls a metric. Losing uvloop costs a little raw throughput; this app
# is I/O-bound on OpenAI calls anyway, so it's not a meaningful tradeoff.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "asyncio"]
