import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import COLLECTIONS, configure_langsmith, settings
from routers.evaluate import router as evaluate_router
from routers.graph import router as graph_router
from routers.ingest import router as ingest_router
from routers.query import router as query_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
configure_langsmith()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-ingest whatever's in backend/data/documents/ every time the
    # backend starts, so "the project folder is the source of truth" is
    # true without any manual step. Safe/idempotent to run on every boot
    # (see rag/ingest.py's manifest-based skip-if-unchanged logic).
    logger.info("Startup: scanning backend/data/documents/ for new/changed files...")
    try:
        from rag.ingest import ingest_folder

        result = ingest_folder()
        logger.info("Startup ingestion result: %s", result)
    except Exception:
        logger.exception("Startup ingestion failed; backend will still start.")
    yield


app = FastAPI(
    title="Internal Knowledge Assistant",
    description=(
        "RAG + Knowledge-Graph internal knowledge assistant. "
        "Stack: FastAPI, LangChain, Chroma (DL/EE/RM/GS collections), "
        "NetworkX, NeMo Guardrails, DeepEval, LangSmith."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the Streamlit origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(graph_router)
app.include_router(evaluate_router)


@app.get("/health", tags=["meta"])
async def health():
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "guardrails_enabled": settings.guardrails_enabled,
        "langsmith_tracing": settings.langsmith_tracing,
        "collections": COLLECTIONS,
    }


@app.get("/", tags=["meta"])
async def root():
    return {"message": "HR-Internal Knowledge Assistant API. See /docs for the OpenAPI UI."}
