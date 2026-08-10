"""
Central configuration for the Internal Knowledge Assistant backend.

All environment-driven settings live here so every module (retriever,
kg_rag, llm_chain, guardrails, eval) reads from a single source of truth.
"""
import os
from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"   # drop source files here; auto-ingested at startup
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_DIR = DATA_DIR / "graph"
GRAPH_FILE = GRAPH_DIR / "knowledge_graph.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"  # tracks what's already indexed, for idempotent re-ingestion

for _p in (DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR, GRAPH_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Chroma collections
# ----------------------------------------------------------------------
# The vector store is split into four collections (all living in the same
# persistent Chroma directory), categorised by what each one embeds
# rather than by any business taxonomy — this is what "DL / EE / RM / GS"
# refers to:
#
#   DL  Document Layer     chunk-level embeddings of the raw ingested
#                          documents (the base RAG layer — what most RAG
#                          systems call "the" vector store).
#   EE  Entity Embeddings   one embedding per unique entity extracted into
#                          the knowledge graph (people, systems, policies,
#                          projects, ...). Lets a query semantically match
#                          an entity even when the wording differs from
#                          the entity's canonical name, instead of relying
#                          on literal substring/token overlap.
#   RM  Relationship        one embedding per extracted relationship
#       Mappings           triple ("X — depends on — Y"), textualised.
#                          Lets a query surface a relevant graph edge
#                          directly, without first having to land on one
#                          of its endpoint entities.
#   GS  Graph Summary       one embedding per document-level summary,
#                          giving a "what is this document broadly about"
#                          layer for thematic/global questions that a
#                          single chunk wouldn't answer well.
#
# See rag/kg_rag.py (builds EE/RM/GS from each document) and
# rag/retriever.py (the generic multi-collection Chroma wrapper).
COLLECTIONS: Dict[str, str] = {
    "DL": "kb_documents",
    "EE": "kb_entities",
    "RM": "kb_relationships",
    "GS": "kb_summaries",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- OpenAI / LLM ----
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ---- Chroma ----
    chroma_persist_dir: str = str(CHROMA_DIR)

    # ---- Retrieval ----
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k_documents: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))     # DL
    top_k_entities: int = int(os.getenv("TOP_K_ENTITIES", "6"))       # EE (graph seeds)
    top_k_relationships: int = int(os.getenv("TOP_K_RELATIONSHIPS", "6"))  # RM
    top_k_summaries: int = int(os.getenv("TOP_K_SUMMARIES", "3"))     # GS
    graph_traversal_depth: int = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "2"))

    # ---- Guardrails ----
    guardrails_enabled: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
    guardrails_config_path: str = str(BASE_DIR / "guardrails" / "config")

    # ---- LangSmith tracing ----
    langsmith_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "internal-knowledge-assistant")
    langsmith_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # ---- Backend server ----
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    backend_base_url: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


settings = Settings()


def configure_langsmith() -> None:
    """Propagate LangSmith settings into the process env so LangChain's
    built-in tracing callbacks pick them up automatically. Call once at
    backend startup, before any chain/LLM objects are constructed."""
    if settings.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
