"""
Generic multi-collection Chroma wrapper.

The knowledge base is split into four named collections (see
config.COLLECTIONS) rather than one: DL (document chunks), EE (entity
embeddings), RM (relationship embeddings), GS (document summaries). This
module doesn't know anything about *what* goes in each collection — that
logic lives in rag/kg_rag.py (populates EE/RM/GS) and rag/ingest.py
(populates DL). It just provides add/search/delete/stats over "a
collection identified by a key", so callers write
`add_to_store("EE", docs)` / `search_store("EE", query, k)` etc.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import COLLECTIONS, settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )


_STORES: Dict[str, Chroma] = {}


def get_store(key: str) -> Chroma:
    """`key` is one of 'DL' | 'EE' | 'RM' | 'GS' (see config.COLLECTIONS).
    All four live in the same on-disk persistent directory as separate
    named Chroma collections."""
    key = key.upper()
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown collection key '{key}'. Valid: {list(COLLECTIONS)}")
    if key not in _STORES:
        _STORES[key] = Chroma(
            collection_name=COLLECTIONS[key],
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _STORES[key]


def add_to_store(key: str, docs: List[Document]) -> int:
    """Adds/upserts `docs` into the given collection. Each doc's
    metadata MUST include a stable 'id' field the caller controls (e.g.
    "<doc_id>-<chunk_index>" for DL, "ent::<name>" for EE) — Chroma
    upserts on a repeated id, which is how re-ingesting an unchanged file
    or a shared entity across documents stays idempotent instead of
    duplicating rows."""
    if not docs:
        return 0
    store = get_store(key)
    ids = [d.metadata["id"] for d in docs]
    store.add_documents(docs, ids=ids)
    return len(docs)


def search_store(key: str, query: str, k: int) -> List[Tuple[Document, float]]:
    """Returns (document, relevance_score) pairs, highest relevance first.
    Score is derived from Chroma's L2 distance (lower distance -> closer
    to 1.0) purely so callers/UI have an intuitive 0-1-ish number."""
    store = get_store(key)
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]


def delete_from_store(key: str, doc_id: str) -> None:
    """Removes every record in the given collection tagged with this
    doc_id (used when a source file changes or is removed on disk)."""
    store = get_store(key)
    store.delete(where={"doc_id": doc_id})


def store_stats() -> dict:
    stats = {}
    for key, collection_name in COLLECTIONS.items():
        store = get_store(key)
        raw = store.get()
        stats[key] = {"collection": collection_name, "count": len(raw.get("ids", []))}
    return stats
