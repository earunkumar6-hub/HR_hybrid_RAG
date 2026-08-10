"""
Folder-based ingestion endpoints. There is no file-upload endpoint —
documents are read directly from backend/data/documents/ (see
rag/ingest.py) and indexed automatically at backend startup. /reindex
exists for ops use (e.g. after dropping new files in without restarting
the server); it is intentionally not wired into the Streamlit UI.
"""
import logging

from fastapi import APIRouter

from models.schemas import DocumentsListResponse, ReindexResponse
from rag.ingest import ingest_folder, list_indexed_documents

logger = logging.getLogger("ingest_router")
router = APIRouter(tags=["ingestion"])


@router.post("/reindex", response_model=ReindexResponse)
async def reindex():
    """Rescans backend/data/documents/ and syncs the DL/EE/RM/GS Chroma
    collections + knowledge graph with whatever's currently on disk."""
    result = ingest_folder()
    return ReindexResponse(**result)


@router.get("/documents", response_model=DocumentsListResponse)
async def documents():
    docs = list_indexed_documents()
    return DocumentsListResponse(documents=docs, total=len(docs))
