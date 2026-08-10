"""
Folder-based document ingestion.

Instead of uploading files through the UI, drop PDF/DOCX/TXT/MD files
into backend/data/documents/ (flat — no subfolders needed) and call
`ingest_folder()`. It runs automatically once at backend startup (see
app.py's lifespan handler) and can also be triggered on demand via
POST /reindex.

Idempotent: a manifest file (data/manifest.json) tracks each file's
relative path, size and modified-time. Unchanged files are skipped on
subsequent runs; changed files have their old DL/EE/RM/GS records +
graph nodes removed and are re-ingested; files deleted from disk are
pruned from the index too.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from config import DOCUMENTS_DIR, MANIFEST_FILE
from rag.document_loader import SUPPORTED_SUFFIXES, load_and_chunk
from rag.kg_rag import get_kg_store
from rag.retriever import add_to_store, delete_from_store

logger = logging.getLogger("ingest")


def _doc_id_for(relative_path: str) -> str:
    """Stable id derived from the file's path relative to DOCUMENTS_DIR,
    so re-ingesting an unchanged file always upserts the same records."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def _load_manifest() -> Dict[str, dict]:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_manifest(manifest: Dict[str, dict]) -> None:
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


def _discover_files() -> List[Path]:
    """Every supported file anywhere under DOCUMENTS_DIR (subfolders are
    fine if you want to organise your own source files that way — they
    have no special meaning to the app, unlike the old department
    folders). The folder's own usage README is excluded so it doesn't
    get indexed as if it were one of your documents."""
    if not DOCUMENTS_DIR.exists():
        return []
    return sorted(
        p for p in DOCUMENTS_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name.lower() != "readme.md"
    )


def ingest_folder() -> dict:
    """Scan DOCUMENTS_DIR and bring the DL/EE/RM/GS Chroma collections +
    knowledge graph in sync with what's on disk. Safe to call repeatedly."""
    manifest = _load_manifest()
    kg = get_kg_store()

    on_disk = _discover_files()
    on_disk_relpaths = set()
    added = updated = skipped = removed = failed = 0

    for path in on_disk:
        rel = str(path.relative_to(DOCUMENTS_DIR))
        on_disk_relpaths.add(rel)
        stat = path.stat()
        fingerprint = f"{stat.st_size}-{int(stat.st_mtime)}"
        doc_id = _doc_id_for(rel)

        prior = manifest.get(rel)
        if prior and prior.get("fingerprint") == fingerprint:
            skipped += 1
            continue

        if prior:
            delete_from_store("DL", doc_id)
            kg.remove_document(doc_id)

        try:
            _doc_id, chunks = load_and_chunk(path, path.name, doc_id=doc_id)
            for c in chunks:
                c.metadata["id"] = f"{doc_id}-{c.metadata['chunk_index']}"
            add_to_store("DL", chunks)

            full_text = "\n".join(c.page_content for c in chunks)
            kg.add_document(doc_id, path.name, full_text)

            manifest[rel] = {"doc_id": doc_id, "fingerprint": fingerprint}
            updated += 1 if prior else 0
            added += 0 if prior else 1
        except Exception as exc:
            logger.warning("Failed to ingest '%s': %s", rel, exc)
            manifest.pop(rel, None)
            failed += 1

    # prune files that were removed from disk since the last run
    for rel in list(manifest.keys()):
        if rel not in on_disk_relpaths:
            entry = manifest.pop(rel)
            delete_from_store("DL", entry["doc_id"])
            kg.remove_document(entry["doc_id"])
            removed += 1

    _save_manifest(manifest)
    result = {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "removed": removed,
        "failed": failed,
        "total_indexed": len(manifest),
    }
    logger.info("Folder ingest complete: %s", result)
    return result


def list_indexed_documents() -> List[dict]:
    manifest = _load_manifest()
    return [{"path": rel, **meta} for rel, meta in sorted(manifest.items())]
