# Drop your documents here

PDF, DOCX, TXT, and MD files placed in this folder (subfolders are fine
too — they have no special meaning) are indexed automatically:

- every time the backend starts (`app.py`'s startup hook calls
  `rag.ingest.ingest_folder()`), and
- on demand via `POST /reindex` if you add files while the server is
  already running and don't want to restart it.

Ingestion is idempotent and incremental — unchanged files are skipped on
re-runs, changed files are re-indexed, and files you delete from here are
pruned from the index too (tracked in `backend/data/manifest.json`).

There's no file-upload button in the UI; this folder is the single
source of truth for what's in the knowledge base.
