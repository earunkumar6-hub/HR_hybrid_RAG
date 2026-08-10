# Internal Knowledge Assistant

A RAG + Knowledge-Graph internal knowledge assistant for corporate document
Q&A, with guardrails and automated evaluation built in.

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Vector DB | ChromaDB (persistent, on-disk, 4 collections — see below), OpenAI `text-embedding-3-small` |
| Backend | FastAPI + LangChain |
| Frontend | Streamlit (chat, knowledge-graph viewer, evaluation) |
| Knowledge Graph RAG | NetworkX (entities/relationships extracted per document) |
| Guardrails | NeMo Guardrails (LLM self-check input/output rails), with an automatic lightweight rule-based fallback |
| Evaluation | DeepEval (Answer Relevancy, Faithfulness, Contextual Relevancy/Precision/Recall) |
| Tracing | LangSmith |

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────────────────────────────────┐
│  Streamlit  │ ───────────────▶│                  FastAPI                    │
│  frontend   │◀─────────────── │  /query  /graph  /reindex  /evaluate         │
└─────────────┘                 └───────────────┬────────────────────────────┘
                                                  │
                     ┌────────────────────────────┼─────────────────────────┐
                     ▼                             ▼                         ▼
             ┌───────────────┐            ┌─────────────────┐      ┌────────────────┐
             │  Guardrails    │            │   RAG chain      │      │  DeepEval       │
             │  (NeMo /       │            │   (LangChain +    │      │  metrics runner │
             │  fallback)     │            │   gpt-4o-mini)    │      └────────────────┘
             └───────────────┘            └────────┬─────────┘
                                                     │
                          ┌──────────────┬───────────┼───────────┬──────────────┐
                          ▼              ▼            ▼           ▼              │
                     ┌────────┐    ┌─────────┐  ┌──────────┐ ┌─────────┐        │
                     │  DL     │    │  EE      │  │  RM       │ │  GS      │        │
                     │  chunks │    │  entities│  │  relations│ │ summaries│        │
                     └────────┘    └─────────┘  └──────────┘ └─────────┘        │
                          └──────────────┴────────────┴───────────┴──────────────┘
                                     Chroma (one persistent store, 4 collections)
                                                     ▲
                                                     │ mirrors entities/relations
                                              ┌──────┴──────┐
                                              │  NetworkX     │
                                              │  graph (nodes/│
                                              │  edges, on disk)│
                                              └─────────────┘
```

Every `/query` call: **guardrail input check → retrieve from DL + EE + RM +
GS → gpt-4o-mini answer generation → guardrail output check**. LangSmith
tracing wraps each retrieval step and the generation step via `@traceable`,
plus LangChain's native tracing on the `ChatOpenAI` call.

## The DL / EE / RM / GS collections

The vector store isn't one Chroma collection — it's four, all living in
the same on-disk persistent directory, **categorised by what each one
embeds** (not by any business taxonomy like departments):

| Code | Name | What's embedded in it | Purpose |
|---|---|---|---|
| **DL** | Document Layer | chunk-level excerpts of every ingested document | the base RAG layer — the actual passages an answer quotes/cites |
| **EE** | Entity Embeddings | one vector per unique entity extracted into the knowledge graph (people, systems, policies, processes, projects, regulations, ...) | lets a query semantically match an entity even when the wording differs from its canonical name, instead of relying on literal keyword overlap |
| **RM** | Relationship Mappings | one vector per extracted relationship triple, textualised (`"X — depends on — Y"`) | lets a query surface a relevant graph edge directly, without first having to land on one of its endpoint entities |
| **GS** | Graph Summary | one vector per document-level summary | a "what is this document broadly about" layer for thematic/global questions a single chunk wouldn't answer well |

A query hits all four: DL for the actual passages, EE to semantically seed
which graph entities are relevant (then NetworkX walks outward from those
to pull in related entities), RM to pull in relevant relationships
directly, and GS for document-level context. All four get combined into
the prompt sent to gpt-4o-mini. See `backend/rag/kg_rag.py` (builds
EE/RM/GS from each document) and `backend/rag/retriever.py` (the generic
multi-collection Chroma wrapper both DL and KG use).

## Project layout

```
internal_knowledge_assistant/
├── backend/
│   ├── app.py                  FastAPI entrypoint (auto-ingests on startup)
│   ├── config.py               Settings, the 4 Chroma collection names, paths
│   ├── models/schemas.py       Pydantic request/response models
│   ├── routers/
│   │   ├── ingest.py           POST /reindex, GET /documents
│   │   ├── query.py            POST /query
│   │   ├── graph.py            GET  /graph
│   │   └── evaluate.py         POST /evaluate
│   ├── rag/
│   │   ├── document_loader.py  PDF/DOCX/TXT/MD -> chunks
│   │   ├── retriever.py        Generic multi-collection Chroma wrapper (DL/EE/RM/GS)
│   │   ├── kg_rag.py           NetworkX graph + populates EE/RM/GS
│   │   ├── ingest.py           Folder scan + idempotent (re)indexing
│   │   └── llm_chain.py        Combines DL+EE+RM+GS context -> gpt-4o-mini
│   ├── guardrails/
│   │   ├── guard.py            GuardrailEngine (NeMo + fallback)
│   │   └── config/              NeMo Guardrails colang config
│   ├── eval/
│   │   ├── deepeval_runner.py  On-demand metric scoring (used by /evaluate)
│   │   └── test_rag_eval.py    `deepeval test run` regression suite
│   ├── data/
│   │   ├── documents/          ← drop your PDF/DOCX/TXT/MD files here
│   │   ├── chroma/              Chroma's on-disk store (gitignored)
│   │   ├── graph/                knowledge_graph.json (gitignored)
│   │   └── manifest.json        tracks what's indexed (gitignored)
│   ├── requirements.txt        Backend-only deps (used by backend/Dockerfile)
│   └── Dockerfile
├── frontend/
│   ├── app.py                  Streamlit app (Chat / Graph / Evaluation tabs)
│   ├── utils/
│   │   ├── api_client.py       Backend HTTP client
│   │   └── graph_viz.py        pyvis knowledge-graph renderer
│   ├── requirements.txt        Frontend-only deps (used by frontend/Dockerfile)
│   └── Dockerfile
├── deploy/nginx/               Optional TLS/websocket reverse proxy for prod
│   ├── nginx.conf
│   └── certs/                  (put fullchain.pem / privkey.pem here)
├── docker-compose.yml          Local dev (hot-reload, source mounted)
├── docker-compose.prod.yml     Production (built image, no reload, limits)
├── .dockerignore
├── requirements.txt            Full local (non-Docker) dev environment
├── .env.example
├── run_backend.bat
└── run_frontend.bat
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

### Add your documents

Drop PDF / DOCX / TXT / MD files into `backend/data/documents/` (flat is
fine, subfolders are fine too — they have no special meaning). There's no
upload button in the UI; this folder is the single source of truth for
what's in the knowledge base. See `backend/data/documents/README.md`.

### Run

```bat
:: terminal 1
run_backend.bat        REM FastAPI on http://localhost:8000  (docs at /docs)

:: terminal 2
run_frontend.bat       REM Streamlit on http://localhost:8501
```

Starting the backend automatically scans `backend/data/documents/` and
indexes anything new or changed (see "Ingestion" below) — open
http://localhost:8501 and start asking questions once it's done.

## Ingestion

Ingestion is **folder-based and startup-triggered**, not upload-based:

- Every time the backend starts, `rag/ingest.py`'s `ingest_folder()` scans
  `backend/data/documents/`, chunks + embeds any new/changed file into the
  **DL** collection, extracts entities/relationships into the knowledge
  graph (mirrored into **EE**/**RM**), generates a summary into **GS**,
  and records it in `backend/data/manifest.json`.
- It's **idempotent**: re-running with no file changes skips everything
  (fast); a changed file has its old records removed and is re-indexed;
  a file you delete from the folder is pruned from all four collections
  and the graph.
- To pick up new files without restarting the whole server, call
  `POST /reindex` directly (e.g. `curl -X POST http://localhost:8000/reindex`
  or via `/docs`) — it runs the same scan on demand. This isn't wired into
  the Streamlit UI by design; add new files while the server is running
  and hit this endpoint (or just restart the backend) when you want them
  indexed.
- `GET /documents` lists what's currently indexed (path, doc id,
  fingerprint) if you want to check ingestion status.

## Running with Docker

Two services, each with its own multi-stage `Dockerfile` (`backend/Dockerfile`,
`frontend/Dockerfile`) so images stay small — the frontend image doesn't
pull in langchain/chromadb/nemoguardrails/deepeval since it only ever
talks to the backend over HTTP. Both run as a non-root user and expose a
`HEALTHCHECK`.

### Local development

```bash
cp .env.example .env      # fill in OPENAI_API_KEY at minimum
docker compose up --build
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

`docker-compose.yml` bind-mounts the whole `./backend` and `./frontend`
folders into their containers, so code edits take effect immediately
(uvicorn `--reload` + Streamlit's own auto-rerun) without rebuilding —
and since `backend/data/documents/` (along with `chroma/`, `graph/`,
`manifest.json`) is part of that same bind-mounted folder, dropping files
into it on your host and restarting the backend container
(`docker compose restart backend`) is all you need; nothing extra to
configure.

### Production

```bash
cp .env.example .env      # real OPENAI_API_KEY, review GUARDRAILS_ENABLED, etc.
docker compose -f docker-compose.prod.yml up -d --build
```

`docker-compose.prod.yml` is a standalone file (not an override) that:
- builds a fixed image snapshot instead of bind-mounting source — except
  `backend/data/documents/`, which stays a host bind mount specifically so
  you can add/remove source files on the production host without
  rebuilding (Chroma + the graph live in Docker-managed named volumes),
- runs a single uvicorn worker with no `--reload` (see the concurrency
  note in `backend/Dockerfile` — the in-process knowledge-graph store and
  the embedded Chroma client aren't safe for concurrent writers across
  multiple workers/replicas yet),
- sets `restart: always` and basic CPU/memory limits,
- binds the backend to `127.0.0.1:8000` only (reachable from the host for
  debugging, not from the public internet — there's no auth layer on the
  API yet, see "Notes" below),
- optionally starts an nginx reverse proxy in front of Streamlit with
  websocket-aware config for TLS termination, behind a Compose profile so
  it doesn't start unless asked for:

  ```bash
  # after putting real certs in deploy/nginx/certs/ and editing
  # deploy/nginx/nginx.conf's server_name:
  docker compose -f docker-compose.prod.yml --profile with-nginx up -d --build
  ```

Since ingestion only runs at startup, pick up new files in production
with either `docker compose -f docker-compose.prod.yml restart backend`
or `POST /reindex` (no restart needed for the latter).

Scaling beyond one backend replica, or adding real authentication in
front of the API, are the two things to tackle first if this needs to
grow past a single-team internal tool — both are called out inline in
`backend/Dockerfile` and the "Notes" section below.

### LangSmith tracing (optional)

Set in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=internal-knowledge-assistant
```
Runs will appear at https://smith.langchain.com under that project.

### Guardrails

`GUARDRAILS_ENABLED=true` (default) tries to initialise **NeMo Guardrails**
using `backend/guardrails/config/` (LLM self-check-input / self-check-output
rails defined in `config.yml` + `rails/general.co`). If `nemoguardrails`
isn't installed, the config fails to load, or a call errors at runtime, it
**automatically falls back** to a fast deterministic rule engine
(`SimpleGuard` in `guard.py`) that catches prompt-injection phrasing and
requests for fabricated sensitive data — so the app keeps working either
way. The active engine name is returned in every `/query` response
(`guardrail_engine` field) and shown in the Streamlit UI.

NeMo Guardrails also ships a `llama_guard_check_input` / `_output` action
out of the box if you'd rather swap in Meta's Llama Guard model as the
safety classifier instead of the LLM self-check prompts — point the
`rails.input.flows` / `rails.output.flows` in `config.yml` at those actions
and configure a `llama_guard` model entry.

### Evaluation

- **In-app**: after asking a question, open the *Evaluation* tab, pick the
  answer, optionally add a ground-truth answer, and click "Run evaluation"
  to score it with DeepEval (calls `/evaluate`, which uses gpt-4o-mini as
  the judge model).
- **CI / regression**: `cd backend && deepeval test run eval/test_rag_eval.py`
  runs the golden-question suite end-to-end (live retrieval across
  DL/EE/RM/GS + live generation + DeepEval judge) and asserts on Answer
  Relevancy / Faithfulness thresholds. Extend `GOLDEN_SET` in that file
  with real questions once you've ingested your own documents.

## API reference (see `/docs` for full OpenAPI UI)

| Method | Path | Purpose |
|---|---|---|
| POST | `/reindex` | Rescan `backend/data/documents/` and sync DL/EE/RM/GS + the graph (also runs automatically on backend startup) |
| GET | `/documents` | List currently indexed documents (path, doc id, fingerprint) |
| POST | `/query` | Ask a question (`question`, optional `chat_history`, `use_knowledge_graph`) |
| GET | `/graph` | Node-link JSON of the knowledge graph |
| POST | `/evaluate` | Score a Q&A turn with DeepEval metrics |
| GET | `/health` | Service + config status (includes the 4 collection names) |

## Notes / things you'll likely want to change

- **Embeddings cost**: `text-embedding-3-small` is used by default for
  cost; swap `EMBEDDING_MODEL` in `.env` if you need higher recall. Note
  every ingested document is embedded up to 4 times over (once per
  chunk into DL, once as an entity into EE, once per relation into RM,
  once as a summary into GS) — still cheap at `text-embedding-3-small`
  prices, but worth knowing if you swap to a pricier embedding model.
- **Entity/relation extraction + summary cost**: `kg_rag.py` makes two LLM
  calls per ingested document (extraction + summary), each capped at
  ~6000 characters of input. For very large documents, consider chunk-wise
  extraction with de-duplication instead.
- **CORS**: `app.py` currently allows all origins (`allow_origins=["*"]`)
  for local development — tighten this before deploying.
- **Auth**: there's no authentication layer here; add one (API keys, OAuth,
  SSO) before exposing this beyond localhost, especially since it's meant
  to serve internal/sensitive documents. This matters more, not less, once
  it's in Docker/prod and reachable over the network.
- **Concurrent writers / scaling out**: the knowledge-graph store persists
  to a single JSON file guarded by an in-process lock, and the vector
  store uses Chroma's embedded persistent client. Both assume a single
  backend process. Running multiple uvicorn workers or multiple container
  replicas that all write to the same volume can race if ingestion runs
  concurrently in more than one process. Fine for one team's internal tool
  behind one backend container; if you need to scale writes horizontally,
  move the KG store to a real datastore and point Chroma at a shared
  `chromadb` server (`HttpClient`) instead of the embedded client used here.
- **PDF OCR**: scanned/image-only PDFs won't extract text via `pypdf`;
  add an OCR step (e.g. `unstructured` with `hi_res` strategy, already in
  `requirements.txt`) if that's part of your corpus.
