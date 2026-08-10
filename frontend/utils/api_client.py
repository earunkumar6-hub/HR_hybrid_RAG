"""Thin HTTP client wrapping the FastAPI backend for the Streamlit UI."""
import os
from typing import Optional

import requests

BACKEND_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
TIMEOUT = 60


def health() -> dict:
    r = requests.get(f"{BACKEND_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def ask(
    question: str,
    chat_history: Optional[list] = None,
    use_knowledge_graph: bool = True,
) -> dict:
    payload = {
        "question": question,
        "chat_history": chat_history or [],
        "use_knowledge_graph": use_knowledge_graph,
    }
    r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_graph() -> dict:
    r = requests.get(f"{BACKEND_URL}/graph", timeout=30)
    r.raise_for_status()
    return r.json()


def evaluate(question: str, answer: str, retrieved_context: list, expected_output: Optional[str] = None) -> dict:
    payload = {
        "question": question,
        "answer": answer,
        "retrieved_context": retrieved_context,
        "expected_output": expected_output,
    }
    # DeepEval metrics run concurrently server-side now, but each still
    # makes several of its own LLM calls — 180s gives real headroom on a
    # slow API day without the UI hanging silently forever.
    r = requests.post(f"{BACKEND_URL}/evaluate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


# ---- Ingestion status (informational; there's no upload/reindex control
# in the UI by design — documents are read from backend/data/documents/
# and indexed automatically at backend startup). Kept here as thin
# wrappers in case a future admin view wants them. ----
def reindex() -> dict:
    r = requests.post(f"{BACKEND_URL}/reindex", timeout=300)
    r.raise_for_status()
    return r.json()


def list_documents() -> dict:
    r = requests.get(f"{BACKEND_URL}/documents", timeout=15)
    r.raise_for_status()
    return r.json()
