"""
Knowledge-Graph-augmented RAG using NetworkX, backed by two of the four
Chroma collections (see config.COLLECTIONS):

  * EE (Entity Embeddings)      — one vector per unique entity node
  * RM (Relationship Mappings)  — one vector per extracted relation triple

Every ingested document contributes:
  * one DOCUMENT node in the NetworkX graph (id = doc_id)
  * N    ENTITY nodes            (people, systems, policies, projects, ...)
  * MENTIONS edges                entity -> document
  * RELATED_TO edges              entity -> entity (with a short relation label)
  * a document-level summary, embedded into the GS (Graph Summary) collection

The graph itself (structure/traversal) lives in NetworkX and is persisted
to disk as node-link JSON. Entities and relationships are *additionally*
embedded into Chroma so a query can semantically match an entity or a
relationship even when the wording differs from how the graph stored it
— graph traversal alone only ever matches literal node labels, which is
too brittle for free-text questions.

Entity/relation extraction and summarisation use the chat LLM
(gpt-4o-mini). If no OPENAI_API_KEY is configured, extraction falls back
to a lightweight capitalised-phrase heuristic and the summary falls back
to a truncated excerpt, so the rest of the pipeline keeps working offline.
"""
from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain_core.documents import Document
from networkx.readwrite import json_graph

from config import GRAPH_FILE, settings
from rag.retriever import add_to_store, delete_from_store, search_store

_LOCK = threading.Lock()

_EXTRACTION_PROMPT = """You are an information-extraction engine for an internal \
company knowledge graph. Read the document excerpt below and extract the \
important named entities (people, teams, systems/products, policies, \
processes, regulations, projects) and the relationships between them.

Return ONLY valid JSON, no prose, no markdown fences, in exactly this shape:
{{
  "entities": [{{"name": "string", "type": "string"}}, ...],
  "relations": [{{"source": "string", "relation": "string", "target": "string"}}, ...]
}}

Rules:
- Use short, canonical entity names (e.g. "Loan Origination System", not \
"the loan origination system used by the team").
- "type" should be a short category label, e.g. person, team, system, \
policy, process, project, regulation.
- Extract at most 15 entities and 15 relations.
- "relation" should be a short verb phrase, e.g. "owns", "approves", \
"depends on", "reports to", "governed by".
- If nothing meaningful can be extracted, return {{"entities": [], "relations": []}}.

DOCUMENT EXCERPT (file: {filename}):
---
{text}
---
"""

_SUMMARY_PROMPT = """Summarise the internal document excerpt below in 2-3 \
sentences: what it's about and the main entities/topics it covers. Plain \
text only, no markdown, no preamble.

FILE: {filename}
---
{text}
---
"""

_MAX_EXTRACTION_CHARS = 6000
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9&]*(?:\s+[A-Z][a-zA-Z0-9&]*){0,3})\b")
_STOPWORDS = {"The", "This", "That", "These", "Those", "It", "In", "On", "For", "A", "An"}


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "x"


def _heuristic_extract(text: str) -> Tuple[List[dict], List[dict]]:
    """Offline fallback: pull capitalised multi-word phrases as entities and
    link consecutive ones with a generic 'related to' edge. Used only when
    no OpenAI API key is available."""
    seen: Dict[str, None] = {}
    for match in _CAP_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        words = phrase.split(" ")
        while words and words[0] in _STOPWORDS:
            words = words[1:]
        phrase = " ".join(words)
        if not phrase or phrase in _STOPWORDS or len(phrase) < 3:
            continue
        seen[phrase] = None
        if len(seen) >= 15:
            break
    names = list(seen.keys())
    entities = [{"name": n, "type": "term"} for n in names]
    relations = [
        {"source": names[i], "relation": "related to", "target": names[i + 1]}
        for i in range(len(names) - 1)
    ]
    return entities, relations


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)


def extract_entities_relations(text: str, filename: str) -> Tuple[List[dict], List[dict]]:
    if not settings.openai_api_key:
        return _heuristic_extract(text[:_MAX_EXTRACTION_CHARS])

    prompt = _EXTRACTION_PROMPT.format(filename=filename, text=text[:_MAX_EXTRACTION_CHARS])
    try:
        raw = _get_llm().invoke(prompt).content
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return data.get("entities", [])[:15], data.get("relations", [])[:15]
    except Exception:
        # LLM hiccup or malformed JSON -> degrade gracefully rather than
        # failing the whole ingest.
        return _heuristic_extract(text[:_MAX_EXTRACTION_CHARS])


def generate_summary(text: str, filename: str) -> str:
    if not settings.openai_api_key:
        excerpt = " ".join(text.split())[:280]
        return f"{filename}: {excerpt}{'...' if len(text) > 280 else ''}"
    try:
        prompt = _SUMMARY_PROMPT.format(filename=filename, text=text[:_MAX_EXTRACTION_CHARS])
        return _get_llm().invoke(prompt).content.strip()
    except Exception:
        excerpt = " ".join(text.split())[:280]
        return f"{filename}: {excerpt}{'...' if len(text) > 280 else ''}"


class KnowledgeGraphStore:
    """Thread-safe singleton wrapper around a persisted nx.Graph, kept in
    sync with the EE/RM/GS Chroma collections."""

    _instance: Optional["KnowledgeGraphStore"] = None

    def __init__(self):
        self.graph = nx.Graph()
        self._load()

    @classmethod
    def instance(cls) -> "KnowledgeGraphStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if GRAPH_FILE.exists():
            try:
                data = json.loads(GRAPH_FILE.read_text())
                self.graph = json_graph.node_link_graph(data, edges="edges")
            except Exception:
                self.graph = nx.Graph()

    def _save(self) -> None:
        data = json_graph.node_link_data(self.graph, edges="edges")
        GRAPH_FILE.write_text(json.dumps(data, indent=2))

    # -- mutation ----------------------------------------------------------
    def add_document(self, doc_id: str, filename: str, full_text: str) -> Tuple[int, int]:
        """Extracts entities/relations/summary from `full_text` and merges
        them into the graph + EE/RM/GS Chroma collections. Safe to call
        again for the same doc_id (upserts throughout)."""
        entities, relations = extract_entities_relations(full_text, filename)
        summary = generate_summary(full_text, filename)

        ee_docs: List[Document] = []
        rm_docs: List[Document] = []

        with _LOCK:
            self.graph.add_node(doc_id, label=filename, type="document")

            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                entity_type = ent.get("type", "term") or "term"
                node_id = f"ent::{name.lower()}"
                if self.graph.has_node(node_id):
                    self.graph.nodes[node_id]["shared"] = True
                else:
                    self.graph.add_node(
                        node_id, label=name, type="entity", entity_type=entity_type, shared=False,
                    )
                self.graph.add_edge(node_id, doc_id, relation="mentioned in")

                ee_docs.append(
                    Document(
                        page_content=f"{name} ({entity_type})",
                        metadata={"id": node_id, "doc_id": doc_id, "entity_type": entity_type, "name": name},
                    )
                )

            for rel in relations:
                s = rel.get("source", "").strip()
                t = rel.get("target", "").strip()
                relation = rel.get("relation", "related to") or "related to"
                if not s or not t:
                    continue
                s_id, t_id = f"ent::{s.lower()}", f"ent::{t.lower()}"
                if not self.graph.has_node(s_id):
                    self.graph.add_node(s_id, label=s, type="entity", entity_type="term", shared=False)
                if not self.graph.has_node(t_id):
                    self.graph.add_node(t_id, label=t, type="entity", entity_type="term", shared=False)
                self.graph.add_edge(s_id, t_id, relation=relation)

                rel_id = f"rel::{s_id}|{_slug(relation)}|{t_id}"
                rm_docs.append(
                    Document(
                        page_content=f"{s} {relation} {t}",
                        metadata={"id": rel_id, "doc_id": doc_id, "source": s, "relation": relation, "target": t},
                    )
                )

            self._save()

        # Chroma writes happen outside the graph lock (they're I/O bound
        # and don't touch shared in-process state).
        add_to_store("EE", ee_docs)
        add_to_store("RM", rm_docs)
        add_to_store(
            "GS",
            [Document(page_content=summary, metadata={"id": doc_id, "doc_id": doc_id, "filename": filename})],
        )

        return len(entities), len(relations)

    def remove_document(self, doc_id: str) -> None:
        with _LOCK:
            if self.graph.has_node(doc_id):
                # drop entity nodes that were only mentioned in this doc
                for neighbor in list(self.graph.neighbors(doc_id)):
                    if self.graph.nodes[neighbor].get("type") != "entity":
                        continue
                    if self.graph.degree(neighbor) <= 1:
                        self.graph.remove_node(neighbor)
                self.graph.remove_node(doc_id)
            self._save()
        delete_from_store("EE", doc_id)
        delete_from_store("RM", doc_id)
        delete_from_store("GS", doc_id)

    # -- query -------------------------------------------------------------
    def query(self, question: str, depth: int = None) -> List[dict]:
        """Semantically finds seed entities via the EE collection, then
        walks `depth` hops out in the graph to collect related entities.
        Returns a list of {entity, related_to} used as extra LLM context."""
        depth = depth or settings.graph_traversal_depth
        if self.graph.number_of_nodes() == 0:
            return []

        hits = search_store("EE", question, k=settings.top_k_entities)
        seeds = [doc.metadata["id"] for doc, _score in hits if self.graph.has_node(doc.metadata["id"])]
        if not seeds:
            return []

        results = []
        visited: Set[str] = set()
        for seed in seeds:
            if seed in visited or not self.graph.has_node(seed):
                continue
            ego = nx.ego_graph(self.graph, seed, radius=depth)
            related = [
                self.graph.nodes[n]["label"]
                for n in ego.nodes
                if n != seed and self.graph.nodes[n].get("type") == "entity"
            ]
            visited.add(seed)
            visited.update(ego.nodes)
            results.append({"entity": self.graph.nodes[seed]["label"], "related_to": related[:10]})
        return results

    def query_relationships(self, question: str, k: int = None) -> List[dict]:
        """Direct semantic search over the RM (relationship) collection —
        surfaces a relevant graph edge even when the query doesn't closely
        match either endpoint entity's name."""
        k = k or settings.top_k_relationships
        hits = search_store("RM", question, k=k)
        return [
            {
                "source": doc.metadata.get("source"),
                "relation": doc.metadata.get("relation"),
                "target": doc.metadata.get("target"),
            }
            for doc, _score in hits
        ]

    # -- visualization -------------------------------------------------------
    def subgraph_for_viz(self, max_nodes: int = 150) -> dict:
        g = self.graph
        if g.number_of_nodes() > max_nodes:
            top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            g = g.subgraph([n for n, _ in top_nodes])

        nodes = [
            {
                "id": n,
                "label": a.get("label", n),
                "type": a.get("type", "entity"),
                "entity_type": a.get("entity_type", ""),
            }
            for n, a in g.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "relation": a.get("relation", "")}
            for u, v, a in g.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "shown_nodes": len(nodes),
            },
        }


@lru_cache(maxsize=1)
def get_kg_store() -> KnowledgeGraphStore:
    return KnowledgeGraphStore.instance()
