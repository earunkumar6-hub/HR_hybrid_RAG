"""Pydantic request/response models shared across API routers."""
from typing import List, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Ingestion (folder-based; see rag/ingest.py)
# ----------------------------------------------------------------------
class ReindexResponse(BaseModel):
    added: int
    updated: int
    skipped: int
    removed: int
    failed: int
    total_indexed: int


class IndexedDocument(BaseModel):
    path: str
    doc_id: str
    fingerprint: str


class DocumentsListResponse(BaseModel):
    documents: List[IndexedDocument]
    total: int


# ----------------------------------------------------------------------
# Query
# ----------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: Optional[List[ChatTurn]] = None
    use_knowledge_graph: bool = True
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: Optional[float] = None


class EntityContextItem(BaseModel):
    entity: str
    related_to: List[str]


class RelationshipContextItem(BaseModel):
    source: str
    relation: str
    target: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    entity_context: List[EntityContextItem]
    relationship_context: List[RelationshipContextItem]
    blocked: bool = False
    block_reason: Optional[str] = None
    guardrail_engine: Optional[str] = None
    latency_ms: Optional[float] = None


# ----------------------------------------------------------------------
# Knowledge graph
# ----------------------------------------------------------------------
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # "entity" | "document"
    entity_type: Optional[str] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: Optional[str] = None


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    shown_nodes: int


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    question: str
    answer: str
    retrieved_context: List[str]
    expected_output: Optional[str] = None


class MetricResult(BaseModel):
    name: str
    score: float
    threshold: float
    passed: bool
    reason: Optional[str] = None


class EvaluateResponse(BaseModel):
    metrics: List[MetricResult]
    overall_passed: bool
