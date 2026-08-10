from fastapi import APIRouter

from models.schemas import GraphResponse
from rag.kg_rag import get_kg_store

router = APIRouter(tags=["knowledge-graph"])


@router.get("/graph", response_model=GraphResponse)
async def get_graph():
    store = get_kg_store()
    data = store.subgraph_for_viz()
    return GraphResponse(**data)
