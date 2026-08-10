import logging
import time

from fastapi import APIRouter

from guardrails.guard import get_guardrail_engine
from models.schemas import QueryRequest, QueryResponse
from rag.llm_chain import generate_answer

logger = logging.getLogger("query")
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    start = time.perf_counter()
    guard = get_guardrail_engine()

    input_check = guard.check_input(payload.question)
    if not input_check.allowed:
        return QueryResponse(
            answer=input_check.reason or "This request can't be processed.",
            sources=[],
            entity_context=[],
            relationship_context=[],
            blocked=True,
            block_reason=input_check.reason,
            guardrail_engine=input_check.engine,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    answer, sources, entities, relationships = generate_answer(
        question=payload.question,
        chat_history=payload.chat_history,
        use_knowledge_graph=payload.use_knowledge_graph,
        top_k=payload.top_k,
    )

    output_check = guard.check_output(answer)
    if not output_check.allowed:
        return QueryResponse(
            answer="I generated a response but it was blocked by the output guardrail: "
            f"{output_check.reason}",
            sources=[],
            entity_context=[],
            relationship_context=[],
            blocked=True,
            block_reason=output_check.reason,
            guardrail_engine=output_check.engine,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        entity_context=entities,
        relationship_context=relationships,
        blocked=False,
        guardrail_engine=output_check.engine,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )
