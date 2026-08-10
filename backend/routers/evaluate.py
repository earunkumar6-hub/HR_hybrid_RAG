from fastapi import APIRouter

from eval.deepeval_runner import evaluate_response
from models.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(payload: EvaluateRequest):
    # evaluate_response() is a native coroutine (uses each DeepEval
    # metric's a_measure(), gathered concurrently) — just await it
    # directly. No thread pool needed: unlike the sync metric.measure()
    # bridge (which needs nest_asyncio to nest inside an already-running
    # loop, and can't with uvloop), a_measure() is a normal awaitable
    # with nothing to nest.
    metrics = await evaluate_response(
        question=payload.question,
        answer=payload.answer,
        retrieved_context=payload.retrieved_context,
        expected_output=payload.expected_output,
    )
    return EvaluateResponse(
        metrics=metrics,
        overall_passed=all(m.passed for m in metrics),
    )
