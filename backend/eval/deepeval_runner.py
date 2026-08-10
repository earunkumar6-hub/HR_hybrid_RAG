"""
LLM evaluation layer using the DeepEval framework.

Computes reference-free RAG metrics (Answer Relevancy, Faithfulness,
Contextual Relevancy) for every answer, and — when an `expected_output`
/ ground-truth answer is supplied — also Contextual Precision and
Contextual Recall, which need a reference answer to score retrieval
ranking quality.

Used both:
  * on-demand from the API (`POST /evaluate`, wired to the Evaluation tab
    in the Streamlit UI), and
  * offline via `deepeval test run eval/test_rag_eval.py` (see that file)
    for CI-style regression testing of the RAG pipeline.

Runs metrics via each metric's native `a_measure()` coroutine, gathered
concurrently with `asyncio.gather` — two things this buys over the naive
`metric.measure(...)` sync call run one-at-a-time in a loop:

  1. Speed: measure() (sync) with 3-5 metrics run sequentially can easily
     take 60-150+ seconds, since each metric makes several of its own LLM
     calls internally. Running them concurrently instead cuts wall-clock
     time roughly in proportion to the metric count.
  2. Correctness under uvicorn+uvloop: measure() internally does
     `get_or_create_event_loop().run_until_complete(self.a_measure(...))`,
     which uses `nest_asyncio` to run inside whatever loop is already
     active — and nest_asyncio can't patch uvloop's loop type, which is
     what uvicorn uses by default. Calling `a_measure()` directly and
     awaiting it from our own already-async route handler sidesteps that
     bridge entirely; there's nothing to nest.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from config import settings
from models.schemas import MetricResult

_THRESHOLD = 0.5


def _build_metrics(expected_output: Optional[str]) -> list:
    metrics = [
        AnswerRelevancyMetric(threshold=_THRESHOLD, model=settings.llm_model, async_mode=True),
        FaithfulnessMetric(threshold=_THRESHOLD, model=settings.llm_model, async_mode=True),
        ContextualRelevancyMetric(threshold=_THRESHOLD, model=settings.llm_model, async_mode=True),
    ]
    if expected_output:
        metrics.append(ContextualPrecisionMetric(threshold=_THRESHOLD, model=settings.llm_model, async_mode=True))
        metrics.append(ContextualRecallMetric(threshold=_THRESHOLD, model=settings.llm_model, async_mode=True))
    return metrics


_METRIC_TIMEOUT_SECONDS = 60


async def _measure_one(metric, test_case: LLMTestCase) -> MetricResult:
    try:
        await asyncio.wait_for(
            metric.a_measure(test_case, _show_indicator=False),
            timeout=_METRIC_TIMEOUT_SECONDS,
        )
        return MetricResult(
            name=metric.__class__.__name__,
            score=round(float(metric.score or 0.0), 4),
            threshold=metric.threshold,
            passed=bool(metric.success),
            reason=getattr(metric, "reason", None),
        )
    except asyncio.TimeoutError:
        # A hung/very slow OpenAI call (network issue, rate limiting,
        # etc.) fails this one metric cleanly instead of hanging the
        # whole /evaluate request indefinitely — asyncio.gather has no
        # timeout of its own, so without this a stuck LLM call would
        # block the response until the *client's* timeout gives up with
        # a generic, unhelpful "Read timed out".
        return MetricResult(
            name=metric.__class__.__name__,
            score=0.0,
            threshold=metric.threshold,
            passed=False,
            reason=f"Metric timed out after {_METRIC_TIMEOUT_SECONDS}s — likely a network issue "
            "reaching the OpenAI API from the backend container, or rate limiting.",
        )
    except Exception as exc:  # e.g. no/invalid OPENAI_API_KEY, transient API error
        return MetricResult(
            name=metric.__class__.__name__,
            score=0.0,
            threshold=metric.threshold,
            passed=False,
            reason=f"Metric could not be computed: {exc}",
        )


async def evaluate_response(
    question: str,
    answer: str,
    retrieved_context: List[str],
    expected_output: Optional[str] = None,
) -> List[MetricResult]:
    """Run DeepEval metrics against one Q&A turn — concurrently — and
    return their results. Each metric fails independently (one bad/slow
    metric doesn't take the others down with it)."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieved_context or [""],
        expected_output=expected_output,
    )
    metrics = _build_metrics(expected_output)
    return await asyncio.gather(*(_measure_one(m, test_case) for m in metrics))
