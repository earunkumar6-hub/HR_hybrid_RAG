"""
Offline regression suite for the RAG pipeline, runnable with:

    cd backend
    deepeval test run eval/test_rag_eval.py

Add real (question, expected_answer) pairs from your ingested corpus to
`GOLDEN_SET` below as your internal knowledge base grows — the sample
below is a placeholder that only checks the refusal behaviour, since a
fresh checkout has nothing ingested yet. Each case pulls LIVE retrieval
(DL/EE/RM/GS) + a live LLM answer through the same code path used by the
`/query` API, so this doubles as an end-to-end pipeline test, not just a
metrics unit test.

Requires OPENAI_API_KEY to be set (LLM-as-judge metrics + gpt-4o-mini
generation both call OpenAI), and that backend/data/documents/ has
already been ingested at least once (it auto-ingests on backend startup;
run `python -c "from rag.ingest import ingest_folder; ingest_folder()"`
first if you're running this without having started the server).
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from config import settings
from rag.llm_chain import generate_answer

# Each entry: (question, expected_answer_or_None)
GOLDEN_SET = [
    (
        "What is the capital of France?",
        None,  # expect a refusal / "not found in knowledge base" style answer
    ),
    # Add real questions once you've ingested documents, e.g.:
    # ("What is our loan origination approval SLA?", "Standard approvals are..."),
]


@pytest.mark.parametrize("question,expected", GOLDEN_SET)
def test_rag_pipeline_case(question, expected):
    answer, sources, _entities, _relationships = generate_answer(question=question)
    retrieval_context = [s.text for s in sources] or ["(no context retrieved)"]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
        expected_output=expected,
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=settings.llm_model),
        FaithfulnessMetric(threshold=0.5, model=settings.llm_model),
    ]
    assert_test(test_case, metrics)
