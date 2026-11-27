from decimal import Decimal
from pathlib import Path

import pytest

from finaudit_rag.corpus import load_manifest
from finaudit_rag.evaluation import Case, Observation, PageKey, load_cases, summarize


def test_frozen_case_files_match_manifest_and_protocol_counts() -> None:
    docs = {item.id: item for item in load_manifest(Path("data/corpus.json"))}
    assert len(load_cases(Path("evals"), docs)) == 120


def test_errors_and_wrong_citations_have_separate_denominators() -> None:
    page = PageKey(document_id="test", page=2)
    other = PageKey(document_id="test", page=3)
    cases = [
        Case(
            id="a",
            category="regulatory",
            query="What interval applies?",
            expected_answer="21 days",
            evidence=[page],
            required_terms=[["21 days"]],
        ),
        Case(
            id="b",
            category="unanswerable",
            query="What is the missing value?",
            evidence=[],
            abstention_reason="The value is missing.",
        ),
    ]
    observations = [
        Observation(
            id="a",
            status="answered",
            text="Within 21 days.",
            citations=[other],
            context=[page, other],
            seconds=0.4,
        ),
        Observation(id="b", status="error", seconds=2.0),
    ]
    result = summarize(cases, observations)
    assert result["answer_accuracy"] == 1.0
    assert result["citation_correctness"] == result["reference_grounded_accuracy"] == 0.0
    assert result["abstention_accuracy"] == 0.0
    assert result["service_errors"] == 1
    assert result["context_precision"] == 0.5
    assert result["p95_seconds"] == 2.0
    with pytest.raises(ValueError, match="match"):
        summarize(cases, observations[:1])


def test_numeric_accuracy_checks_result_and_unit() -> None:
    docs = {item.id: item for item in load_manifest(Path("data/corpus.json"))}
    case = next(case for case in load_cases(Path("evals"), docs) if case.id == "calc-01")
    assert case.expected_result is not None
    match = Observation(
        id=case.id,
        status="answered",
        result=case.expected_result + Decimal("0.01"),
        unit="%",
        citations=case.evidence,
        seconds=0.0,
    )
    assert summarize([case], [match])["numerical_accuracy"] == 1.0
    mismatch = match.model_copy(update={"unit": "percentage points"})
    assert summarize([case], [mismatch])["numerical_accuracy"] == 0.0
