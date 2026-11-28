import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from evals.run import Meter
from finaudit_rag.contracts import Citation, CitedLookup
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


@pytest.mark.parametrize(
    ("period", "scope", "amount"),
    [
        ("FY2024", "standalone", "30880"),
        ("FY2025", "consolidated", "30880"),
        ("FY2025", "standalone", "28813"),
    ],
)
def test_financial_lookup_scoring_rejects_wrong_metadata(
    period: str, scope: str, amount: str
) -> None:
    docs = {item.id: item for item in load_manifest(Path("data/corpus.json"))}
    case = next(case for case in load_cases(Path("evals"), docs) if case.id == "fin-05")
    lookup = CitedLookup(
        metric="Operating profit",
        period="FY2025",
        scope="standalone",
        amount=Decimal("30880"),
        unit="INR crore",
        citation=Citation(
            document_id="infosys-fy25-annual",
            title="Annual report",
            page=264,
            url="https://example.org/report.pdf#page=264",
            chunk_id="source",
        ),
    )
    observation = Observation(
        id=case.id,
        status="answered",
        text="Operating profit 30880 INR crore",
        citations=case.evidence,
        seconds=0.0,
        lookup=lookup,
    )
    assert summarize([case], [observation])["reference_grounded_accuracy"] == 1.0
    mismatch = observation.model_copy(
        update={
            "lookup": lookup.model_copy(
                update={"period": period, "scope": scope, "amount": Decimal(amount)}
            )
        }
    )
    assert summarize([case], [mismatch])["reference_grounded_accuracy"] == 0.0


def test_attempt_ledger_preserves_usage_before_a_rate_limit(tmp_path: Path) -> None:
    ledger = tmp_path / "run.attempts.jsonl"
    meter = Meter(ledger, "case-1")
    meter.capture(
        httpx.Response(200, json={"usage": {"prompt_tokens": 120, "completion_tokens": 40}})
    )
    meter.capture(httpx.Response(429, headers={"retry-after": "15"}))
    assert meter.rate_limited and meter.retry_seconds == 15
    resumed = Meter(ledger, "case-1")
    resumed.capture(
        httpx.Response(200, json={"usage": {"prompt_tokens": 130, "completion_tokens": 30}})
    )
    records = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
    assert sum(item["input_tokens"] for item in records) == 250
    assert sum(item["output_tokens"] for item in records) == 70
    assert [item["status"] for item in records] == [200, 429, 200]
