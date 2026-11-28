"""Validate frozen benchmark cases and score persisted observations."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finaudit_rag.contracts import CitedLookup
from finaudit_rag.corpus import Document

Category = Literal["regulatory", "financial_lookup", "calculation", "unanswerable"]
TARGET_COUNTS: dict[Category, int] = {
    "regulatory": 35,
    "financial_lookup": 30,
    "calculation": 30,
    "unanswerable": 25,
}


class PageKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    document_id: str = Field(min_length=1)
    page: int = Field(gt=0)


class GoldOperand(PageKey):
    metric: str
    fiscal_year: int
    amount: Decimal
    unit: str
    scope: Literal["standalone", "consolidated"]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    category: Category
    query: str = Field(min_length=1, max_length=1000)
    evidence: list[PageKey]
    expected_answer: str | None = None
    abstention_reason: str | None = None
    required_terms: list[list[str]] = Field(default_factory=list)
    expected_result: Decimal | None = None
    unit: str | None = None
    tolerance: Decimal | None = Field(default=None, ge=0)
    formula: str | None = None
    operands: list[GoldOperand] = Field(default_factory=list)
    fiscal_year: int | None = None
    scope: Literal["standalone", "consolidated"] | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.category == "unanswerable":
            if self.evidence or self.expected_answer or not self.abstention_reason:
                raise ValueError("unanswerable cases require a reason and no answer or evidence")
        elif not self.evidence or not self.expected_answer or self.abstention_reason:
            raise ValueError("answerable cases require an answer and cited evidence")
        elif self.category == "calculation":
            if (
                self.expected_result is None
                or self.tolerance is None
                or not self.unit
                or not self.formula
                or len(self.operands) not in {2, 4}
            ):
                raise ValueError("calculation case lacks numerical reference data")
            for operand in self.operands:
                if PageKey(document_id=operand.document_id, page=operand.page) not in self.evidence:
                    raise ValueError("operand source is outside the labeled evidence")
        elif not self.required_terms or any(
            not alternatives or any(not term.strip() for term in alternatives)
            for alternatives in self.required_terms
        ):
            raise ValueError("lookup cases require nonempty groups of acceptable terms")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: Literal["answered", "abstained", "error"]
    text: str = ""
    result: Decimal | None = None
    unit: str | None = None
    citations: list[PageKey] = Field(default_factory=list)
    context: list[PageKey] = Field(default_factory=list)
    seconds: float = Field(ge=0, allow_inf_nan=False)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    lookup: CitedLookup | None = None


def load_cases(directory: Path, documents: Mapping[str, Document]) -> list[Case]:
    """Load the protocol's files with unique questions and valid physical PDF pages."""
    cases = [
        Case.model_validate_json(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({case.id for case in cases}) != len(cases) or len(
        {" ".join(case.query.casefold().split()) for case in cases}
    ) != len(cases):
        raise ValueError("benchmark IDs and questions must be unique")
    if Counter(case.category for case in cases) != TARGET_COUNTS:
        raise ValueError("benchmark case counts do not match the frozen protocol")
    for case in cases:
        for ref in case.evidence:
            source = documents.get(ref.document_id)
            if source is None or ref.page > source.pages:
                raise ValueError(f"invalid physical source page in {case.id}")
    return cases


def _has_term(text: str, term: str) -> bool:
    pattern = rf"(?<!\w){re.escape(term.casefold())}(?!\w)"
    return re.search(pattern, text) is not None


def is_correct(case: Case, observed: Observation) -> bool:
    """Check expected answer content independently of citation correctness."""
    if case.category == "unanswerable":
        return observed.status == "abstained"
    if observed.status != "answered":
        return False
    if case.category == "calculation":
        return (
            case.expected_result is not None
            and case.tolerance is not None
            and observed.result is not None
            and observed.unit == case.unit
            and abs(observed.result - case.expected_result) <= case.tolerance
        )
    text = " ".join(observed.text.casefold().split())
    if case.category == "financial_lookup" and observed.lookup is not None:
        lookup = observed.lookup
        if (
            lookup.period != f"FY{case.fiscal_year}"
            or lookup.scope != case.scope
            or lookup.unit != case.unit
            or not any(
                Decimal(term.replace(",", "")) == lookup.amount
                for alternatives in case.required_terms
                for term in alternatives
                if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", term)
            )
        ):
            return False
    return all(any(_has_term(text, term) for term in group) for group in case.required_terms)


def has_gold_citations(case: Case, observed: Observation) -> bool:
    """Require supplied citations to refer exclusively to labeled physical pages."""
    citations = set(observed.citations)
    return bool(citations) and citations == set(case.evidence)


def summarize(
    cases: Sequence[Case], observations: Sequence[Observation]
) -> dict[str, int | float | None]:
    """Compute denominators without rewarding errors or dropped benchmark cases."""
    indexed = {item.id: item for item in observations}
    if len(indexed) != len(observations) or set(indexed) != {case.id for case in cases}:
        raise ValueError("observations must match the evaluation case IDs exactly")
    answered = correct_citations = correct_answers = grounded_answers = 0
    answerable = numeric = correct_numeric = unanswerable = correct_abstentions = errors = 0
    relevant_context = total_context = 0
    for case in cases:
        observation = indexed[case.id]
        errors += observation.status == "error"
        correct = is_correct(case, observation)
        if case.category == "unanswerable":
            unanswerable += 1
            correct_abstentions += correct
            continue
        answerable += 1
        expected_pages = set(case.evidence)
        total_context += len(observation.context)
        relevant_context += sum(page in expected_pages for page in observation.context)
        if observation.status == "answered":
            answered += 1
            correct_answers += correct
            cited = has_gold_citations(case, observation)
            correct_citations += cited
            grounded_answers += cited and correct
        if case.category == "calculation":
            numeric += 1
            correct_numeric += correct

    def ratio(count: int, total: int) -> float | None:
        return count / total if total else None

    timings = sorted(item.seconds for item in observations)
    return {
        "answerable": answerable,
        "answered": answered,
        "correct_answers": correct_answers,
        "correct_citations": correct_citations,
        "numeric": numeric,
        "correct_numeric": correct_numeric,
        "unanswerable": unanswerable,
        "correct_abstentions": correct_abstentions,
        "service_errors": errors,
        "answerable_coverage": ratio(answered, answerable),
        "answer_accuracy": ratio(correct_answers, answerable),
        "reference_grounded_accuracy": ratio(grounded_answers, answerable),
        "citation_correctness": ratio(correct_citations, answered),
        "numerical_accuracy": ratio(correct_numeric, numeric),
        "abstention_accuracy": ratio(correct_abstentions, unanswerable),
        "context_precision": ratio(relevant_context, total_context),
        "p95_seconds": timings[ceil(0.95 * len(timings)) - 1] if timings else None,
        "input_tokens": sum(item.input_tokens for item in observations),
        "output_tokens": sum(item.output_tokens for item in observations),
    }
