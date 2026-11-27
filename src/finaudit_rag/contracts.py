"""Typed request, evidence selection, and response contracts."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from finaudit_rag.arithmetic import Operation
from finaudit_rag.lookup import ReportingScope

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: Question


class ExcerptSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(min_length=1, max_length=200)
    start_line: int = Field(ge=0, lt=300)
    end_line: int = Field(ge=0, lt=300)


class OperandSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(min_length=1, max_length=200)
    row: int = Field(ge=0, lt=100)
    column: int = Field(ge=1, lt=30)


class AnswerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["excerpt", "calculation", "abstain"]
    excerpts: list[ExcerptSelection] = Field(default_factory=list, max_length=3)
    operands: list[OperandSelection] = Field(default_factory=list, max_length=4)


class Citation(BaseModel):
    document_id: str
    title: str
    page: int = Field(ge=1)
    url: str
    chunk_id: str


class CitedExcerpt(BaseModel):
    quote: str
    citation: Citation


class CitedOperand(BaseModel):
    metric: str
    period: str
    amount: Decimal
    unit: str
    citation: Citation


class CitedLookup(CitedOperand):
    scope: ReportingScope


class CalculationResult(BaseModel):
    operation: Operation
    result: Decimal
    unit: str
    formula: str
    operands: list[CitedOperand]


class QueryResponse(BaseModel):
    status: Literal["answered", "abstained"]
    redacted_question: str
    reason: str | None = None
    excerpts: list[CitedExcerpt] = Field(default_factory=list)
    calculation: CalculationResult | None = None
    lookup: CitedLookup | None = None
