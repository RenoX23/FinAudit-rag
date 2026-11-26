"""Compute financial metrics from verified, unambiguous PDF table cells."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Literal

from finaudit_rag.ingestion import Chunk

Operation = Literal["yoy_change", "operating_margin", "operating_margin_change"]
NUMBER = re.compile(r"(?:\d+|[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d?(?:,\d{2})+,\d{3})(?:\.\d+)?")
FY = re.compile(r"(?:19|20|21)\d{2}")
NOTE = re.compile(r"\s+\d+(?:\.\d+)+$")
INR_CRORE = re.compile(r"(?:₹|\bINR\b|\bRs\.?\s)\s*crore\b", re.IGNORECASE)
YEAR_ENDED = re.compile(r"\b(?:year|twelve months) ended\b", re.IGNORECASE)
QUARTER_ENDED = re.compile(r"\b(?:quarter|three months) ended\b", re.IGNORECASE)
PROFIT = {"operating profit", "profit from operations"}
REVENUE = {"revenue", "revenue from operations"}
MONEY_METRICS = PROFIT | REVENUE | {"gross profit", "net income", "profit before tax"}


class UnverifiedCalculation(ValueError):
    """The retrieved cells cannot support the requested calculation."""


@dataclass(frozen=True)
class CellReference:
    chunk_id: str
    row: int
    column: int


@dataclass(frozen=True)
class Operand:
    chunk_id: str
    row_label: str
    period: str
    amount: Decimal
    unit: str


@dataclass(frozen=True)
class Calculation:
    operation: Operation
    result: Decimal
    unit: str
    formula: str
    operands: tuple[Operand, ...]


def _amount(raw: str) -> Decimal:
    text = raw.strip()
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    if text.startswith(("-", "+")):
        text = text[1:]
    elif text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if len(text) > 25 or not NUMBER.fullmatch(text):
        raise UnverifiedCalculation("the cited cell is not a supported amount")
    value = Decimal(text.replace(",", ""))
    return -value if negative else value


def _read_operand(reference: CellReference, evidence: Mapping[str, Chunk]) -> Operand:
    if (chunk := evidence.get(reference.chunk_id)) is None or chunk.table is None:
        raise UnverifiedCalculation("the operand is not in a retrieved table")
    table = chunk.table
    if not (0 <= reference.row < len(table.rows) and 0 < reference.column < len(table.columns)):
        raise UnverifiedCalculation("the operand points outside the table")
    row = table.rows[reference.row]
    if reference.column >= len(row) or not row[reference.column] or not row[0]:
        raise UnverifiedCalculation("the operand is missing or blank")
    label, raw, period = row[0], row[reference.column], table.columns[reference.column]
    if label is None or raw is None:
        raise UnverifiedCalculation("the operand is missing or blank")
    heading = "\n".join(table.context)
    if (
        not FY.fullmatch(period)
        or not YEAR_ENDED.search(heading)
        or QUARTER_ENDED.search(heading)
        or table.columns.count(period) != 1
    ):
        raise UnverifiedCalculation("the financial year cannot be bound to the source column")
    if not INR_CRORE.search(heading):
        raise UnverifiedCalculation("the source does not give an unambiguous INR crore unit")
    if _metric(label) not in MONEY_METRICS:
        raise UnverifiedCalculation("the source row is outside the supported monetary metrics")
    return Operand(chunk.id, label, f"FY{period}", _amount(raw), "INR crore")


def _metric(label: str) -> str:
    return NOTE.sub("", " ".join(label.casefold().split()))


def _consecutive(current: Operand, previous: Operand) -> None:
    if int(current.period[2:]) - int(previous.period[2:]) != 1:
        raise UnverifiedCalculation("the operands must use consecutive financial years")


def _margin(profit: Operand, revenue: Operand) -> Decimal:
    if _metric(profit.row_label) not in PROFIT or _metric(revenue.row_label) not in REVENUE:
        raise UnverifiedCalculation("an operating margin requires operating profit and revenue")
    if profit.period != revenue.period or revenue.amount <= 0:
        raise UnverifiedCalculation("operating profit and positive revenue must share a year")
    return profit.amount / revenue.amount * 100


def _basis(chunk: Chunk) -> tuple[tuple[str, ...], tuple[str, ...]]:
    table = chunk.table
    if table is None:
        raise UnverifiedCalculation("the operand is not in a retrieved table")
    final_header = max(
        index
        for index, text in enumerate(table.context)
        if YEAR_ENDED.search(text) or INR_CRORE.search(text)
    )
    return table.columns, table.context[: final_header + 1]


def calculate(
    operation: Operation, references: Sequence[CellReference], evidence: Mapping[str, Chunk]
) -> Calculation:
    """Calculate year-over-year growth, operating margin, or margin change."""
    expected = 4 if operation == "operating_margin_change" else 2
    if len(references) != expected or len(set(references)) != expected:
        raise UnverifiedCalculation("the operation requires distinct cited operands")
    operands = tuple(
        sorted(
            (_read_operand(reference, evidence) for reference in references),
            key=lambda item: (-int(item.period[2:]), _metric(item.row_label) not in PROFIT),
        )
    )
    sources = [evidence[operand.chunk_id] for operand in operands]
    if len({(source.document_id, source.page) for source in sources}) != 1:
        raise UnverifiedCalculation("all operands must be anchored to the same source page")
    if len({_basis(source) for source in sources}) != 1:
        raise UnverifiedCalculation("the operands use different financial statement headings")
    first, second = operands[:2]
    unit = "%"
    with localcontext() as decimal_context:
        decimal_context.prec = 64
        if operation == "yoy_change":
            _consecutive(first, second)
            if _metric(first.row_label) != _metric(second.row_label) or second.amount <= 0:
                raise UnverifiedCalculation(
                    "growth requires the same metric and a positive prior value"
                )
            amount = (first.amount - second.amount) / second.amount * 100
            formula = "(current amount - prior amount) / prior amount × 100"
        elif operation == "operating_margin":
            amount = _margin(first, second)
            formula = "operating profit / revenue × 100"
        else:
            prior_profit, prior_revenue = operands[2:]
            _consecutive(first, prior_profit)
            amount = _margin(first, second) - _margin(prior_profit, prior_revenue)
            unit = "percentage points"
            formula = "current operating margin - prior operating margin"
        rounded = amount.quantize(Decimal("0.01"), ROUND_HALF_UP)
    return Calculation(operation, rounded, unit, formula, operands)
