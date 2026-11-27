from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from finaudit_rag.answer_service import _source_text
from finaudit_rag.arithmetic import CellReference, UnverifiedCalculation, calculate
from finaudit_rag.corpus import load_manifest
from finaudit_rag.financial_headers import verified_financial_header
from finaudit_rag.ingestion import Chunk, Table

SOURCE = "filing:264:table:0"
PROFIT = ("Operating profit", "30,880", "28,813")
REVENUE = ("Revenue from operations 2.18", "1,36,592", "1,28,933")


def test_verified_source_uses_annual_columns_and_rejects_quarters() -> None:
    source = next(
        item for item in load_manifest(Path("data/corpus.json")) if item.id == "tcs-fy25-q4"
    )
    chunk = Chunk(
        "tcs-fy25-q4:7:table:0",
        source.id,
        7,
        "table",
        "Revenue from operations",
        Table(
            (
                "Audited Consolidated Statement of Financial Results",
                "Three months ended Year ended",
            ),
            (
                "Line item",
                "March 31, 2025 (column 1)",
                "December 31, 2024 (column 2)",
                "March 31, 2024 (column 3)",
                "March 31, 2025 (column 4)",
                "March 31, 2024 (column 5)",
            ),
            (
                ("Revenue from operations", "64,479", "63,973", "61,237", "2,55,324", "2,40,893"),
                ("Account number", "", "", "", "123456789", "987654321"),
            ),
        ),
    )
    verified = verified_financial_header(source, chunk)
    evidence = _source_text(verified)
    assert "4=2025 | 5=2024" in evidence
    assert "4=2,55,324 | 5=2,40,893" in evidence
    assert "64,479" not in evidence
    assert "Account number" not in evidence
    assert "123456789" not in evidence
    cells = {chunk.id: verified}
    result = calculate("yoy_change", [CellReference(chunk.id, 0, n) for n in (4, 5)], cells)
    assert result.result == Decimal("5.99")
    with pytest.raises(UnverifiedCalculation, match="financial year"):
        calculate("yoy_change", [CellReference(chunk.id, 0, n) for n in (1, 3)], cells)
    assert verified_financial_header(replace(source, sha256="0" * 64), chunk) is chunk


def make_evidence(
    context: tuple[str, ...] = ("In ₹ crore", "Year ended March 31,"),
    periods: tuple[str, ...] = ("Line item", "2025", "2024"),
    rows: tuple[tuple[str, ...], ...] = (PROFIT, REVENUE),
) -> dict[str, Chunk]:
    return {
        SOURCE: Chunk(SOURCE, "filing", 264, "table", "evidence", Table(context, periods, rows))
    }


def test_calculates_margin_change_in_percentage_points() -> None:
    refs = [CellReference(SOURCE, row, col) for col in (2, 1) for row in (1, 0)]
    result = calculate("operating_margin_change", refs, make_evidence())
    assert result.result == Decimal("0.26")
    assert result.unit == "percentage points"
    assert [operand.period for operand in result.operands] == [
        "FY2025",
        "FY2025",
        "FY2024",
        "FY2024",
    ]


def test_calculates_growth_on_the_same_labeled_metric() -> None:
    refs = [CellReference(SOURCE, 1, 1), CellReference(SOURCE, 1, 2)]
    result = calculate("yoy_change", refs, make_evidence())
    assert result.result == Decimal("5.94")


def test_refuses_to_combine_different_statement_scopes_on_one_page() -> None:
    evidence = make_evidence(("Standalone statement", "In ₹ crore", "Year ended"))
    other_id = "filing:264:table:1"
    other = Table(
        ("Consolidated statement", "In ₹ crore", "Year ended"),
        ("Line item", "2025", "2024"),
        (REVENUE,),
    )
    evidence[other_id] = Chunk(other_id, "filing", 264, "table", "another statement", other)
    with pytest.raises(UnverifiedCalculation, match="different financial statement headings"):
        calculate(
            "operating_margin",
            [CellReference(SOURCE, 0, 1), CellReference(other_id, 0, 1)],
            evidence,
        )


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (make_evidence(("Year ended March 31",)), "unit"),
        (make_evidence(("In ₹ crore", "Year ended", "Three months ended")), "financial year"),
        (make_evidence(periods=("Line item", "2025", "2025")), "financial year"),
        (
            make_evidence(rows=(PROFIT, ("Revenue from operations", "1,36,592", "0"))),
            "positive prior",
        ),
        (
            make_evidence(rows=(PROFIT, ("Revenue from operations", "1,36,59", "99"))),
            "supported amount",
        ),
    ],
)
def test_rejects_ambiguous_or_invalid_inputs(evidence: dict[str, Chunk], message: str) -> None:
    refs = [CellReference(SOURCE, 1, 1), CellReference(SOURCE, 1, 2)]
    with pytest.raises(UnverifiedCalculation, match=message):
        calculate("yoy_change", refs, evidence)
