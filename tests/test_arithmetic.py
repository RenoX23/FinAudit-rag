from decimal import Decimal

import pytest

from finaudit_rag.arithmetic import CellReference, UnverifiedCalculation, calculate
from finaudit_rag.ingestion import Chunk, Table

SOURCE = "filing:264:table:0"
PROFIT = ("Operating profit", "30,880", "28,813")
REVENUE = ("Revenue from operations 2.18", "1,36,592", "1,28,933")


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
