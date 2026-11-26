import json
from pathlib import Path

import pytest

from finaudit_rag.ingestion import Word, aligned_numeric_tables, write_chunks


def test_aligns_amounts_under_years_without_moving_the_note_column() -> None:
    words = [
        Word("Amounts in INR crore", 90, 10),
        Word("2025", 280, 20),
        Word("2024", 360, 20),
        Word("Revenue", 90, 30),
        Word("2.18", 200, 30),
        Word("1,36,592", 280, 30),
        Word("1,28,933", 360, 30),
        Word("Cost of sales", 110, 40),
        Word("94,111", 280, 40),
        Word("89,032", 360, 40),
        Word("Gross profit", 100, 50),
        Word("42,481", 280, 50),
        Word("39,901", 360, 50),
    ]

    (table,) = aligned_numeric_tables(words)
    assert table.columns == ("Line item", "2025", "2024")
    assert table.rows[0] == ("Revenue 2.18", "1,36,592", "1,28,933")
    assert table.context[0] == "Amounts in INR crore"


def test_does_not_treat_unaligned_values_as_a_table() -> None:
    words = [
        Word("Label", 50, 10),
        Word("10", 180, 10),
        Word("20", 250, 10),
        Word("Separate label", 90, 20),
        Word("30", 240, 20),
        Word("40", 310, 20),
        Word("Last label", 70, 30),
        Word("50", 195, 30),
        Word("60", 275, 30),
    ]
    assert list(aligned_numeric_tables(words)) == []


def test_keeps_segment_revenue_and_results_under_their_own_headings() -> None:
    words = [
        Word("2025", 280, 10),
        Word("2024", 360, 10),
        Word("SEGMENT REVENUE", 90, 20),
        Word("Manufacturing", 90, 30),
        Word("6,395", 280, 30),
        Word("6,194", 360, 30),
        Word("Consumer Business", 90, 40),
        Word("10,146", 280, 40),
        Word("10,035", 360, 40),
        Word("SEGMENT RESULT", 90, 50),
        Word("Manufacturing", 90, 60),
        Word("2,029", 280, 60),
        Word("2,042", 360, 60),
        Word("Consumer Business", 90, 70),
        Word("2,929", 280, 70),
        Word("2,719", 360, 70),
    ]

    revenue, results = aligned_numeric_tables(words)
    assert revenue.context == ("2025 2024", "SEGMENT REVENUE")
    assert results.context == ("2025 2024", "SEGMENT RESULT")
    assert revenue.rows[0] == ("Manufacturing", "6,395", "6,194")
    assert results.rows[0] == ("Manufacturing", "2,029", "2,042")


def test_undecoded_glyphs_do_not_become_table_cells() -> None:
    words = [
        word
        for index, label in enumerate(("Sales", "Costs (cid:79)", "Profit", "Interest"))
        for word in (
            Word(label, 120, index * 10),
            Word("10", 220, index * 10),
            Word("20", 320, index * 10),
        )
    ]
    (table,) = aligned_numeric_tables(words)
    assert [row[0] for row in table.rows] == ["Sales", "Profit", "Interest"]


def test_failed_extraction_preserves_previous_export(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "v1",
                "documents": [
                    {
                        "id": "missing",
                        "title": "Missing filing",
                        "issuer": "Example issuer",
                        "kind": "filing",
                        "url": "https://example.org/filing.pdf",
                        "sha256": "0" * 64,
                        "pages": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    output.write_bytes(b"last successful export")
    with pytest.raises(FileNotFoundError):
        write_chunks(manifest, tmp_path, output, "table-aware")
    assert output.read_bytes() == b"last successful export"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chunks.jsonl", "manifest.json"]
