"""Extract page-cited text and tables from the pinned PDF snapshots."""

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import pdfplumber
from pdfplumber.page import Page

from finaudit_rag.corpus import Document, load_manifest, verify_pdf

Mode = Literal["baseline", "table-aware"]
Cell = str | None
Grid = tuple[tuple[Cell, ...], ...]
NUMBER = re.compile(r"(?:[-–—]|[+-]?(?:\(\d[\d,]*(?:\.\d+)?\)|\d[\d,]*(?:\.\d+)?)(?:%|\*)?)")
YEAR = re.compile(r"(?:19|20|21)\d{2}")
ROW_ALIGNMENT = 4.0
MAX_TABLE_ROWS = 8


@dataclass(frozen=True)
class Word:
    text: str
    x1: float
    top: float


@dataclass(frozen=True)
class Table:
    context: tuple[str, ...]
    columns: tuple[str, ...]
    rows: Grid


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    page: int
    kind: Literal["text", "table"]
    text: str
    table: Table | None = None


def chunk_from_record(record: Mapping[str, Any]) -> Chunk:
    """Read a chunk serialized by ingestion or stored in the database."""
    raw_table = record["table"]
    table: Table | None = None
    if raw_table is not None:
        table = Table(
            tuple(raw_table["context"]),
            tuple(raw_table["columns"]),
            tuple(tuple(row) for row in raw_table["rows"]),
        )
    return Chunk(
        record["id"],
        record["document_id"],
        record["page"],
        record["kind"],
        record["text"],
        table,
    )


def _text_windows(text: str, mode: Mode, size: int = 800, overlap: int = 100) -> Iterator[str]:
    if mode == "baseline":
        for start in range(0, len(text), size - overlap):
            if block := text[start : start + size].strip():
                yield block
            if start + size >= len(text):
                break
        return

    current: list[str] = []
    length = 0
    for line in text.splitlines():
        if length + len(line) > size and current:
            yield "\n".join(current)
            current = []
            length = 0
        if len(line) > size:
            yield from _text_windows(line, "baseline", size, overlap)
        elif line.strip():
            current.append(line.strip())
            length += len(line) + 1
    if current:
        yield "\n".join(current)


def _lines(words: Iterable[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda item: item.top):
        if not lines or word.top - lines[-1][0].top > 2.5:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda item: item.x1) for line in lines]


def _value_suffix(line: list[Word]) -> tuple[Word, ...]:
    for index in range(len(line) - 1, -1, -1):
        if not NUMBER.fullmatch(line[index].text):
            if any(char.isalpha() for word in line[: index + 1] for char in word.text):
                return tuple(line[index + 1 :])
            break
    return ()


def _rows_align(left: tuple[Word, ...], right: tuple[Word, ...]) -> bool:
    return all(
        abs(a.x1 - b.x1) <= ROW_ALIGNMENT for a, b in zip(left[-2:], right[-2:], strict=True)
    )


def _column_labels(context: list[list[Word]], width: int) -> tuple[str, ...]:
    for index in range(len(context) - 1, -1, -1):
        line = context[index]
        if len(line) == width and all(YEAR.fullmatch(word.text) for word in line):
            labels = [word.text for word in line]
            if index:
                preceding = " ".join(word.text for word in context[index - 1])
                dates = re.findall(r"[A-Z][a-z]+ \d{1,2},", preceding)
                if len(dates) == width:
                    labels = [f"{date} {year}" for date, year in zip(dates, labels, strict=True)]
            if len(set(labels)) != width:
                labels = [f"{label} (column {index})" for index, label in enumerate(labels, 1)]
            return ("Line item", *labels)
    return ("Line item", *(f"Column {index}" for index in range(1, width + 1)))


def _build_numeric_table(
    lines: list[list[Word]], group: list[tuple[int, tuple[Word, ...]]], previous_end: int
) -> Table | None:
    if len(group) < 3:
        return None
    counts = Counter(len(cells) for _, cells in group)
    width = min(counts, key=lambda count: (-counts[count], count))
    anchors = next(cells for _, cells in group if len(cells) == width)
    context = lines[max(previous_end + 1, group[0][0] - 6) : group[0][0]]
    columns = _column_labels(context, width)
    rows = [
        (
            " ".join(word.text for word in lines[index][:-width]),
            *(word.text for word in cells[-width:]),
        )
        for index, cells in group
        if len(cells) >= width
        and all(
            abs(actual.x1 - expected.x1) <= ROW_ALIGNMENT
            for actual, expected in zip(cells[-width:], anchors, strict=True)
        )
    ]
    if len(rows) < 3:
        return None
    return Table(
        context=tuple(" ".join(word.text for word in line) for line in context),
        columns=columns,
        rows=tuple(rows),
    )


def aligned_numeric_tables(words: Iterable[Word]) -> Iterator[Table]:
    """Read repeated right-aligned numeric cells in unruled financial tables."""
    lines = _lines(words)
    group: list[tuple[int, tuple[Word, ...]]] = []
    previous_end = -1
    for index, line in enumerate(lines):
        cells = _value_suffix(line)
        if not 2 <= len(cells) <= 10:
            continue
        if group and (index - group[-1][0] > 3 or not _rows_align(group[-1][1], cells)):
            if table := _build_numeric_table(lines, group, previous_end):
                yield table
                previous_end = group[-1][0]
            group = []
        group.append((index, cells))
    if table := _build_numeric_table(lines, group, previous_end):
        yield table


def _ruled_tables(page: Page) -> Iterator[Table]:
    for candidate in page.find_tables():
        grid = [
            tuple(" ".join(cell.split()) if cell else None for cell in row)
            for row in candidate.extract()
            if any(cell and cell.strip() for cell in row)
        ]
        if len(grid) < 2 or len(grid[0]) < 2 or sum(cell is not None for cell in grid[0]) < 2:
            continue
        headings = tuple(cell or f"Column {index}" for index, cell in enumerate(grid[0], 1))
        yield Table(context=(), columns=headings, rows=tuple(grid[1:]))


def _table_text(table: Table) -> str:
    lines = [*table.context, " | ".join(table.columns)]
    lines.extend(" | ".join(cell or "" for cell in row) for row in table.rows)
    return "\n".join(lines)


def _page_chunks(document: Document, page: Page, page_number: int, mode: Mode) -> Iterator[Chunk]:
    prefix = f"{document.id}:{page_number}"
    text: str = page.extract_text() or ""
    for ordinal, block in enumerate(_text_windows(text, mode)):
        yield Chunk(f"{prefix}:text:{ordinal}", document.id, page_number, "text", block)

    if mode == "baseline":
        return
    words = (Word(raw["text"], raw["x1"], raw["top"]) for raw in page.extract_words())
    aligned = list(aligned_numeric_tables(words)) if document.kind == "filing" else []
    tables = aligned or list(_ruled_tables(page))
    ordinal = 0
    for table in tables:
        for start in range(0, len(table.rows), MAX_TABLE_ROWS):
            part = Table(table.context, table.columns, table.rows[start : start + MAX_TABLE_ROWS])
            yield Chunk(
                f"{prefix}:table:{ordinal}",
                document.id,
                page_number,
                "table",
                _table_text(part),
                part,
            )
            ordinal += 1


def extract_chunks(document: Document, pdf_dir: Path, mode: Mode) -> Iterator[Chunk]:
    """Yield stable, one-based page citations from a checksum-verified document."""
    verify_pdf(document, pdf_dir)
    with pdfplumber.open(pdf_dir / f"{document.id}.pdf") as pdf:
        if len(pdf.pages) != document.pages:
            raise ValueError(f"{document.id}: page count differs from the manifest")
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                yield from _page_chunks(document, page, page_number, mode)
            finally:
                page.close()


def write_chunks(manifest: Path, pdf_dir: Path, output: Path, mode: Mode) -> int:
    """Replace the export atomically only after every source has been extracted."""
    documents = load_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as file:
        pending = Path(file.name)
    try:
        with pending.open("w", encoding="utf-8", newline="\n") as file:
            for document in documents:
                document_count = 0
                for chunk in extract_chunks(document, pdf_dir, mode):
                    file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    document_count += 1
                if not document_count:
                    raise ValueError(f"{document.id}: no extractable chunks")
                count += document_count
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return count


def main() -> None:
    """Export source-cited baseline or table-aware chunks as JSON Lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("baseline", "table-aware"), default="table-aware")
    args = parser.parse_args()
    try:
        count = write_chunks(args.manifest, args.pdf_dir, args.output, args.mode)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Exported {count} {args.mode} chunks to {args.output}.")


if __name__ == "__main__":
    main()
