"""Measure whether the candidate can verify the gold cells without using a model."""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from finaudit_rag.answer_service import AnswerService
from finaudit_rag.arithmetic import (
    CellReference,
    Operation,
    UnverifiedCalculation,
    _read_operand,
)
from finaudit_rag.contracts import AnswerPlan, OperandSelection
from finaudit_rag.corpus import load_manifest
from finaudit_rag.evaluation import GoldOperand, load_cases
from finaudit_rag.financial_headers import verified_financial_header
from finaudit_rag.ingestion import Chunk
from finaudit_rag.retrieval import load_export


def find_operand(expected: GoldOperand, evidence: Mapping[str, Chunk]) -> OperandSelection | None:
    """Find the one verified cell that matches the independent source label."""
    matches = []
    for chunk in evidence.values():
        if (
            chunk.table is None
            or chunk.document_id != expected.document_id
            or chunk.page != expected.page
        ):
            continue
        for row_index, row in enumerate(chunk.table.rows):
            if not row[0] or not row[0].casefold().startswith(expected.metric.casefold()):
                continue
            for column in range(1, len(row)):
                try:
                    cell = _read_operand(CellReference(chunk.id, row_index, column), evidence)
                except UnverifiedCalculation:
                    continue
                if cell.period == f"FY{expected.fiscal_year}" and cell.amount == expected.amount:
                    matches.append(
                        OperandSelection(chunk_id=chunk.id, row=row_index, column=column)
                    )
    return matches[0] if len(matches) == 1 else None


class LabeledPlanner:
    """Use only the verified references from a gold case."""

    def __init__(self, picks: list[OperandSelection]) -> None:
        self._picks = picks

    def plan(
        self, question: str, operation: Operation | None, evidence: Sequence[dict[str, str]]
    ) -> AnswerPlan:
        return AnswerPlan(decision="calculation", operands=self._picks)


def main() -> None:
    """Report how many numerical references pass the current application gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus.json"))
    parser.add_argument("--cases", type=Path, default=Path("evals"))
    args = parser.parse_args()
    docs = {item.id: item for item in load_manifest(args.manifest)}
    cases = [case for case in load_cases(args.cases, docs) if case.category == "calculation"]
    _, chunks = load_export(args.export)
    failures: Counter[str] = Counter()
    verified = 0
    for case in cases:
        pages = {(ref.document_id, ref.page) for ref in case.evidence}
        evidence = {
            chunk.id: chunk
            for chunk in chunks
            if chunk.table and (chunk.document_id, chunk.page) in pages
        }
        normalized = {
            chunk.id: verified_financial_header(docs[chunk.document_id], chunk)
            for chunk in evidence.values()
        }
        picks = [
            match
            for operand in case.operands
            if (match := find_operand(operand, normalized)) is not None
        ]
        if len(picks) != len(case.operands):
            failures["gold operand unavailable in a supported annual column"] += 1
            continue

        def retrieve(
            _query: str, source: tuple[Chunk, ...] = tuple(evidence.values())
        ) -> tuple[Chunk, ...]:
            return source

        service = AnswerService(docs, retrieve, LabeledPlanner(picks))
        result = service.answer(case.query)
        if result.calculation and result.calculation.result == case.expected_result:
            verified += 1
        else:
            failures[result.reason or "incorrect result"] += 1
    print(json.dumps({"verified": verified, "cases": len(cases), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
