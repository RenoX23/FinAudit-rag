"""Answer from retrieved excerpts and validated table references."""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from finaudit_rag.arithmetic import (
    FY,
    MONEY_METRICS,
    CellReference,
    Operand,
    Operation,
    UnverifiedCalculation,
    _metric,
    calculate,
)
from finaudit_rag.contracts import (
    AnswerPlan,
    CalculationResult,
    Citation,
    CitedExcerpt,
    CitedLookup,
    CitedOperand,
    QueryResponse,
)
from finaudit_rag.corpus import Document
from finaudit_rag.financial_headers import verified_financial_header
from finaudit_rag.ingestion import Chunk
from finaudit_rag.lookup import LookupRequest, parse_lookup
from finaudit_rag.privacy import mask_sensitive_text

YEAR_REFERENCE = re.compile(r"\bFY\s*(?:20)?(\d{2})\b", re.IGNORECASE)
BARE_YEAR = re.compile(r"\b(?:19|20|21)\d{2}\b")
PDF_PAGE = re.compile(r"\bPDF page (\d{1,4})\b", re.IGNORECASE)
SCOPE = re.compile(r"\b(standalone|consolidated)\b", re.IGNORECASE)
CHANGE = re.compile(r"\b(?:change|growth|increase|decrease|difference|year.over.year|yoy)\b")
UNSUPPORTED_INSTRUCTION = re.compile(r"\b(?:calculate|compute|derive)\b")
LIVE_RULE = re.compile(
    r"\b(?:latest|current) (?:rule|circular|directions|requirements?)\b|\bas of (?:today|now)\b",
    re.IGNORECASE,
)
SUBANNUAL = re.compile(
    r"\b[QH][1-4](?=\b|FY)|\b(?:quarter(?:ly)?|half[- ]year|three months|"
    r"six months|[36] months|monthly)\b",
    re.IGNORECASE,
)
MONTH = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"(?:\d{1,2},?\s+)?(?:(?:19|20|21)\d{2}|FY\s*(?:20)?\d{2})\b",
    re.IGNORECASE,
)
ISO_DATE = re.compile(r"\b(?:19|20|21)\d{2}-\d{2}-\d{2}\b")


class Planner(Protocol):
    def plan(
        self, question: str, operation: Operation | None, evidence: Sequence[dict[str, str]]
    ) -> AnswerPlan: ...


class AnswerCache(Protocol):
    def get(self, question: str) -> QueryResponse | None: ...

    def put(self, question: str, response: QueryResponse) -> None: ...


def _operation(question: str) -> Operation | None:
    normalized = question.casefold()
    if "operating margin" in normalized:
        return "operating_margin_change" if CHANGE.search(normalized) else "operating_margin"
    if CHANGE.search(normalized) and (
        YEAR_REFERENCE.search(question) or "year" in normalized or "yoy" in normalized
    ):
        return "yoy_change"
    return None


def _years(question: str, include_bare: bool) -> set[str]:
    years = {f"20{year}" for year in YEAR_REFERENCE.findall(question)}
    if include_bare:
        years.update(BARE_YEAR.findall(question))
    return years


def _source_text(chunk: Chunk) -> str:
    if chunk.table is None:
        source = mask_sensitive_text(chunk.text).text
        return "\n".join(f"line {index}: {line}" for index, line in enumerate(source.splitlines()))
    annual = [
        (index, label)
        for index, label in enumerate(chunk.table.columns)
        if index and FY.fullmatch(label)
    ]
    headers = " | ".join(f"{index}={label}" for index, label in annual)
    rows = [
        f"row {index}: {row[0]} | "
        + " | ".join(f"{column}={row[column] or ''}" for column, _ in annual if column < len(row))
        for index, row in enumerate(chunk.table.rows)
        if row and row[0] and _metric(row[0]) in MONEY_METRICS
    ]
    return "\n".join([*chunk.table.context, f"columns: {headers}", *rows])


class AnswerService:
    """Validate model selections and generate response values locally."""

    def __init__(
        self,
        documents: Mapping[str, Document],
        retrieve: Callable[[str], Sequence[Chunk]],
        planner: Planner,
        cache: AnswerCache | None = None,
        lookup: Callable[[LookupRequest], tuple[Chunk, Operand] | None] | None = None,
    ) -> None:
        self._documents = documents
        self._retrieve = retrieve
        self._planner = planner
        self._cache = cache
        self._lookup = lookup

    def answer(self, question: str) -> QueryResponse:
        """Return verifiable evidence or an explicit abstention."""
        masked = mask_sensitive_text(question)
        cache = self._cache if not masked.replacements else None
        if cache is not None and (cached := cache.get(masked.text)) is not None:
            return cached
        response = self._answer(masked.text)
        if cache is not None and response.status == "answered":
            cache.put(masked.text, response)
        return response

    def _answer(self, question: str) -> QueryResponse:
        def abstain(reason: str) -> QueryResponse:
            return QueryResponse(status="abstained", redacted_question=question, reason=reason)

        operation = _operation(question)
        if LIVE_RULE.search(question):
            return abstain("The corpus contains dated source snapshots.")
        if operation is not None and (
            SUBANNUAL.search(question) or MONTH.search(question) or ISO_DATE.search(question)
        ):
            return abstain("Calculations require fiscal-year labels without a specific date.")
        if operation is None and UNSUPPORTED_INSTRUCTION.search(question.casefold()):
            return abstain("The requested calculation is outside the supported operations.")
        if self._lookup is not None and (lookup := parse_lookup(question)) is not None:
            match = self._lookup(lookup)
            if match is None:
                return abstain("The source statements did not verify a unique requested amount.")
            chunk, operand = match
            return QueryResponse(
                status="answered",
                redacted_question=question,
                lookup=CitedLookup(
                    metric=operand.row_label,
                    period=operand.period,
                    amount=operand.amount,
                    unit=operand.unit,
                    citation=self._citation(chunk),
                    scope=lookup.scope,
                ),
            )
        issuers = {
            document.issuer.casefold()
            for document in self._documents.values()
            if re.search(rf"\b{re.escape(document.issuer)}\b", question, re.IGNORECASE)
        }
        scopes = {scope.casefold() for scope in SCOPE.findall(question)}
        if operation is not None and (len(issuers) != 1 or len(scopes) != 1):
            return abstain("Calculations require one named issuer and one reporting scope.")
        requested_years = _years(question, include_bare=operation is not None)
        pages = {int(page) for page in PDF_PAGE.findall(question)}
        expected_kind = "table" if operation is not None else "text"
        matching = {
            chunk.id: chunk
            for chunk in self._retrieve(question)
            if chunk.document_id in self._documents
            and 1 <= chunk.page <= self._documents[chunk.document_id].pages
            and chunk.kind == expected_kind
            and (not pages or chunk.page in pages)
            and (not issuers or self._documents[chunk.document_id].issuer.casefold() in issuers)
            and (
                not requested_years
                or requested_years.intersection(_years(chunk.text, include_bare=True))
            )
        }
        chunks = dict(list(matching.items())[:5])
        if not chunks:
            return abstain("The corpus did not return usable evidence.")
        if operation is not None:
            chunks = {
                chunk.id: verified_financial_header(self._documents[chunk.document_id], chunk)
                for chunk in chunks.values()
            }
        context = [
            {
                "chunk_id": chunk.id,
                "document": self._documents[chunk.document_id].title,
                "kind": chunk.kind,
                "text": mask_sensitive_text(_source_text(chunk)).text[:2400],
            }
            for chunk in chunks.values()
        ]
        plan = self._planner.plan(question, operation, context)
        if plan.decision == "calculation" and operation is not None and not plan.excerpts:
            try:
                calculation = self._calculation(question, operation, plan, chunks)
            except UnverifiedCalculation as error:
                return abstain(str(error))
            return QueryResponse(
                status="answered", redacted_question=question, calculation=calculation
            )
        if plan.decision == "excerpt" and operation is None and not plan.operands:
            excerpts = self._excerpts(plan, chunks)
            excerpt_years = {
                year
                for excerpt in excerpts
                for year in _years(chunks[excerpt.citation.chunk_id].text, include_bare=True)
            }
            if excerpts and requested_years.issubset(excerpt_years):
                return QueryResponse(
                    status="answered", redacted_question=question, excerpts=excerpts
                )
        return abstain("The retrieved evidence could not verify an answer.")

    def _citation(self, chunk: Chunk) -> Citation:
        document = self._documents[chunk.document_id]
        return Citation(
            document_id=document.id,
            title=document.title,
            page=chunk.page,
            url=f"{document.url}#page={chunk.page}",
            chunk_id=chunk.id,
        )

    def _excerpts(self, plan: AnswerPlan, chunks: Mapping[str, Chunk]) -> list[CitedExcerpt]:
        excerpts: list[CitedExcerpt] = []
        for selection in plan.excerpts:
            chunk = chunks.get(selection.chunk_id)
            if chunk is None or chunk.kind != "text":
                return []
            lines = mask_sensitive_text(chunk.text).text.splitlines()
            start, end = selection.start_line, selection.end_line
            if not (start <= end < min(len(lines), start + 8)):
                return []
            quote = " ".join(" ".join(lines[start : end + 1]).split())
            if not 12 <= len(quote) <= 700:
                return []
            excerpts.append(CitedExcerpt(quote=quote, citation=self._citation(chunk)))
        return excerpts

    def _calculation(
        self,
        question: str,
        operation: Operation,
        plan: AnswerPlan,
        chunks: Mapping[str, Chunk],
    ) -> CalculationResult:
        references = [CellReference(pick.chunk_id, pick.row, pick.column) for pick in plan.operands]
        result = calculate(operation, references, chunks)
        if operation == "yoy_change":
            label = re.sub(r"\s+\d+(?:\.\d+)+$", "", result.operands[0].row_label.casefold())
            query = question.casefold()
            matches_revenue = label == "revenue from operations" and "revenue" in query
            if label not in query and not matches_revenue:
                raise UnverifiedCalculation("the source row does not match the requested metric")
        requested_years = {f"FY{year}" for year in _years(question, include_bare=True)}
        if requested_years and (
            not requested_years.issubset({operand.period for operand in result.operands})
            or result.operands[0].period != max(requested_years)
        ):
            raise UnverifiedCalculation(
                "the cited operands do not match the requested financial years"
            )
        scopes = {scope.casefold() for scope in SCOPE.findall(question)}
        if scopes:
            for operand in result.operands:
                table = chunks[operand.chunk_id].table
                source_scopes = (
                    {match.casefold() for match in SCOPE.findall(" ".join(table.context))}
                    if table
                    else set()
                )
                if len(scopes) != 1 or scopes != source_scopes:
                    raise UnverifiedCalculation("the requested reporting scope is ambiguous")
        return CalculationResult(
            operation=result.operation,
            result=result.result,
            unit=result.unit,
            formula=result.formula,
            operands=[
                CitedOperand(
                    metric=operand.row_label,
                    period=operand.period,
                    amount=operand.amount,
                    unit=operand.unit,
                    citation=self._citation(chunks[operand.chunk_id]),
                )
                for operand in result.operands
            ],
        )
