"""Recognize a single reported annual statement value."""

import re
from dataclasses import dataclass
from typing import Literal, cast

from finaudit_rag.arithmetic import MONEY_METRICS, Operation

ReportingScope = Literal["standalone", "consolidated"]
METRIC = "|".join(re.escape(label) for label in sorted(MONEY_METRICS, key=len, reverse=True))
ISSUER = r"(?P<issuer>[\w .&-]+?)"
SCOPE = r"(?P<scope>standalone|consolidated)"
YEAR = r"FY\s*(?:20)?(?P<year>\d{2})"
SOURCE = (
    r"(?: (?:in|from) (?:the )?FY\s*(?:20)?(?P<filing>\d{2}) "
    r"(?:filing|annual report|audited results))?"
    r"(?: on PDF page (?P<page>[1-9]\d{0,3}))?\??"
)
REPORTED = re.compile(
    rf"what (?P<metric>{METRIC})(?: in INR crore)? did {ISSUER} report "
    rf"(?:in|for) (?:its )?{SCOPE} {YEAR}(?: figures)?{SOURCE}",
    re.IGNORECASE,
)
DIRECT = re.compile(
    rf"what (?:was|is) {ISSUER}['’]s {SCOPE} {YEAR} (?P<metric>{METRIC})"
    rf"(?: in INR crore)?{SOURCE}",
    re.IGNORECASE,
)
FILING_YEAR = re.compile(r"\bFY(?:20)?\d{2}[-–](?:20)?(\d{2})\b", re.IGNORECASE)
INCOME_STATEMENT = re.compile(r"\bstatement of (?:financial results|profit and loss)\b", re.I)
OWNER = rf"{ISSUER}['’]s {SCOPE}"
PRIOR = r"FY\s*(?:20)?(?P<prior>\d{2})"
GROWTH = re.compile(
    rf"(?:calculate|compute) {OWNER} (?:year-over-year|yoy) growth in (?P<metric>{METRIC}) "
    rf"from {PRIOR} to {YEAR}(?:,? (?:expressed )?as a percentage)?{SOURCE}",
    re.I,
)
MARGIN = re.compile(
    rf"(?:what (?:is|was)|calculate) {OWNER} operating margin (?:for|in) {YEAR}"
    rf"(?:, calculated from operating profit and revenue from operations| as a percentage)?"
    rf"{SOURCE}",
    re.I,
)
MARGIN_CHANGE = re.compile(
    rf"how many percentage points did (?:the )?{ISSUER}(?:['’]s)? {SCOPE} operating margin "
    rf"change (?:between|from) {PRIOR} (?:and|to) {YEAR}{SOURCE}",
    re.I,
)


@dataclass(frozen=True)
class LookupRequest:
    issuer: str
    scope: ReportingScope
    year: str
    metric: str
    filing_year: str | None
    page: int | None


@dataclass(frozen=True)
class CalculationRequest:
    source: LookupRequest
    operation: Operation
    years: tuple[str, ...]
    metrics: tuple[str, ...]


def parse_calculation(question: str) -> CalculationRequest | None:
    """Recognize one scoped annual calculation with explicit reporting periods."""
    normalized = " ".join(question.split()).rstrip(".?")
    match = GROWTH.fullmatch(normalized)
    operation: Operation = "yoy_change"
    if match is None:
        match = MARGIN.fullmatch(normalized)
        operation = "operating_margin"
    if match is None:
        match = MARGIN_CHANGE.fullmatch(normalized)
        operation = "operating_margin_change"
    if match is None:
        return None
    groups = match.groupdict()
    year = f"20{match['year']}"
    years: tuple[str, ...] = (year,)
    if operation != "operating_margin":
        prior = f"20{match['prior']}"
        if int(year) - int(prior) != 1:
            return None
        years = (year, prior)
    metrics = (
        (match["metric"].casefold(),)
        if operation == "yoy_change"
        else ("operating profit", "revenue from operations")
    )
    return CalculationRequest(
        source=LookupRequest(
            issuer=match["issuer"].casefold(),
            scope=cast(ReportingScope, match["scope"].casefold()),
            year=year,
            metric=metrics[0],
            filing_year=groups.get("filing"),
            page=int(match["page"]) if match["page"] else None,
        ),
        operation=operation,
        years=years,
        metrics=metrics,
    )


def parse_lookup(question: str) -> LookupRequest | None:
    """Bind the reporting year independently of the optional filing date."""
    normalized = " ".join(question.split())
    match = REPORTED.fullmatch(normalized) or DIRECT.fullmatch(normalized)
    if match is None:
        return None
    return LookupRequest(
        issuer=match["issuer"].casefold(),
        scope=cast(ReportingScope, match["scope"].casefold()),
        year=f"20{match['year']}",
        metric=match["metric"].casefold(),
        filing_year=match["filing"],
        page=int(match["page"]) if match["page"] else None,
    )
