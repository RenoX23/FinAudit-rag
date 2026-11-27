"""Bind verified annual headers to the pinned source pages that supplied them."""

from dataclasses import replace

from finaudit_rag.corpus import Document
from finaudit_rag.ingestion import Chunk

# The Infosys standalone section begins at physical page 182 and ends before page 265.
INFOSYS_SNAPSHOT = "6224d11244d3f44f7e95901428a8ff23164f383cd1960e02a47483c48612a3c5"
TCS_SNAPSHOTS = {
    "e1b9367f93ba7a31c01945fab2f57a2037954c5be4bfdca436b151306887e988": "2024",
    "10ae2cfd51f966036774091acf830e31ed6f1d330dd63b4a13763607787d6fb7": "2025",
}


def verified_financial_header(document: Document, chunk: Chunk) -> Chunk:
    """Recover annual column labels from visually verified, digest-bound PDF headers."""
    table = chunk.table
    if table is None or chunk.document_id != document.id:
        return chunk
    if (
        document.sha256 == INFOSYS_SNAPSHOT
        and chunk.page == 264
        and table.context
        and table.context[0] == "2.28 Function-wise classification of Statement of Profit and Loss"
        and table.columns == ("Line item", "2025", "2024")
    ):
        context = ("Standalone financial statements", *table.context)
        return replace(chunk, table=replace(table, context=context))

    year = TCS_SNAPSHOTS.get(document.sha256)
    if year is None or chunk.page not in {7, 16}:
        return chunk
    scope = "Consolidated" if chunk.page == 7 else "Standalone"
    heading = f"Audited {scope} Statement of Financial Results"
    prior = str(int(year) - 1)
    columns = (
        "Line item",
        f"March 31, {year} (column 1)",
        f"December 31, {prior} (column 2)",
        f"March 31, {prior} (column 3)",
        f"March 31, {year} (column 4)",
        f"March 31, {prior} (column 5)",
    )
    if heading not in table.context or columns != table.columns:
        return chunk
    if not any("year ended" in line.casefold() for line in table.context):
        return chunk

    # The source's Year ended span covers the last two columns. Keep all row indices.
    annual_labels = (columns[0], *columns[1:4], year, prior)
    annual_context = (heading, "(In INR crore)", "Year ended March 31")
    return replace(chunk, table=replace(table, columns=annual_labels, context=annual_context))
