from collections.abc import Sequence
from pathlib import Path

import pytest

from finaudit_rag.corpus import load_manifest
from finaudit_rag.ingestion import Chunk, Table
from finaudit_rag.retrieval import HybridRetriever, tokens

CIRCULAR = "SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/80"


class SearchStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_chunks(self, collection_id: str) -> list[Chunk]:
        return self.chunks

    def cosine_search(
        self, collection_id: str, query_vector: Sequence[float], limit: int
    ) -> list[Chunk]:
        return self.chunks[1:limit]


def test_keeps_compound_circular_ids_searchable() -> None:
    assert CIRCULAR.casefold() in tokens(f"See {CIRCULAR}.")
    assert "mirsd" in tokens(CIRCULAR)


def test_circular_issuer_page_ranks_above_a_better_scoring_cross_reference() -> None:
    exact = Chunk("notice:1:text:0", "notice", 1, "text", f"Circular\n{CIRCULAR}\nNew guidelines")
    cited = Chunk("another:1:text:0", "another", 1, "text", f"See {CIRCULAR}\nRelated guidelines")
    store = SearchStore([exact, cited])
    retriever = HybridRetriever(store, "test", {})

    def rerank(query: str, passages: Sequence[str]) -> list[float]:
        return [0.1 if "\nCircular\n" in passage else 0.9 for passage in passages]

    results = retriever.search(f"Find {CIRCULAR}", [1.0], rerank)
    assert results[0] == exact
    assert cited in results


def test_invalid_reranker_output_fails_explicitly() -> None:
    chunks = [
        Chunk("a:1:text:0", "a", 1, "text", "Other report"),
        Chunk("b:3:text:0", "b", 3, "text", "Balance sheet"),
    ]
    retriever = HybridRetriever(SearchStore(chunks), "test", {})
    with pytest.raises(ValueError, match="invalid scores"):
        retriever.search("balance sheet", [1.0], lambda _query, _passages: [float("nan")])


def test_calculation_ranks_source_eligible_tables_outside_the_dense_shortlist() -> None:
    documents = {document.id: document for document in load_manifest(Path("data/corpus.json"))}
    source_id = "infosys-fy25-annual"
    repeated = "Infosys standalone revenue growth FY2024 FY2025 PDF page 264"
    noise = [
        Chunk(f"{source_id}:{page}:text:0", source_id, page, "text", repeated)
        for page in range(1, 24)
    ]
    table = Table(
        (
            "2.28 Function-wise classification of Statement of Profit and Loss",
            "(In INR crore)",
            "Year ended March 31,",
        ),
        ("Line item", "2025", "2024"),
        (("Revenue from operations", "136592", "128933"),),
    )
    statement = Chunk(f"{source_id}:264:table:0", source_id, 264, "table", "Revenue", table)
    retriever = HybridRetriever(SearchStore([*noise, statement]), "test", documents)

    def rerank(_query: str, passages: Sequence[str]) -> list[float]:
        assert len(passages) == 1
        assert passages[0].startswith("Standalone financial statements\n")
        return [1.0]

    question = "Calculate Infosys standalone revenue growth from FY2024 to FY2025"
    assert retriever.search(question, [1.0], rerank) == [statement]
    assert retriever.search(f"{question} using PDF page 264", [1.0], rerank) == [statement]
    assert retriever.search(f"{question} using PDF page 265", [1.0], rerank) == []
    assert retriever.search(question.replace("standalone", "consolidated"), [1.0], rerank) == []
    assert retriever.search(question.replace("FY2025", "FY2029"), [1.0], rerank) == []
