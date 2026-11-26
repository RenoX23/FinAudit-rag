from collections.abc import Sequence

import pytest

from finaudit_rag.ingestion import Chunk
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
    retriever = HybridRetriever(store, "test")

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
    retriever = HybridRetriever(SearchStore(chunks), "test")
    with pytest.raises(ValueError, match="invalid scores"):
        retriever.search("balance sheet", [1.0], lambda _query, _passages: [float("nan")])
