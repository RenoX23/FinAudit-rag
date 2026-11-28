from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finaudit_rag.answer_service import AnswerService
from finaudit_rag.api import create_app
from finaudit_rag.arithmetic import Operation
from finaudit_rag.contracts import AnswerPlan
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


@pytest.mark.parametrize("conflicting_amount", [False, True])
def test_financial_lookup_verifies_one_annual_amount_without_model_calls(
    conflicting_amount: bool,
) -> None:
    documents = {document.id: document for document in load_manifest(Path("data/corpus.json"))}
    source_id = "infosys-fy25-annual"
    table = Table(
        (
            "2.28 Function-wise classification of Statement of Profit and Loss",
            "(In INR crore)",
            "Year ended March 31,",
        ),
        ("Line item", "2025", "2024"),
        (("Revenue from operations 2.18", "136592", "128933"),),
    )
    chunk = Chunk(f"{source_id}:264:table:0", source_id, 264, "table", "Revenue", table)
    chunks = [chunk]
    if conflicting_amount:
        mismatch = replace(table, rows=(("Revenue from operations", "136592", "100000"),))
        chunks.append(replace(chunk, id=f"{source_id}:264:table:1", table=mismatch))
    search = HybridRetriever(SearchStore(chunks), "test", documents)

    class UnexpectedPlanner:
        def plan(
            self, question: str, operation: Operation | None, evidence: Sequence[dict[str, str]]
        ) -> AnswerPlan:
            pytest.fail("A scoped single-value lookup called the model")

    def unexpected_retrieval(_question: str) -> list[Chunk]:
        pytest.fail("A scoped single-value lookup used external retrieval")

    service = AnswerService(
        documents, unexpected_retrieval, UnexpectedPlanner(), lookup=search.lookup
    )
    question = (
        "What revenue from operations in INR crore did Infosys report "
        "in its standalone FY2024 figures in the FY2025 filing?"
    )
    with TestClient(create_app(service)) as client:
        response = client.post("/query", json={"question": question})
        assert response.status_code == 200
        body = response.json()
        if conflicting_amount:
            assert body["status"] == "abstained"
            assert body["lookup"] is None
        else:
            assert body["status"] == "answered"
            assert body["lookup"]["amount"] == "128933"
            assert body["lookup"]["period"] == "FY2024"
            assert body["lookup"]["scope"] == "standalone"
            assert body["lookup"]["citation"]["page"] == 264
        wrong_year = client.post("/query", json={"question": question.replace("FY2024", "FY2020")})
        assert wrong_year.json()["status"] == "abstained"


@pytest.mark.parametrize("conflict", [False, True])
def test_direct_calculations_validate_cells_and_reject_ambiguous_requests(conflict: bool) -> None:
    documents = {document.id: document for document in load_manifest(Path("data/corpus.json"))}
    source_id = "infosys-fy25-annual"
    table = Table(
        ("Standalone Statement of Profit and Loss", "(In INR crore)", "Year ended March 31"),
        ("Line item", "2025", "2024"),
        (
            ("Revenue from operations", "136592", "128933"),
            ("Operating profit", "30880", "28813"),
        ),
    )
    chunk = Chunk(f"{source_id}:264:table:0", source_id, 264, "table", "Annual results", table)
    chunks = [chunk]
    if conflict:
        other = replace(table, rows=(("Revenue from operations", "136000", "128933"),))
        chunks.append(replace(chunk, id=f"{source_id}:264:table:1", table=other))
    search = HybridRetriever(SearchStore(chunks), "test", documents)

    class UnexpectedPlanner:
        def plan(
            self, question: str, operation: Operation | None, evidence: Sequence[dict[str, str]]
        ) -> AnswerPlan:
            pytest.fail("A direct calculation called the model")

    def unexpected_retrieval(_question: str) -> list[Chunk]:
        pytest.fail("A direct calculation called external retrieval")

    service = AnswerService(
        documents, unexpected_retrieval, UnexpectedPlanner(), calculation=search.calculation
    )
    question = (
        "Calculate Infosys's standalone year-over-year growth in revenue from operations "
        "from FY2024 to FY2025, expressed as a percentage"
    )
    with TestClient(create_app(service)) as client:
        response = client.post("/query", json={"question": question})
        assert response.status_code == 200
        if conflict:
            assert response.json()["status"] == "abstained"
        else:
            assert response.json()["calculation"]["result"] == "5.94"
            assert {
                item["citation"]["page"] for item in response.json()["calculation"]["operands"]
            } == {264}
            filing = client.post("/query", json={"question": f"{question} in the FY2025 filing"})
            assert filing.json()["calculation"]["result"] == "5.94"
            margin = client.post(
                "/query",
                json={
                    "question": "What is Infosys's standalone operating margin "
                    "for FY2024 in the FY2025 filing?"
                },
            )
            assert margin.json()["calculation"]["result"] == "22.35"
        rejected = [
            question.replace("revenue from operations", "revenue and gross profit"),
            question.replace("Infosys", "Infosys and Wipro"),
            question.replace("FY2024 to FY2025", "FY2023 to FY2025"),
            question.replace("from FY2024 to FY2025", "for FY2025"),
            f"{question} in the FY2024 filing",
            f"{question} on PDF page 10",
            f"{question} on PDF page 264 and PDF page 277",
            "Calculate Infosys's standalone operating margin percentage change "
            "from FY2024 to FY2025",
        ]
        for query in rejected:
            result = client.post("/query", json={"question": query})
            assert result.status_code == 200
            assert result.json()["status"] == "abstained", query
