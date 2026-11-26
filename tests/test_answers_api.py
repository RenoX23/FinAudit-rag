import json
from collections.abc import Callable
from secrets import token_urlsafe

import httpx
from fastapi.testclient import TestClient

from finaudit_rag.answer_service import AnswerService
from finaudit_rag.api import create_app
from finaudit_rag.contracts import QueryResponse
from finaudit_rag.corpus import Document
from finaudit_rag.ingestion import Chunk, Table
from finaudit_rag.planner import ChatPlanner

URL = "https://example.org/annual.pdf"
DOCUMENT = Document("example", "Example annual filing", "Example", "filing", URL, "0" * 64, 300)
QUOTE = "Investment advisers must publish a monthly report."
POLICY = Chunk("example:10:text:0", "example", 10, "text", f"{QUOTE}\nContact contact@example.org.")
TABLE = Chunk(
    "example:264:table:0",
    "example",
    264,
    "table",
    "Standalone results for 2025 and 2024",
    Table(
        ("Standalone financial statements", "In ₹ crore", "Year ended March 31"),
        ("Line item", "2025", "2024"),
        (
            ("Operating profit", "30,880", "28,813"),
            ("Revenue from operations", "1,36,592", "1,28,933"),
        ),
    ),
)


def completion(plan: dict[str, object] | str) -> httpx.Response:
    content = json.dumps(plan) if isinstance(plan, dict) else plan
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def service(
    chunk: Chunk,
    handler: Callable[[httpx.Request], httpx.Response],
) -> AnswerService:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    planner = ChatPlanner("https://example.org/model", "test-model", token_urlsafe(12), client)
    return AnswerService({DOCUMENT.id: DOCUMENT}, lambda _query: [chunk], planner)


def test_policy_answer_cites_exact_page_and_masks_outbound_text() -> None:
    responses = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        serialized = body["messages"][1]["content"]
        assert "ABCPD1234E" not in serialized
        assert "contact@example.org" not in serialized
        assert "[PAN]" in serialized and "[EMAIL]" in serialized
        responses.append(serialized)
        return completion(
            {
                "decision": "excerpt",
                "excerpts": [{"chunk_id": POLICY.id, "start_line": 0, "end_line": 0}],
            }
        )

    with TestClient(create_app(service(POLICY, handler))) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        reply = client.post(
            "/query", json={"question": "What must advisers publish? PAN ABCPD1234E"}
        )
    assert reply.status_code == 200
    assert len(responses) == 1
    body = reply.json()
    assert body["status"] == "answered"
    assert body["excerpts"][0]["quote"] == QUOTE
    assert body["excerpts"][0]["citation"]["url"] == f"{URL}#page=10"
    assert body["redacted_question"].endswith("[PAN]")


def test_routes_numeric_question_and_returns_python_result_with_input_citations() -> None:
    operands = [
        {"chunk_id": TABLE.id, "row": row, "column": column} for column in (1, 2) for row in (0, 1)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["operation"] == "operating_margin_change"
        assert "row 0:" in payload["evidence"][0]["text"]
        return completion({"decision": "calculation", "operands": operands})

    with TestClient(create_app(service(TABLE, handler))) as client:
        reply = client.post(
            "/query", json={"question": "What was the standalone operating margin change in FY25?"}
        )
    assert reply.status_code == 200
    body = reply.json()
    assert body["calculation"]["result"] == "0.26"
    assert body["calculation"]["unit"] == "percentage points"
    assert [item["citation"]["page"] for item in body["calculation"]["operands"]] == [264] * 4


def test_abstains_on_missing_year_without_calling_model() -> None:
    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a missing year should abstain before an outbound request")

    with TestClient(create_app(service(TABLE, unexpected))) as client:
        reply = client.post("/query", json={"question": "What was the operating margin in FY2029?"})
    assert reply.status_code == 200
    assert reply.json()["status"] == "abstained"


def test_retries_malformed_json_once_and_rejects_invalid_line_reference() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return completion("invalid json")
        return completion(
            {
                "decision": "excerpt",
                "excerpts": [{"chunk_id": POLICY.id, "start_line": 3, "end_line": 4}],
            }
        )

    with TestClient(create_app(service(POLICY, handler))) as client:
        reply = client.post("/query", json={"question": "What must advisers publish?"})
    assert attempts == 2
    assert reply.json()["status"] == "abstained"


def test_model_error_is_a_service_error_and_invalid_input_is_not_echoed() -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with TestClient(create_app(service(POLICY, fail))) as client:
        provider_reply = client.post("/query", json={"question": "What must advisers publish?"})
        invalid_reply = client.post("/query", json={"question": {"secret": "ABCPD1234E"}})
    assert provider_reply.status_code == 503
    assert provider_reply.json()["error"]["code"] == "model_unavailable"
    assert invalid_reply.status_code == 422
    assert "ABCPD1234E" not in invalid_reply.text


def test_sensitive_questions_bypass_cache() -> None:
    class MemoryCache:
        def __init__(self) -> None:
            self.data: dict[str, QueryResponse] = {}

        def get(self, question: str) -> QueryResponse | None:
            return self.data.get(question)

        def put(self, question: str, response: QueryResponse) -> None:
            self.data[question] = response

    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return completion(
            {
                "decision": "excerpt",
                "excerpts": [{"chunk_id": POLICY.id, "start_line": 0, "end_line": 0}],
            }
        )

    responses = MemoryCache()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    planner = ChatPlanner("https://example.org/model", "test-model", token_urlsafe(12), client)
    instance = AnswerService({DOCUMENT.id: DOCUMENT}, lambda _query: [POLICY], planner, responses)
    plain = "What must advisers publish?"
    instance.answer(plain)
    instance.answer(plain)
    instance.answer(f"{plain} My PAN is ABCPD1234E.")
    instance.answer(f"{plain} My PAN is ABCPD1234E.")
    assert calls == 3
    assert set(responses.data) == {plain}
