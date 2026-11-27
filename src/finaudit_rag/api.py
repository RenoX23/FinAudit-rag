"""FastAPI entry point for source-cited responses."""

import hashlib
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis import Redis

from finaudit_rag.answer_service import AnswerService
from finaudit_rag.cache import RedisAnswerCache
from finaudit_rag.contracts import QueryRequest, QueryResponse
from finaudit_rag.corpus import load_manifest
from finaudit_rag.ingestion import Chunk
from finaudit_rag.models import RERANKER_REVISION, CpuModels
from finaudit_rag.planner import PROMPT_VERSION, ChatPlanner, PlannerUnavailable
from finaudit_rag.retrieval import RETRIEVAL_VERSION, HybridRetriever
from finaudit_rag.vector_store import PgChunkStore

REQUIRED = (
    "FINAUDIT_DATABASE_URL",
    "FINAUDIT_COLLECTION_ID",
    "FINAUDIT_LLM_ENDPOINT",
    "FINAUDIT_LLM_MODEL",
    "FINAUDIT_LLM_API_KEY",
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _runtime() -> tuple[AnswerService, Callable[[], None]]:
    load_dotenv(dotenv_path=Path(".env"))
    missing = [name for name in REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing environment settings: {', '.join(missing)}")
    manifest = Path(os.environ.get("FINAUDIT_MANIFEST", "data/corpus.json"))
    documents = {document.id: document for document in load_manifest(manifest)}
    collection = os.environ["FINAUDIT_COLLECTION_ID"]
    if not collection.startswith("table-aware:"):
        raise ValueError("FINAUDIT_COLLECTION_ID must refer to table-aware evidence")
    store = PgChunkStore(os.environ["FINAUDIT_DATABASE_URL"])
    retriever = HybridRetriever(store, collection, documents)
    models = CpuModels(Path(os.environ.get("FINAUDIT_MODEL_CACHE", "data/local/models")))
    planner = ChatPlanner(
        os.environ["FINAUDIT_LLM_ENDPOINT"],
        os.environ["FINAUDIT_LLM_MODEL"],
        os.environ["FINAUDIT_LLM_API_KEY"],
    )
    connection = None
    cache = None
    if redis_url := os.environ.get("FINAUDIT_REDIS_URL"):
        connection = Redis.from_url(redis_url, socket_connect_timeout=0.4, socket_timeout=0.4)
        manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        namespace = ":".join(
            (
                collection,
                manifest_digest,
                RERANKER_REVISION,
                RETRIEVAL_VERSION,
                PROMPT_VERSION,
                os.environ["FINAUDIT_LLM_ENDPOINT"],
                os.environ["FINAUDIT_LLM_MODEL"],
            )
        )
        cache = RedisAnswerCache(connection, namespace)

    def retrieve(question: str) -> list[Chunk]:
        return retriever.search(question, models.embed_query(question), models.rerank, limit=12)

    def close() -> None:
        planner.close()
        if connection is not None:
            connection.close()

    return AnswerService(documents, retrieve, planner, cache, lookup=retriever.lookup), close


def create_app(service: AnswerService | None = None) -> FastAPI:
    """Assemble the API with its configured providers or a supplied test service."""
    if service is None:
        service, close = _runtime()
    else:
        close = None
    answer = service.answer
    capacity = BoundedSemaphore(1)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if close is not None:
                close()

    app = FastAPI(title="FinAudit-RAG", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid_request", "Send a question containing 1–1000 characters.")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse | JSONResponse:
        if not capacity.acquire(blocking=False):
            return _error(503, "busy", "The service is processing another request.")
        try:
            return answer(payload.question)
        except PlannerUnavailable:
            return _error(
                503, "model_unavailable", "The model provider is temporarily unavailable."
            )
        except psycopg.Error:
            return _error(
                503, "storage_unavailable", "The document index is temporarily unavailable."
            )
        finally:
            capacity.release()

    return app
