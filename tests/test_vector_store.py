import os
from uuid import uuid4

import psycopg
import pytest

from finaudit_rag.ingestion import Chunk, Table
from finaudit_rag.vector_store import EMBEDDING_DIMENSIONS, PgChunkStore


def test_pgvector_exact_search_and_atomic_replacement() -> None:
    if not (database_url := os.environ.get("FINAUDIT_TEST_DATABASE_URL")):
        pytest.skip("requires an isolated PostgreSQL database with pgvector available")
    store = PgChunkStore(database_url)
    store.initialize()
    collection = f"integration-{uuid4().hex}"
    table = Table(("INR crore",), ("Line item", "2025"), (("Revenue", "54,136"),))
    first = Chunk("annual:16:table:0", "annual", 16, "table", "Revenue | 54,136", table)
    other = Chunk("notice:1:text:0", "notice", 1, "text", "Investor charter")
    first_vector = [1.0, 0.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
    other_vector = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)

    for _ in range(2):
        store.replace_collection(collection, [first, other], [first_vector, other_vector])
    assert store.list_chunks(collection) == [first, other]
    assert store.cosine_search(collection, other_vector, 2) == [other, first]

    with pytest.raises(psycopg.errors.UniqueViolation):
        store.replace_collection(collection, [first, first], [first_vector, other_vector])
    assert store.list_chunks(collection) == [first, other]
