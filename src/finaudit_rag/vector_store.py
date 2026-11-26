"""PostgreSQL storage and exact cosine search for versioned chunk collections."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from finaudit_rag.ingestion import Chunk, chunk_from_record

EMBEDDING_DIMENSIONS = 384
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS finaudit_chunks (
    collection_id text NOT NULL,
    id text NOT NULL,
    document_id text NOT NULL,
    page integer NOT NULL CHECK (page > 0),
    kind text NOT NULL CHECK (kind IN ('text', 'table')),
    text text NOT NULL CHECK (length(text) > 0),
    table_data jsonb,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (collection_id, id),
    CHECK ((kind = 'table') = (table_data IS NOT NULL))
);
"""
FIELDS = "id, document_id, page, kind, text, table_data AS table"


def _vector_literal(vector: Sequence[float]) -> str:
    if (
        len(vector) != EMBEDDING_DIMENSIONS
        or not all(math.isfinite(element) for element in vector)
        or not any(vector)
    ):
        raise ValueError(f"expected a nonzero finite vector of length {EMBEDDING_DIMENSIONS}")
    return json.dumps(list(vector), separators=(",", ":"), allow_nan=False)


class PgChunkStore:
    """Persist and retrieve independent snapshots of the same PDF corpus."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def initialize(self) -> None:
        """Install pgvector and the chunk table in the configured database."""
        with psycopg.connect(self._database_url, connect_timeout=5) as connection:
            connection.execute(SCHEMA)

    def replace_collection(
        self, collection_id: str, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Replace a collection in one transaction so readers see complete snapshots."""
        if not collection_id or not chunks or len(chunks) != len(embeddings):
            raise ValueError("expected a collection ID and one vector per chunk")
        records = [
            (
                collection_id,
                chunk.id,
                chunk.document_id,
                chunk.page,
                chunk.kind,
                chunk.text,
                Jsonb(asdict(chunk.table)) if chunk.table is not None else None,
                _vector_literal(vector),
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        with psycopg.connect(self._database_url, connect_timeout=5) as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (collection_id,)
            )
            connection.execute(
                "DELETE FROM finaudit_chunks WHERE collection_id = %s", (collection_id,)
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO finaudit_chunks
                        (collection_id, id, document_id, page, kind, text, table_data, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    records,
                )

    def list_chunks(self, collection_id: str) -> list[Chunk]:
        """Return the content of a collection in stable ID order."""
        with psycopg.connect(
            self._database_url, row_factory=dict_row, connect_timeout=5
        ) as connection:
            rows = connection.execute(
                f"SELECT {FIELDS} FROM finaudit_chunks WHERE collection_id = %s ORDER BY id",
                (collection_id,),
            ).fetchall()
        return [chunk_from_record(row) for row in rows]

    def cosine_search(
        self, collection_id: str, query_vector: Sequence[float], limit: int
    ) -> list[Chunk]:
        """Return the closest chunks under pgvector cosine distance."""
        with psycopg.connect(
            self._database_url, row_factory=dict_row, connect_timeout=5
        ) as connection:
            rows = connection.execute(
                f"""
                SELECT {FIELDS} FROM finaudit_chunks WHERE collection_id = %s
                ORDER BY embedding <=> %s::vector, id LIMIT %s
                """,
                (collection_id, _vector_literal(query_vector), limit),
            ).fetchall()
        return [chunk_from_record(row) for row in rows]
