"""Build searchable corpus snapshots and query baseline or hybrid retrieval."""

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from finaudit_rag.ingestion import Chunk, chunk_from_record
from finaudit_rag.models import EMBEDDING_REVISION, CpuModels
from finaudit_rag.vector_store import PgChunkStore

WORDS = re.compile(r"\w+(?:[./-]\w+)*")
CIRCULAR_ID = re.compile(r"(?<!\w)[a-z]{2,}(?:/[a-z\d.-]+)+/\d+(?!\w)", re.IGNORECASE)
MAX_CANDIDATES = 20


class ChunkStore(Protocol):
    def list_chunks(self, collection_id: str) -> list[Chunk]: ...

    def cosine_search(
        self, collection_id: str, query_vector: Sequence[float], limit: int
    ) -> list[Chunk]: ...


def tokens(text: str) -> list[str]:
    """Keep compound document identifiers and their plain component terms."""
    result: list[str] = []
    for match in WORDS.findall(text.casefold()):
        result.append(match)
        if not match.isalnum():
            result.extend(part for part in re.split(r"[./-]+", match) if part)
    return result


def search_text(chunk: Chunk) -> str:
    """Keep the issuer and release identifiable when a chunk omits the cover page."""
    return f"{chunk.document_id.replace('-', ' ')}\n{chunk.text}"


class HybridRetriever:
    """Combine BM25 and cosine ranks, then score the evidence with a cross-encoder."""

    def __init__(self, store: ChunkStore, collection_id: str) -> None:
        self._store = store
        self._collection_id = collection_id
        self._chunks = store.list_chunks(collection_id)
        if not self._chunks:
            raise ValueError(f"unknown or empty chunk collection {collection_id}")
        self._lexical = BM25Okapi([tokens(search_text(chunk)) for chunk in self._chunks])

    def search(
        self,
        query: str,
        vector: Sequence[float],
        rerank: Callable[[str, Sequence[str]], list[float]],
        limit: int = 5,
    ) -> list[Chunk]:
        """Return reranked evidence while protecting exact primary circular IDs."""
        query_ids = {match.casefold() for match in CIRCULAR_ID.findall(query)}
        exact = [
            chunk
            for chunk in self._chunks
            if query_ids
            and chunk.page == 1
            and any(
                identifier.casefold() in query_ids
                for line in chunk.text.splitlines()[:5]
                for identifier in CIRCULAR_ID.findall(line)
                if line.lstrip().casefold().startswith(identifier.casefold())
            )
        ][:MAX_CANDIDATES]
        lexical_scores = self._lexical.get_scores(tokens(query))
        lexical = sorted(
            range(len(self._chunks)),
            key=lambda index: (-lexical_scores[index], self._chunks[index].id),
        )
        sparse = [
            self._chunks[index] for index in lexical[:MAX_CANDIDATES] if lexical_scores[index] > 0
        ]
        dense = self._store.cosine_search(self._collection_id, vector, MAX_CANDIDATES)
        scores: dict[str, float] = defaultdict(float)
        candidates: dict[str, Chunk] = {}
        for ranked in (sparse, dense):
            for rank, chunk in enumerate(ranked, 1):
                candidates[chunk.id] = chunk
                scores[chunk.id] += 1 / (60 + rank)

        fused = sorted(candidates.values(), key=lambda chunk: (-scores[chunk.id], chunk.id))
        pool = list({chunk.id: chunk for chunk in (*exact, *fused)}.values())[:MAX_CANDIDATES]
        if not pool:
            return []
        reranked = rerank(query, [search_text(chunk) for chunk in pool])
        if len(reranked) != len(pool) or not all(math.isfinite(score) for score in reranked):
            raise ValueError("reranker returned invalid scores")
        exact_ids = {chunk.id for chunk in exact}
        ranked_pool = sorted(
            zip(pool, reranked, strict=True),
            key=lambda result: (result[0].id not in exact_ids, -result[1], -scores[result[0].id]),
        )
        return [chunk for chunk, _ in ranked_pool[:limit]]


def load_export(path: Path) -> tuple[str, list[Chunk]]:
    """Load a trusted, versioned ingestion export."""
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        chunks = [chunk_from_record(json.loads(line)) for line in source]
    return digest, chunks


def index_export(path: Path, mode: str, store: PgChunkStore, models: CpuModels) -> str:
    """Use an export digest and model revision to address an immutable collection."""
    digest, chunks = load_export(path)
    collection_id = f"{mode}:{digest}:{EMBEDDING_REVISION}"
    if mode == "baseline" and any(chunk.kind == "table" for chunk in chunks):
        raise ValueError("baseline collections require a baseline export")
    store.initialize()
    store.replace_collection(
        collection_id, chunks, models.embed_documents([search_text(chunk) for chunk in chunks])
    )
    return collection_id


def main() -> None:
    """Index extracted chunks or print retrieved evidence as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, default=Path("data/local/models"))
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("export", type=Path)
    index.add_argument("--mode", choices=("baseline", "table-aware"), required=True)
    query = commands.add_parser("search")
    query.add_argument("collection_id")
    query.add_argument("question")
    args = parser.parse_args()
    if not (database_url := os.environ.get("FINAUDIT_DATABASE_URL")):
        parser.error("set FINAUDIT_DATABASE_URL before indexing or searching")
    store = PgChunkStore(database_url)
    models = CpuModels(args.model_cache)
    if args.command == "index":
        collection_id = index_export(args.export, args.mode, store, models)
        print(collection_id)
        return
    if not args.question.strip():
        query.error("question must contain text")
    vector = models.embed_query(args.question)
    if args.collection_id.startswith("baseline:"):
        hits = store.cosine_search(args.collection_id, vector, 5)
    else:
        hits = HybridRetriever(store, args.collection_id).search(
            args.question, vector, models.rerank
        )
    print(json.dumps([asdict(chunk) for chunk in hits], indent=2))


if __name__ == "__main__":
    main()
