"""Pinned CPU embeddings and reranking, with model artifacts cached locally."""

from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

EMBEDDING_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
RERANKER_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"


def _snapshot(repo: str, revision: str, model_file: str, cache: Path) -> str:
    paths = ["*.json", "*.txt", model_file]
    try:
        location = snapshot_download(
            repo,
            revision=revision,
            allow_patterns=paths,
            cache_dir=cache,
            local_files_only=True,
            token=False,
        )
    except LocalEntryNotFoundError:
        location = ""
    if location and all(
        (Path(location) / required).is_file()
        for required in (model_file, "config.json", "tokenizer.json")
    ):
        return location
    return snapshot_download(
        repo, revision=revision, allow_patterns=paths, cache_dir=cache, token=False, max_workers=3
    )


class CpuModels:
    """Run model inference without sending queries or evidence to a remote API."""

    def __init__(self, cache: Path) -> None:
        self._cache = cache
        location = _snapshot(
            "qdrant/bge-small-en-v1.5-onnx-q", EMBEDDING_REVISION, "model_optimized.onnx", cache
        )
        self._embedding = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(cache),
            specific_model_path=location,
            threads=2,
        )
        self._reranker: TextCrossEncoder | None = None

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Embed document text in small CPU batches."""
        return [
            [float(element) for element in embedding]
            for embedding in self._embedding.passage_embed(documents, batch_size=16)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using the model's query encoding."""
        return [float(element) for element in next(iter(self._embedding.query_embed(query)))]

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score paired query and document inputs with the cross-encoder."""
        if self._reranker is None:
            location = _snapshot(
                "Xenova/ms-marco-MiniLM-L-6-v2",
                RERANKER_REVISION,
                "onnx/model.onnx",
                self._cache,
            )
            self._reranker = TextCrossEncoder(
                model_name="Xenova/ms-marco-MiniLM-L-6-v2",
                cache_dir=str(self._cache),
                specific_model_path=location,
                threads=2,
            )
        return [float(score) for score in self._reranker.rerank(query, passages, batch_size=8)]
