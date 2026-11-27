"""Cache exact, non-sensitive queries under corpus and model revisions."""

import hashlib
import logging

from redis import Redis
from redis.exceptions import RedisError

from finaudit_rag.contracts import QueryResponse

LOG = logging.getLogger(__name__)


class RedisAnswerCache:
    def __init__(self, connection: Redis, namespace: str) -> None:
        self._connection = connection
        self._namespace = namespace

    def _key(self, question: str) -> str:
        key = hashlib.sha256(f"{self._namespace}\n{question}".encode()).hexdigest()
        return f"finaudit:answer:{key}"

    def get(self, question: str) -> QueryResponse | None:
        """Return a valid response for the exact question and corpus revision."""
        try:
            raw = self._connection.get(self._key(question))
            if isinstance(raw, str | bytes):
                return QueryResponse.model_validate_json(raw)
        except (RedisError, ValueError):
            LOG.warning("answer cache read unavailable")
        return None

    def put(self, question: str, response: QueryResponse) -> None:
        """Store a verified response for up to five minutes."""
        try:
            self._connection.set(self._key(question), response.model_dump_json(), ex=300)
        except RedisError:
            LOG.warning("answer cache write unavailable")
