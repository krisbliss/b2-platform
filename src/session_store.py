from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

DEFAULT_SESSION_TTL = timedelta(hours=72)


class FirestoreSessionStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        project: str | None = None,
        collection: str = "sessions",
        server_timestamp: Any | None = None,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        now: Callable[[], datetime] | None = None,
    ):
        firestore = None if client is not None else _firestore_module()
        self._client = client or firestore.Client(project=project or os.getenv("GOOGLE_CLOUD_PROJECT"))
        self._collection = collection
        self._server_timestamp = (
            server_timestamp
            if server_timestamp is not None
            else (firestore or _firestore_module()).SERVER_TIMESTAMP
        )
        self._ttl = ttl
        self._now = now or _utc_now

    def load_history(self, session_id: str) -> list[ModelMessage]:
        document = self._document(session_id)
        snapshot = document.get()
        if not snapshot.exists:
            return []

        data = snapshot.to_dict() or {}
        expires_at = data.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= self._now():
            document.delete()
            return []

        history = data.get("history") or []
        return list(ModelMessagesTypeAdapter.validate_python(history))

    def save_history(
        self,
        session_id: str,
        history: Sequence[ModelMessage],
        *,
        agent_name: str | None = None,
        channel: str | None = None,
    ) -> None:
        document = self._document(session_id)
        snapshot = document.get()
        data: dict[str, Any] = {
            "session_id": session_id,
            "history": ModelMessagesTypeAdapter.dump_python(history, mode="json"),
            "updated_at": self._server_timestamp,
            "expires_at": self._now() + self._ttl,
        }

        if not snapshot.exists:
            data["created_at"] = self._server_timestamp
        if agent_name is not None:
            data["agent_name"] = agent_name
        if channel is not None:
            data["channel"] = channel

        document.set(data, merge=True)

    def _document(self, session_id: str) -> Any:
        return self._client.collection(self._collection).document(session_id)


def _firestore_module() -> Any:
    from google.cloud import firestore

    return firestore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
