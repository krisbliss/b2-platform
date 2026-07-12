from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL = timedelta(hours=72)

# Firestore caps a document at ~1 MB; base64 inflates bytes by ~33%. Skip anything
# whose encoded form would risk the limit and log it rather than fail the write.
MAX_MEDIA_BYTES = 700_000


class FirestoreSessionStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        project: str | None = None,
        collection: str = "sessions",
        media_collection: str = "session_media",
        server_timestamp: Any | None = None,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        now: Callable[[], datetime] | None = None,
    ):
        firestore = None if client is not None else _firestore_module()
        self._client = client or firestore.Client(project=project or os.getenv("GOOGLE_CLOUD_PROJECT"))
        self._collection = collection
        self._media_collection = media_collection
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

    def save_media(self, session_id: str, image_bytes: bytes, *, mime_type: str) -> bool:
        """Persist the most recent inbound image for a session.

        Stores one document per session (overwritten on each upload) so tools can
        later pull the latest media by session_id without the image ever passing
        through the model. Returns False (and skips the write) if the image is too
        large for a Firestore document.
        """
        if not image_bytes:
            return False
        if len(image_bytes) > MAX_MEDIA_BYTES:
            logger.warning(
                "session_store.save_media skipped — image too large bytes=%d limit=%d session=%.8s",
                len(image_bytes),
                MAX_MEDIA_BYTES,
                session_id,
            )
            return False

        document = self._media_document(session_id)
        document.set(
            {
                "session_id": session_id,
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                "mime_type": mime_type,
                "created_at": self._server_timestamp,
                "expires_at": self._now() + self._ttl,
            }
        )
        return True

    def load_latest_media(self, session_id: str) -> tuple[bytes, str] | None:
        """Return (image_bytes, mime_type) for the most recent image, or None."""
        document = self._media_document(session_id)
        snapshot = document.get()
        if not snapshot.exists:
            return None

        data = snapshot.to_dict() or {}
        expires_at = data.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= self._now():
            document.delete()
            return None

        encoded = data.get("image_b64")
        if not encoded:
            return None

        return base64.b64decode(encoded), data.get("mime_type") or "image/jpeg"

    def _document(self, session_id: str) -> Any:
        return self._client.collection(self._collection).document(session_id)

    def _media_document(self, session_id: str) -> Any:
        return self._client.collection(self._media_collection).document(session_id)


def _firestore_module() -> Any:
    from google.cloud import firestore

    return firestore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
