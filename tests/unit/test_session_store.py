from datetime import datetime, timedelta, timezone

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, UserPromptPart

from src.session_store import FirestoreSessionStore


class FakeSnapshot:
    def __init__(self, exists: bool, data=None):
        self.exists = exists
        self._data = data

    def to_dict(self):
        return self._data


class FakeDocument:
    def __init__(self, snapshot: FakeSnapshot):
        self.snapshot = snapshot
        self.writes = []
        self.deletes = 0

    def get(self):
        return self.snapshot

    def set(self, data, *, merge: bool):
        self.writes.append((data, merge))
        self.snapshot = FakeSnapshot(True, data)

    def delete(self):
        self.deletes += 1
        self.snapshot = FakeSnapshot(False)


class FakeCollection:
    def __init__(self, document: FakeDocument):
        self.document_ref = document

    def document(self, session_id: str):
        assert session_id == "session-1"
        return self.document_ref


class FakeClient:
    def __init__(self, document: FakeDocument):
        self.document_ref = document
        self.collection_name = None

    def collection(self, name: str):
        self.collection_name = name
        return FakeCollection(self.document_ref)


def test_load_history_deserializes_stored_messages() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]
    store = FirestoreSessionStore(
        client=FakeClient(
            FakeDocument(
                FakeSnapshot(
                    True,
                    {
                        "history": ModelMessagesTypeAdapter.dump_python(history, mode="json"),
                        "expires_at": now + timedelta(hours=1),
                    },
                )
            )
        ),
        server_timestamp="SERVER_TIME",
        now=lambda: now,
    )

    assert store.load_history("session-1") == history


def test_load_history_returns_empty_list_and_deletes_expired_document() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document = FakeDocument(
        FakeSnapshot(
            True,
            {
                "history": ModelMessagesTypeAdapter.dump_python(
                    [ModelRequest(parts=[UserPromptPart(content="hello")])],
                    mode="json",
                ),
                "expires_at": now,
            },
        )
    )
    store = FirestoreSessionStore(
        client=FakeClient(document),
        server_timestamp="SERVER_TIME",
        now=lambda: now,
    )

    assert store.load_history("session-1") == []
    assert document.deletes == 1


def test_load_history_returns_empty_list_for_missing_document() -> None:
    store = FirestoreSessionStore(
        client=FakeClient(FakeDocument(FakeSnapshot(False))),
        server_timestamp="SERVER_TIME",
    )

    assert store.load_history("session-1") == []


def test_save_history_serializes_messages_and_sets_metadata_on_create() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    document = FakeDocument(FakeSnapshot(False))
    client = FakeClient(document)
    store = FirestoreSessionStore(client=client, server_timestamp="SERVER_TIME", now=lambda: now)
    history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]

    store.save_history("session-1", history, agent_name="support", channel="whatsapp")

    assert client.collection_name == "sessions"
    assert len(document.writes) == 1
    data, merge = document.writes[0]
    assert merge is True
    assert data["session_id"] == "session-1"
    assert data["updated_at"] == "SERVER_TIME"
    assert data["created_at"] == "SERVER_TIME"
    assert data["expires_at"] == now + timedelta(hours=72)
    assert data["agent_name"] == "support"
    assert data["channel"] == "whatsapp"
    assert data["history"][0]["kind"] == "request"
    assert data["history"][1]["kind"] == "response"
