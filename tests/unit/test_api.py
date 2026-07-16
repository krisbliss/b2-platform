"""Unit tests for src/api.py — WhatsApp Channel Adapter endpoint."""

import importlib
import os

import pytest
from fastapi.testclient import TestClient


SAMPLE_TEXT_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "16179164660",
                            "phone_number_id": "1104821716055506",
                        },
                        "contacts": [
                            {"profile": {"name": "Test User"}, "wa_id": "16508106640"}
                        ],
                        "messages": [
                            {
                                "from": "16508106640",
                                "id": "wamid.test",
                                "timestamp": "1780358445",
                                "text": {"body": "Hello B2, what is Givelight?"},
                                "type": "text",
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}

STATUS_UPDATE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "statuses": [
                            {
                                "id": "wamid.test",
                                "status": "delivered",
                                "timestamp": "1780358446",
                                "recipient_id": "16508106640",
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}

IMAGE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "contacts": [
                            {"profile": {"name": "Test User"}, "wa_id": "16508106640"}
                        ],
                        "messages": [
                            {
                                "from": "16508106640",
                                "id": "wamid.test2",
                                "timestamp": "1780358447",
                                "type": "image",
                                "image": {"id": "media-123", "mime_type": "image/jpeg"},
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}


def _make_client(secret: str = "", monkeypatch=None):
    """Import api module fresh with WEBHOOK_SECRET env var set."""
    if monkeypatch is not None:
        monkeypatch.setenv("WEBHOOK_SECRET", secret)

    # Re-import to pick up env var at module level
    import src.api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app), api_module


# ---------------------------------------------------------------------------
# _extract_message unit tests
# ---------------------------------------------------------------------------

def test_extract_message_text():
    from src.api import _extract_message
    msg = _extract_message(SAMPLE_TEXT_PAYLOAD)
    assert msg.wa_id == "16508106640"
    assert msg.text == "Hello B2, what is Givelight?"
    assert msg.media_id is None


def test_extract_message_status_update_returns_empty():
    from src.api import _extract_message
    msg = _extract_message(STATUS_UPDATE_PAYLOAD)
    assert msg.text is None
    assert msg.media_id is None


def test_extract_message_image_returns_media_id():
    from src.api import _extract_message
    msg = _extract_message(IMAGE_PAYLOAD)
    assert msg.wa_id == "16508106640"
    assert msg.text is None
    assert msg.media_id == "media-123"
    assert msg.mime_type == "image/jpeg"


def test_extract_message_empty_payload():
    from src.api import _extract_message
    msg = _extract_message({})
    assert msg.wa_id is None
    assert msg.text is None
    assert msg.media_id is None


def test_extract_message_blank_text():
    from src.api import _extract_message
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{"type": "text", "text": {"body": "  "}, "from": "123"}]}}]}]
    }
    msg = _extract_message(payload)
    assert msg.text is None
    assert msg.media_id is None


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

def test_health():
    from src.api import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /message — no secret configured (open mode)
# ---------------------------------------------------------------------------

def test_message_status_update_returns_null(monkeypatch):
    import src.api as api_module
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")

    def fake_chat(**kwargs):
        return "should not be called"

    monkeypatch.setattr(api_module, "chat", fake_chat)

    client = TestClient(api_module.app)
    r = client.post("/message", json=STATUS_UPDATE_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"response": None}


def test_message_image_download_unavailable_returns_null(monkeypatch):
    """If media can't be downloaded (e.g. no WHATSAPP_TOKEN), respond with null."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "chat", lambda **kw: "should not be called")

    async def fake_download(media_id):
        return None

    monkeypatch.setattr(api_module, "download_media", fake_download)

    client = TestClient(api_module.app)
    r = client.post("/message", json=IMAGE_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"response": None}


def test_message_image_downloads_and_calls_chat(monkeypatch):
    """A downloadable image is fetched and routed to chat as image_bytes."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")

    captured = {}

    async def fake_download(media_id):
        captured["media_id"] = media_id
        return b"\xff\xd8jpeg-bytes", "image/jpeg"

    def fake_chat(*, image_bytes=None, image_media_type=None, session_id=None, **kwargs):
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        captured["session_id"] = session_id
        return "Your request is being processed."

    monkeypatch.setattr(api_module, "download_media", fake_download)
    monkeypatch.setattr(api_module, "chat", fake_chat)

    client = TestClient(api_module.app)
    r = client.post("/message", json=IMAGE_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"response": "Your request is being processed."}
    assert captured["media_id"] == "media-123"
    assert captured["image_bytes"] == b"\xff\xd8jpeg-bytes"
    assert captured["image_media_type"] == "image/jpeg"
    assert captured["session_id"] == "16508106640"


def test_message_text_calls_chat(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")

    captured = {}

    def fake_chat(*, text, session_id, **kwargs):
        captured["text"] = text
        captured["session_id"] = session_id
        return "Givelight is an orphan aid programme."

    monkeypatch.setattr(api_module, "chat", fake_chat)

    client = TestClient(api_module.app)
    r = client.post("/message", json=SAMPLE_TEXT_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"response": "Givelight is an orphan aid programme."}
    assert captured["text"] == "Hello B2, what is Givelight?"
    assert captured["session_id"] == "16508106640"


# ---------------------------------------------------------------------------
# /message — secret header enforcement
# ---------------------------------------------------------------------------

def test_message_missing_secret_returns_401(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")

    client = TestClient(api_module.app, raise_server_exceptions=False)
    r = client.post("/message", json=SAMPLE_TEXT_PAYLOAD)
    assert r.status_code == 401


def test_message_wrong_secret_returns_401(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")

    client = TestClient(api_module.app, raise_server_exceptions=False)
    r = client.post(
        "/message",
        json=SAMPLE_TEXT_PAYLOAD,
        headers={"X-Webhook-Secret": "wrongsecret"},
    )
    assert r.status_code == 401


def test_message_correct_secret_passes(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")
    monkeypatch.setattr(api_module, "chat", lambda *, text, **kw: "ok")

    client = TestClient(api_module.app)
    r = client.post(
        "/message",
        json=SAMPLE_TEXT_PAYLOAD,
        headers={"X-Webhook-Secret": "mysecret"},
    )
    assert r.status_code == 200
    assert r.json()["response"] == "ok"
