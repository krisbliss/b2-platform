"""FastAPI entry point for the b2 WhatsApp Channel Adapter.

Exposes POST /message — called by the central benevolent-bandwidth webhook
for every inbound WhatsApp message routed to b2-platform.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .chat import chat
from .whatsapp_media import download_media

logger = logging.getLogger(__name__)

app = FastAPI(title="b2 WhatsApp Adapter")

_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# WhatsApp media message types that carry a downloadable media ID.
_MEDIA_TYPES = ("image", "document")


@dataclass
class InboundMessage:
    """A normalized inbound WhatsApp message."""

    wa_id: str | None = None
    text: str | None = None
    media_id: str | None = None
    mime_type: str | None = None


def _extract_message(payload: dict[str, Any]) -> InboundMessage:
    """Normalize a WhatsApp webhook payload into an InboundMessage.

    Handles text and media (image/document) messages; returns an empty
    InboundMessage for anything else (e.g. status updates).
    """
    try:
        entry = (payload.get("entry") or [])[0]
        value = (entry.get("changes") or [])[0].get("value", {})
        contacts = value.get("contacts") or []
        messages = value.get("messages") or []

        if not messages:
            return InboundMessage()

        msg = messages[0]
        msg_type = msg.get("type")
        wa_id = (contacts[0].get("wa_id") if contacts else None) or msg.get("from")

        if msg_type == "text":
            text = (msg.get("text") or {}).get("body", "").strip()
            if not text:
                return InboundMessage(wa_id=wa_id)
            return InboundMessage(wa_id=wa_id, text=text)

        if msg_type in _MEDIA_TYPES:
            media = msg.get(msg_type) or {}
            media_id = media.get("id")
            if not media_id:
                return InboundMessage(wa_id=wa_id)
            return InboundMessage(wa_id=wa_id, media_id=media_id, mime_type=media.get("mime_type"))

        return InboundMessage(wa_id=wa_id)
    except (IndexError, AttributeError, TypeError, KeyError):
        return InboundMessage()


@app.post("/message")
async def message_endpoint(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, str | None]:
    if _WEBHOOK_SECRET and x_webhook_secret != _WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    logger.info("api.message received object=%s", payload.get("object"))

    message = _extract_message(payload)

    if message.text is not None:
        logger.info("api.message routing text wa_id=%s chars=%d", message.wa_id, len(message.text))
        response = await run_in_threadpool(chat, text=message.text, session_id=message.wa_id)
        logger.info("api.message done wa_id=%s response_chars=%d", message.wa_id, len(response))
        return {"response": response}

    if message.media_id is not None:
        logger.info("api.message media wa_id=%s media=%s", message.wa_id, message.media_id)
        media = await download_media(message.media_id)
        if media is None:
            logger.info("api.message skipped — media download unavailable")
            return {"response": None}
        image_bytes, mime_type = media
        response = await run_in_threadpool(
            chat,
            image_bytes=image_bytes,
            image_media_type=mime_type,
            session_id=message.wa_id,
        )
        logger.info("api.message done wa_id=%s response_chars=%d", message.wa_id, len(response))
        return {"response": response}

    logger.info("api.message skipped — no actionable payload")
    return {"response": None}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
