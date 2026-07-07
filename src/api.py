"""FastAPI entry point for the b2 WhatsApp Channel Adapter.

Exposes POST /message — called by the central benevolent-bandwidth webhook
for every inbound WhatsApp message routed to b2-platform.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .chat import chat

logger = logging.getLogger(__name__)

app = FastAPI(title="b2 WhatsApp Adapter")

_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


def _extract_message(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (wa_id, text) for a text message, or (None, None) for anything else."""
    try:
        entry = (payload.get("entry") or [])[0]
        value = (entry.get("changes") or [])[0].get("value", {})
        contacts = value.get("contacts") or []
        messages = value.get("messages") or []

        if not messages:
            return None, None

        msg = messages[0]
        if msg.get("type") != "text":
            return None, None

        wa_id = (contacts[0].get("wa_id") if contacts else None) or msg.get("from")
        text = (msg.get("text") or {}).get("body", "").strip()
        if not text:
            return None, None

        return wa_id, text
    except (IndexError, AttributeError, TypeError, KeyError):
        return None, None


@app.post("/message")
async def message_endpoint(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, str | None]:
    if _WEBHOOK_SECRET and x_webhook_secret != _WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    logger.info("api.message received object=%s", payload.get("object"))

    wa_id, text = _extract_message(payload)
    if text is None:
        logger.info("api.message skipped — no text payload")
        return {"response": None}

    logger.info("api.message routing wa_id=%s chars=%d", wa_id, len(text))
    response = await run_in_threadpool(chat, text=text, session_id=wa_id)
    logger.info("api.message done wa_id=%s response_chars=%d", wa_id, len(response))
    return {"response": response}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
