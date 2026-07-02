from __future__ import annotations

from collections.abc import Sequence

from dotenv import load_dotenv
from pydantic_ai.messages import BinaryImage, ImageUrl, TextContent, UserContent

from .router import AgentRouter
from .session import Session
from .session_store import FirestoreSessionStore

DEFAULT_IMAGE_MEDIA_TYPE = "image/jpeg"
IMAGE_PROMPT_TEXT = "The user sent this image on WhatsApp. Analyze it and provide a helpful response."
IMAGE_ROUTING_TEXT = "A WhatsApp user sent an image and needs help interpreting or responding to it."


def chat(
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
    image_url: str | None = None,
    image_media_type: str = DEFAULT_IMAGE_MEDIA_TYPE,
    session_id: str | None = None,
    channel: str = "whatsapp",
) -> str:
    """Route one WhatsApp text or image message through the agent loop and return the response."""

    load_dotenv()
    prompt = _build_prompt(
        text=text,
        image_bytes=image_bytes,
        image_url=image_url,
        image_media_type=image_media_type,
    )
    route_query = text.strip() if text is not None else IMAGE_ROUTING_TEXT

    router = AgentRouter()
    agent, _metadata = router.route_with_metadata(route_query)
    store = FirestoreSessionStore() if session_id is not None else None
    history = store.load_history(session_id) if store is not None else None
    session = Session(agent, history=history)

    response = "".join(session.send_stream(prompt))
    if store is not None:
        store.save_history(
            session_id,
            session.history,
            agent_name=getattr(agent, "name", None),
            channel=channel,
        )
    return response


def _build_prompt(
    *,
    text: str | None,
    image_bytes: bytes | None,
    image_url: str | None,
    image_media_type: str,
) -> str | Sequence[UserContent]:
    supplied = sum(value is not None for value in (text, image_bytes, image_url))
    if supplied != 1:
        raise ValueError("Provide exactly one of text, image_bytes, or image_url")

    if text is not None:
        if not text.strip():
            raise ValueError("text must not be blank")
        return text.strip()

    if image_bytes is not None:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        return [
            TextContent(content=IMAGE_PROMPT_TEXT),
            BinaryImage(data=image_bytes, media_type=image_media_type),
        ]

    if image_url is None or not image_url.strip():
        raise ValueError("image_url must not be blank")

    return [
        TextContent(content=IMAGE_PROMPT_TEXT),
        ImageUrl(url=image_url.strip(), media_type=image_media_type),
    ]
