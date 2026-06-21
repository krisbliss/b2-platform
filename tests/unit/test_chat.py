from pydantic_ai.messages import BinaryImage, ImageUrl, TextContent

from src import chat as chat_module
from src.chat import IMAGE_PROMPT_TEXT, _build_prompt, chat


def test_build_prompt_accepts_text_only() -> None:
    assert _build_prompt(
        text="  hello  ",
        image_bytes=None,
        image_url=None,
        image_media_type="image/jpeg",
    ) == "hello"


def test_build_prompt_accepts_image_bytes_only() -> None:
    prompt = _build_prompt(
        text=None,
        image_bytes=b"image",
        image_url=None,
        image_media_type="image/png",
    )

    assert isinstance(prompt, list)
    assert prompt == [
        TextContent(content=IMAGE_PROMPT_TEXT),
        BinaryImage(data=b"image", media_type="image/png"),
    ]


def test_build_prompt_accepts_image_url_only() -> None:
    prompt = _build_prompt(
        text=None,
        image_bytes=None,
        image_url=" https://example.test/image.jpg ",
        image_media_type="image/jpeg",
    )

    assert isinstance(prompt, list)
    assert prompt == [
        TextContent(content=IMAGE_PROMPT_TEXT),
        ImageUrl(url="https://example.test/image.jpg", media_type="image/jpeg"),
    ]


def test_build_prompt_rejects_multiple_inputs() -> None:
    try:
        _build_prompt(
            text="hello",
            image_bytes=b"image",
            image_url=None,
            image_media_type="image/jpeg",
        )
    except ValueError as exc:
        assert str(exc) == "Provide exactly one of text, image_bytes, or image_url"
    else:
        raise AssertionError("expected ValueError")


def test_chat_routes_and_returns_streamed_text(monkeypatch) -> None:
    class FakeRouter:
        def route_with_metadata(self, query: str):
            assert query == "hello"
            return object(), {"score": 1.0}

    class FakeSession:
        def __init__(self, agent: object):
            self.agent = agent

        def send_stream(self, prompt: str):
            assert prompt == "hello"
            yield "hi"
            yield " there"

    monkeypatch.setattr(chat_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(chat_module, "AgentRouter", FakeRouter)
    monkeypatch.setattr(chat_module, "Session", FakeSession)

    assert chat(text="hello") == "hi there"
