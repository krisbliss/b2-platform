import json

import httpx
import pytest

from e2e import runner
from e2e.cases import Case


@pytest.mark.parametrize("mime_type, message_type", [
    ("application/pdf", "document"),
    ("image/png", "image"),
    ("image/jpeg", "image"),
])
@pytest.mark.parametrize("has_media_id", [True, False])
def test_remote_case_posts_webhooks_without_registering_local_media(
    monkeypatch, tmp_path, mime_type, message_type, has_media_id,
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "response": "Verification passed",
            "debug": {"tool_results": [{
                "tool": "death_certificate_verification", "accepted": True,
            }]},
        })

    def reject_local_media(*args):
        pytest.fail("Remote cases must not register local media")

    monkeypatch.setattr(runner, "register_local_media", reject_local_media)
    monkeypatch.setattr(runner, "summarize_case_result", lambda result: "Test summary")
    expected = {"expected_outcome": "accept", "mime_type": mime_type}
    if has_media_id:
        expected["media_id"] = "meta-media-123"
    case = Case(
        country="Indonesia", kind="real", case_id="case_001", path=tmp_path,
        narrative="Please verify this certificate.",
        certificate_path=tmp_path / "certificate.pdf", expected=expected,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = runner.run_case(
            client, case, base_url="https://adapter.example", secret="test-secret",
            local_mode=False, debug_tools=True,
        )

    assert len(requests) == (2 if has_media_id else 1)
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == "https://adapter.example/message"
        assert request.headers["X-Webhook-Secret"] == "test-secret"
        assert request.headers["X-E2E-Debug"] == "true"
        assert json.loads(request.content)["object"] == "whatsapp_business_account"
    if has_media_id:
        value = json.loads(requests[-1].content)["entry"][0]["changes"][0]["value"]
        message = value["messages"][0]
        assert message["type"] == message_type
        assert message[message_type] == {"id": "meta-media-123", "mime_type": mime_type}
        assert result["passed"] is True
    else:
        assert result["passed"] is False
        assert result["classification"] == "UNKNOWN"
        assert "media turn skipped" in result["skipped_turns"][0]["reason"]
