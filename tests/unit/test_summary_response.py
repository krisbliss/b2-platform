from __future__ import annotations

from src import summary_response


def test_summary_payload_uses_verification_summary_without_tool_framing() -> None:
    payload = summary_response.summary_payload(
        "The request was accepted.",
        [
            {
                "tool": "death_certificate_verification",
                "accepted": True,
                "handed_off": True,
                "summary": "Verification passed.",
                "flags": [],
                "authenticity": {
                    "verdict": "PASS",
                    "risk_score": 0.0,
                    "checks": [],
                },
            }
        ],
    )

    assert payload == {
        "assistant_response": "The request was accepted.",
        "death_certificate_verification": {
            "status": None,
            "accepted": True,
            "handed_off": True,
            "summary": "Verification passed.",
            "flags": [],
        },
    }


def test_summary_prompt_excludes_internal_e2e_language() -> None:
    prompt = summary_response.summary_prompt(
        "The request was accepted.",
        [
            {
                "tool": "death_certificate_verification",
                "accepted": True,
                "summary": "Verification passed.",
                "flags": [],
            }
        ],
    )

    assert "WhatsApp-ready" in prompt
    assert "Do not mention internal tools" in prompt
    assert "e2e test case result" not in prompt
    assert "debug output" in prompt


def test_generate_summary_tool_response_returns_none_on_error(monkeypatch) -> None:
    async def fake_call(messages):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(summary_response, "call_summary_tool", fake_call)

    result = summary_response.generate_summary_tool_response(
        final_response="The request was accepted.",
        tool_events=[],
    )

    import asyncio

    assert asyncio.run(result) is None
