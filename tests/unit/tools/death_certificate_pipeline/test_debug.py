from __future__ import annotations

from tools.death_certificate_pipeline.debug import build_verification_debug_event
from tools.fake_image_detector.models import CheckResult, Escalation, ToolResult, Verdict


def test_build_verification_debug_event_includes_bounded_authenticity_details() -> None:
    authenticity = ToolResult(
        verdict=Verdict.FLAG,
        risk_score=0.2,
        escalation=Escalation.HUMAN_REVIEW,
        early_exit=True,
        early_exit_reason="ocr_document runtime error",
        checks=[
            CheckResult(
                check="ocr_document",
                passed=False,
                skipped=False,
                fake_score=0.2,
                confidence=1.0,
                flags=["CHECK_RUNTIME_ERROR"],
                human_escalate=False,
                escalation_reasons=[],
                error="provider unavailable",
                signals={"raw_document_text": "must not leak"},
            )
        ],
    )

    event = build_verification_debug_event(
        {
            "status": "verified",
            "score": 92,
            "band": "escalate",
            "handed_off": False,
            "flags": ["HARD_ESCALATION"],
            "extracted_fields": {"full_name": "Jane Doe"},
            "summary": "Needs review.",
        },
        accepted=False,
        authenticity=authenticity,
    )

    assert event["authenticity"] == {
        "verdict": "FLAG",
        "risk_score": 0.2,
        "escalation": "HUMAN_REVIEW",
        "early_exit": True,
        "early_exit_reason": "ocr_document runtime error",
        "checks": [
            {
                "check": "ocr_document",
                "passed": False,
                "skipped": False,
                "fake_score": 0.2,
                "confidence": 1.0,
                "flags": ["CHECK_RUNTIME_ERROR"],
                "human_escalate": False,
                "escalation_reasons": [],
                "error": "provider unavailable",
            }
        ],
    }
    assert "signals" not in event["authenticity"]["checks"][0]
    assert "normalized_signals" not in event["authenticity"]["checks"][0]


def test_build_verification_debug_event_omits_authenticity_when_unavailable() -> None:
    event = build_verification_debug_event(
        {"status": "no_document", "handed_off": False, "summary": "Send a document."},
        accepted=None,
        authenticity=None,
    )

    assert "authenticity" not in event
