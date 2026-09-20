"""Bounded E2E diagnostics for death-certificate verification."""

from __future__ import annotations

from typing import Any

from tools.fake_image_detector.models import ToolResult


def build_verification_debug_event(
    payload: dict[str, Any],
    accepted: bool | None,
    authenticity: ToolResult | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe debug event without raw document-derived signals."""
    event: dict[str, Any] = {
        "tool": "death_certificate_verification",
        "status": payload.get("status"),
        "score": payload.get("score"),
        "band": payload.get("band"),
        "accepted": accepted,
        "handed_off": bool(payload.get("handed_off", False)),
        "flags": list(payload.get("flags", [])),
        "extracted_fields": dict(payload.get("extracted_fields", {})),
        "summary": str(payload.get("summary", "")),
    }
    if authenticity is not None:
        event["authenticity"] = {
            "verdict": authenticity.verdict.value,
            "risk_score": authenticity.risk_score,
            "escalation": authenticity.escalation.value,
            "early_exit": authenticity.early_exit,
            "early_exit_reason": authenticity.early_exit_reason,
            "checks": [
                {
                    "check": check.check,
                    "passed": check.passed,
                    "skipped": check.skipped,
                    "fake_score": check.fake_score,
                    "confidence": check.confidence,
                    "flags": list(check.flags),
                    "human_escalate": check.human_escalate,
                    "escalation_reasons": list(check.escalation_reasons),
                    "error": check.error,
                }
                for check in authenticity.checks
            ],
        }
    return event
