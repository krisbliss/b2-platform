"""Unit tests for the context-aware death-certificate verification tool."""

from types import SimpleNamespace

from tools.death_certificate_pipeline import verify as verify_module
from tools.death_certificate_pipeline.models import Band, ReliabilityResult
from tools.death_certificate_pipeline.pipeline import PipelineExecution
from tools.death_certificate_pipeline.verify import verify_death_certificate
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict


class FakeStore:
    def __init__(self, media):
        self._media = media

    def load_latest_media(self, session_id):
        return self._media


def _ctx(store, history_text="my mother Jane Doe passed away", session_id="wa-123", debug_events=None):
    return SimpleNamespace(
        deps=SimpleNamespace(
            session_id=session_id,
            store=store,
            history_text=history_text,
            debug_events=debug_events,
        )
    )


def _result(band, flags=None):
    return ReliabilityResult(
        score=80 if band in (Band.HIGH, Band.MEDIUM) else 30,
        band=band,
        sub_scores={},
        weights={},
        flags=flags or [],
        justification="ok",
        extracted_fields={"full_name": "Jane Doe"},
    )


def _execution(band, flags=None):
    return PipelineExecution(
        result=_result(band, flags),
        authenticity=ToolResult(
            verdict=Verdict.PASS,
            risk_score=0.1,
            escalation=Escalation.AUTO_ACCEPT,
            checks=[],
        ),
    )


async def test_verify_accepts_and_hands_off(monkeypatch):
    seen = {}

    async def fake_pipeline(submission):
        seen["narrative"] = submission.narrative
        return _execution(Band.HIGH)

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["payload"] = payload
        seen["handoff_image"] = image_bytes
        seen["handoff_mime"] = mime_type
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    result = await verify_death_certificate(_ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg"))))

    assert result["status"] == "verified"
    assert result["band"] == "high"
    assert result["handed_off"] is True
    # narrative came from the conversation history, not the model
    assert "Jane Doe" in seen["narrative"]
    # handoff payload carries a non-reversible contact id and the score
    assert seen["payload"]["score"] == 80
    assert seen["payload"]["contact_identifier"] and len(seen["payload"]["contact_identifier"]) == 64
    assert seen["handoff_image"] == b"\xff\xd8jpeg"
    assert seen["handoff_mime"] == "image/jpeg"


async def test_verify_holds_back_on_hard_escalation(monkeypatch):
    delivered = {"called": False}

    async def fake_pipeline(submission):
        return _execution(Band.ESCALATE, flags=["HARD_ESCALATION"])

    async def fake_deliver(payload, image_bytes, mime_type):
        delivered["called"] = True
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    result = await verify_death_certificate(_ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg"))))

    assert result["status"] == "verified"
    assert result["handed_off"] is False
    assert delivered["called"] is False


async def test_verify_reports_no_document_when_store_empty(monkeypatch):
    async def fake_pipeline(submission):  # pragma: no cover - must not run
        raise AssertionError("pipeline should not run without media")

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)

    result = await verify_death_certificate(_ctx(FakeStore(None), history_text=""))

    assert result["status"] == "no_document"
    assert result["handed_off"] is False


async def test_verify_appends_debug_event(monkeypatch):
    async def fake_pipeline(submission):
        return _execution(Band.MEDIUM)

    async def fake_deliver(payload, image_bytes, mime_type):
        return False

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    debug_events = []
    result = await verify_death_certificate(
        _ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg")), debug_events=debug_events)
    )

    assert result["status"] == "verified"
    assert debug_events == [
        {
            "tool": "death_certificate_verification",
            "status": "verified",
            "score": 80,
            "band": "medium",
            "accepted": True,
            "handed_off": False,
            "flags": [],
            "extracted_fields": {"full_name": "Jane Doe"},
            "summary": result["summary"],
            "authenticity": {
                "verdict": "PASS",
                "risk_score": 0.1,
                "escalation": "AUTO_ACCEPT",
                "early_exit": False,
                "early_exit_reason": None,
                "checks": [],
            },
        }
    ]
