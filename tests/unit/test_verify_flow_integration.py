"""Integration test: real orchestrator agent + verify tool, LLM faked with TestModel.

Drives the novel mechanism end to end — the model calls the context-aware tool,
which pulls the user's image from the transient store (via SessionContext deps)
and runs the pipeline + handoff. Only the external boundaries (the LLM, the
reliability pipeline, and GiveLight delivery) are faked; the tool registration,
deps threading, and image-through-context path are all exercised for real.
"""

from pathlib import Path

from pydantic_ai.models.test import TestModel

from src.orchestrator import agent as agent_module
from src.orchestrator.context import SessionContext
from src.session import Session
from tools.death_certificate_pipeline import verify as verify_module
from tools.death_certificate_pipeline.models import Band, ReliabilityResult

_AGENT_YAML = Path(__file__).resolve().parents[2] / "agents" / "poc_deathCertParserAgent.yaml"


class _FakeStore:
    def __init__(self, media):
        self._media = media
        self.pulled_for = None

    def load_latest_media(self, session_id):
        self.pulled_for = session_id
        return self._media


def test_model_call_pulls_image_and_hands_off(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")

    seen = {}

    async def fake_pipeline(submission):
        seen["narrative"] = submission.narrative
        seen["image"] = submission.image
        return ReliabilityResult(
            score=82, band=Band.HIGH, sub_scores={}, weights={}, flags=[],
            justification="ok", extracted_fields={"full_name": "Jane Doe"},
        )

    async def fake_deliver(payload):
        seen["handoff"] = payload
        return True

    monkeypatch.setattr(verify_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    definition = agent_module._load_agent_definition_from_file(_AGENT_YAML)
    agent = agent_module.Agent(definition)

    store = _FakeStore((b"\xff\xd8jpeg-image-bytes", "image/jpeg"))
    deps = SessionContext(
        session_id="wa-1",
        store=store,
        history_text="My mother Jane Doe passed away last week.",
    )

    with agent.pydantic_ai_agent.override(model=TestModel()):
        session = Session(agent, deps=deps)
        "".join(session.send_stream("The user has just uploaded a document image."))

    # the tool pulled the image for the right session, out of band from the model
    assert store.pulled_for == "wa-1"
    assert seen["image"] == b"\xff\xd8jpeg-image-bytes"
    assert seen["narrative"] == "My mother Jane Doe passed away last week."
    # and the passing result was forwarded to GiveLight
    assert seen["handoff"]["score"] == 82
    assert len(seen["handoff"]["contact_identifier"]) == 64
