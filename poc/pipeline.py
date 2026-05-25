"""POC-1 pipeline — run_pipeline() with six stubbed stages.

Stage order: intake → document analysis → authenticity → consistency →
             scorer → result

All stage functions are async from the start so POC-2 can drop in real
implementations (the fake_image_detector is async) without changing
run_pipeline's signature.

Stub default values produce:
  sub_scores  = {document: 1.0, authenticity: 0.9, consistency: 0.8}
  weights     = {document: 0.2, authenticity: 0.4, consistency: 0.4}
  weighted    = 1.0×0.2 + 0.9×0.4 + 0.8×0.4 = 0.88
  score       = 88
  band        = Band.HIGH

Hard escalation (AuthenticitySignal.result.escalation == HUMAN_REVIEW or
AUTO_REJECT) forces Band.ESCALATE and appends "HARD_ESCALATION" to flags,
overriding the numeric score band.
"""

from __future__ import annotations

from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

from poc.models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)

# ---------------------------------------------------------------------------
# Default stub values
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_RESULT = ToolResult(
    verdict=Verdict.PASS,
    risk_score=0.1,
    escalation=Escalation.AUTO_ACCEPT,
    checks=[],
)

_STAGE_WEIGHTS: dict[str, float] = {
    "document":     0.2,
    "authenticity": 0.4,
    "consistency":  0.4,
}


# ---------------------------------------------------------------------------
# Stage stubs — POC-2 replaces each body with real implementation
# ---------------------------------------------------------------------------

async def _stage_document(submission: Submission) -> DocumentSignal:
    """Stub: assumes document is legible and typed as death_certificate."""
    return DocumentSignal(
        legible=True,
        document_type="death_certificate",
    )


async def _stage_authenticity(submission: Submission) -> AuthenticitySignal:
    """Stub: returns a clean PASS from the fake_image_detector defaults."""
    return AuthenticitySignal(result=_DEFAULT_TOOL_RESULT)


async def _stage_consistency(submission: Submission) -> ConsistencySignal:
    """Stub: returns high consistency with empty extracted fields."""
    return ConsistencySignal(
        consistency_score=0.8,
        consistency_label="high",
        confidence=0.75,
        summary="Stub pipeline: no LLM analysis performed.",
    )


async def _stage_score(
    doc: DocumentSignal,
    auth: AuthenticitySignal,
    con: ConsistencySignal,
) -> ReliabilityResult:
    """Compute ReliabilityResult from the three signal outputs.

    Hard escalation from the authenticity stage overrides the numeric band
    regardless of score — consistent with GL9_HARD_ESCALATION_FLAGS logic
    already in the fake_image_detector pipeline.
    """
    # Derive per-stage scores from signal outputs
    doc_score  = 1.0 if doc.legible else 0.3
    auth_score = round(1.0 - auth.result.risk_score, 3)
    con_score  = con.consistency_score

    sub_scores: dict[str, float] = {
        "document":     round(doc_score, 3),
        "authenticity": auth_score,
        "consistency":  round(con_score, 3),
    }

    weighted = sum(sub_scores[k] * _STAGE_WEIGHTS[k] for k in _STAGE_WEIGHTS)
    raw_score = max(1, min(100, round(weighted * 100)))

    flags: list[str] = []

    # Hard escalation overrides band — mirrors Escalation.HUMAN_REVIEW /
    # AUTO_REJECT semantics from the fake_image_detector pipeline.
    hard_escalation = auth.result.escalation in (
        Escalation.HUMAN_REVIEW,
        Escalation.AUTO_REJECT,
    )

    if hard_escalation:
        band = Band.ESCALATE
        flags.append("HARD_ESCALATION")
    elif raw_score >= 75:
        band = Band.HIGH
    elif raw_score >= 50:
        band = Band.MEDIUM
    elif raw_score >= 25:
        band = Band.LOW
    else:
        band = Band.ESCALATE

    justification = (
        f"document={'legible' if doc.legible else 'illegible'} "
        f"(type={doc.document_type or 'unknown'}), "
        f"authenticity={auth.result.verdict.value} "
        f"(risk={auth.result.risk_score:.2f}), "
        f"consistency={con.consistency_label} "
        f"(score={con.consistency_score:.2f})"
    )

    return ReliabilityResult(
        score=raw_score,
        band=band,
        sub_scores=sub_scores,
        weights=dict(_STAGE_WEIGHTS),
        flags=flags,
        justification=justification,
        extracted_fields=con.extracted_fields,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_pipeline(submission: Submission) -> ReliabilityResult:
    """Run the full reliability pipeline on a Submission.

    Stages run sequentially: document → authenticity → consistency → score.
    Each stage is async so POC-2 can wire in the real implementations
    (fake_image_detector is async) without touching this function.
    """
    doc_signal  = await _stage_document(submission)
    auth_signal = await _stage_authenticity(submission)
    con_signal  = await _stage_consistency(submission)
    result      = await _stage_score(doc_signal, auth_signal, con_signal)
    return result
