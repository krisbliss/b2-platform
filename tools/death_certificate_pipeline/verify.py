"""Context-aware death-certificate verification tool.

Registered as the ``death_certificate_verification`` agent tool. The model calls
it with no arguments once a user has sent a document; the image itself is never
seen by the model. This handler pulls the user's most recent image from the
transient session store (via the run's SessionContext), runs the full
reliability pipeline, and — when the result clears the acceptance policy — hands
the case off to GiveLight.

Flow: SessionContext → latest image → run_pipeline → (policy) → handoff.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from tools.death_certificate_pipeline.handoff import build_handoff_payload, deliver_to_gl
from tools.death_certificate_pipeline.models import Band, ReliabilityResult, Submission
from tools.death_certificate_pipeline.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Bands that are strong enough to forward to GiveLight automatically. Anything
# lower — or any hard escalation flag from the authenticity stage — is held back
# for human review instead.
_ACCEPT_BANDS = {Band.HIGH, Band.MEDIUM}
_NARRATIVE_FALLBACK = "No prior conversation details were provided by the claimant."


def _contact_identifier(session_id: str | None) -> str | None:
    """Stable, non-reversible join key for GiveLight (placeholder until GL-20)."""
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _accepted(result: ReliabilityResult) -> bool:
    return result.band in _ACCEPT_BANDS and "HARD_ESCALATION" not in result.flags


async def verify_death_certificate(ctx: Any) -> dict[str, Any]:
    """Verify the user's most recent document and hand off to GiveLight if it passes."""
    deps = getattr(ctx, "deps", None)
    session_id = getattr(deps, "session_id", None)
    store = getattr(deps, "store", None)

    media = store.load_latest_media(session_id) if (store is not None and session_id) else None
    if media is None:
        logger.info("verify.no_media session=%.8s", str(session_id or ""))
        return {
            "status": "no_document",
            "handed_off": False,
            "summary": "No document has been received yet — ask the user to send a photo or PDF of the death certificate.",
        }

    image_bytes, _mime = media
    narrative = (getattr(deps, "history_text", "") or "").strip() or _NARRATIVE_FALLBACK

    submission = Submission(
        image=image_bytes,
        narrative=narrative,
        case_fields={"channel": "whatsapp", "contact_identifier": _contact_identifier(session_id)},
    )

    result = await run_pipeline(submission)
    handed_off = False

    if _accepted(result):
        payload = build_handoff_payload(
            result,
            contact_identifier=_contact_identifier(session_id),
            case_fields=submission.case_fields,
        )
        handed_off = await deliver_to_gl(payload)

    logger.info(
        "verify.done session=%.8s score=%d band=%s accepted=%s handed_off=%s",
        str(session_id or ""),
        result.score,
        result.band.value,
        _accepted(result),
        handed_off,
    )

    return {
        "status": "verified",
        "score": result.score,
        "band": result.band.value,
        "handed_off": handed_off,
        "flags": result.flags,
        "extracted_fields": result.extracted_fields,
        "summary": (
            "Verification passed and the case was forwarded to GiveLight."
            if handed_off
            else "Verification did not meet the automatic-approval threshold; the case needs human review. "
            "Consider asking the user for a clearer photo of the full certificate."
        ),
    }
