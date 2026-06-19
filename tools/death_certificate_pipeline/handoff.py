"""GL handoff — package the scored result for GiveLight delivery.

Stub implementation. Wire real GL endpoint when GL-20 lands.

The handoff bundle contains:
  - contact_identifier  (HMAC-SHA256 join key — no raw PII)
  - ReliabilityResult   (score, band, flags, extracted_fields)
  - case_fields         (operator metadata from Submission)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from tools.death_certificate_pipeline.models import ReliabilityResult

logger = logging.getLogger(__name__)


def build_handoff_payload(
    result: ReliabilityResult,
    contact_identifier: str | None,
    case_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serialisable payload sent to GiveLight."""
    return {
        "schema_version":      "1.0",
        "submitted_at":        datetime.now(timezone.utc).isoformat(),
        "contact_identifier":  contact_identifier,
        "score":               result.score,
        "band":                result.band.value,
        "sub_scores":          result.sub_scores,
        "flags":               result.flags,
        "justification":       result.justification,
        "extracted_fields":    result.extracted_fields,
        "case_fields":         case_fields,
    }


async def deliver_to_gl(payload: dict[str, Any]) -> bool:
    """Stub: log the handoff payload. Replace with real GL endpoint (GL-20).

    Returns True on confirmed delivery, False on failure.
    """
    logger.info(
        "gl.handoff_stub score=%d band=%s contact=%.8s — GL-20 endpoint not yet wired",
        payload.get("score"),
        payload.get("band"),
        str(payload.get("contact_identifier") or ""),
    )
    logger.debug("gl.handoff_payload %s", json.dumps(payload, ensure_ascii=False))
    return True
