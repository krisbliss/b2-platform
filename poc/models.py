"""POC-1 data models — Submission, signal types, ReliabilityResult.

All models are Pydantic BaseModel so they serialise to JSON natively and
integrate with FastAPI response_model without extra boilerplate.

Design decisions (locked via grill-me session):
- Pydantic throughout (matches fake_image_detector/models.py)
- Submission.image: bytes — I/O conversion happens at the boundary (API/CLI)
- case_fields: dict[str, Any] — free-form, scorer defines structure later
- AuthenticitySignal wraps ToolResult directly (no flattening)
- extracted_fields: dict[str, Any] — not typed until scorer consumes it
- Band: single four-value enum; "escalate" overrides quality bands
- sub_scores: 0.0–1.0 floats; final score: int 1–100
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from tools.fake_image_detector.models import ToolResult


# ---------------------------------------------------------------------------
# Band enum
# ---------------------------------------------------------------------------

class Band(str, Enum):
    """Customer-facing reliability band.

    HIGH     — score 75–100: strong evidence, proceed with confidence.
    MEDIUM   — score 50–74: reasonable evidence, minor gaps present.
    LOW      — score 25–49: significant gaps, manual follow-up advised.
    ESCALATE — score 0–24 OR hard flag from authenticity stage: requires
               immediate human review regardless of numeric score.
    """

    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Submission — the input contract
# ---------------------------------------------------------------------------

class Submission(BaseModel):
    """Incoming case submission.

    narrative is a first-class field (not buried in case_fields) because it
    is the primary claim being cross-referenced against the certificate.
    image is always bytes — conversion from Path or upload happens at the
    API/CLI boundary before Submission is constructed.
    """

    image: bytes
    narrative: str
    case_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("narrative")
    @classmethod
    def narrative_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be blank")
        return v

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Signal types — one per pipeline stage
# ---------------------------------------------------------------------------

class DocumentSignal(BaseModel):
    """Output of the document analysis stage.

    Answers: can we read this document and what type is it?
    Does NOT score fakeness — that is AuthenticitySignal's job.
    """

    legible: bool = False
    document_type: str | None = None
    page_count: int = 1
    language_hint: str | None = None
    notes: list[str] = Field(default_factory=list)
    stage: str = "document_analysis"


class AuthenticitySignal(BaseModel):
    """Output of the authenticity stage.

    Wraps ToolResult from the fake_image_detector pipeline directly.
    POC-2 constructs this as AuthenticitySignal(result=tool_result).
    """

    result: ToolResult
    stage: str = "authenticity"

    model_config = {"arbitrary_types_allowed": True}


class ConsistencySignal(BaseModel):
    """Output of the narrative consistency stage.

    Maps directly from analyze_death_certificate_consistency() output.
    extracted_fields carries certificate data through to ReliabilityResult
    where it is surfaced to the customer.

    Note: POC-2 maps 'mismatches' → 'contradictions' when constructing this.
    """

    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_label: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    matches: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    uncertain_points: list[str] = Field(default_factory=list)
    summary: str = ""
    stage: str = "consistency"


# ---------------------------------------------------------------------------
# ReliabilityResult — the output contract
# ---------------------------------------------------------------------------

class ReliabilityResult(BaseModel):
    """Final customer-facing reliability verdict.

    score        — 1–100 integer computed from weighted sub_scores.
    band         — customer-facing quality band (see Band enum).
    sub_scores   — per-stage scores (0.0–1.0); keys: document, authenticity,
                   consistency.
    weights      — per-stage weights summing to 1.0.
    flags        — list of named concerns, e.g. ["HARD_ESCALATION"].
    justification — human-readable explanation of the verdict.
    extracted_fields — certificate data from ConsistencySignal, surfaced
                       directly to the customer (POC-2 populates this).
    """

    score: int = Field(ge=1, le=100)
    band: Band
    sub_scores: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    justification: str
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
