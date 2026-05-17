from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"
    REJECT = "REJECT"


class Escalation(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    AUTO_REJECT = "AUTO_REJECT"


GL9_FLAG_FOUND_ONLINE = "FOUND_ONLINE"
GL9_FLAG_POSSIBLE_STOCK = "POSSIBLE_STOCK"
GL9_FLAG_EDITING_ARTIFACTS = "EDITING_ARTIFACTS"
GL9_FLAG_AGE_INCONSISTENCY = "AGE_INCONSISTENCY"

GL9_HARD_ESCALATION_FLAGS = {
    GL9_FLAG_FOUND_ONLINE,
    GL9_FLAG_POSSIBLE_STOCK,
}


class NormalizedSignals(BaseModel):
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    indicators: list[str] = Field(default_factory=list)
    document_type: str | None = None
    country_code: str | None = None
    synthetic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    manipulation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    staging_score: float | None = Field(default=None, ge=0.0, le=1.0)


class CheckResult(BaseModel):
    check: str
    passed: bool
    fake_score: float = Field(default=0.0, ge=0.0, le=1.0)  # P(fake) — drives risk
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # reliability weight of this check
    flags: list[str] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)
    normalized_signals: NormalizedSignals | None = None
    skipped: bool = False
    error: str | None = None
    human_escalate: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    verdict: Verdict
    risk_score: float = Field(ge=0.0, le=1.0)
    escalation: Escalation
    checks: list[CheckResult]
    early_exit: bool = False
    early_exit_reason: str | None = None
