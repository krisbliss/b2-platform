"""Compatibility re-exports — canonical implementation moved to tools/death_certificate_pipeline/."""

from tools.death_certificate_pipeline.models import (  # noqa: F401
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)
