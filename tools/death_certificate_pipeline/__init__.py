"""Death certificate pipeline — scoring, consistency analysis, and GL handoff."""

from .death_certificate_consistency import (
    analyze_death_certificate_consistency,
    analyze_death_certificate_consistency_base64,
)
from .models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)
from .pipeline import run_pipeline

__all__ = [
    "analyze_death_certificate_consistency",
    "analyze_death_certificate_consistency_base64",
    "AuthenticitySignal",
    "Band",
    "ConsistencySignal",
    "DocumentSignal",
    "ReliabilityResult",
    "Submission",
    "run_pipeline",
]
