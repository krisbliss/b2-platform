"""Compatibility re-exports for death certificate consistency analysis."""

from tools.death_certificate_pipeline.death_certificate_consistency import (
    analyze_death_certificate_consistency,
    analyze_death_certificate_consistency_base64,
)

__all__ = [
    "analyze_death_certificate_consistency",
    "analyze_death_certificate_consistency_base64",
]
