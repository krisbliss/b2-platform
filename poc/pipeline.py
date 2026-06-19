"""Compatibility re-exports — canonical implementation moved to tools/death_certificate_pipeline/."""

from tools.death_certificate_pipeline.pipeline import (  # noqa: F401
    _stage_authenticity,
    _stage_consistency,
    _stage_document,
    _stage_score,
    run_pipeline,
)
