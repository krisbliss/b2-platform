from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class CheckConfig:
    id: str
    enabled: bool
    early_exit_on_fail: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class GeminiConfig:
    enabled: bool = False
    timeout_seconds: float = 30.0
    max_retries: int = 3
    max_concurrency: int = 5


@dataclass
class PipelineConfig:
    clear_fail: float
    clear_pass: float
    checks: list[CheckConfig] = field(default_factory=list)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    path = path or CONFIG_DIR / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    thresholds = raw.get("thresholds", {})
    if "clear_fail" not in thresholds or "clear_pass" not in thresholds:
        raise ValueError(f"Missing required thresholds in pipeline config: {path}")

    gemini_raw = raw.get("gemini", {})
    gemini = GeminiConfig(
        enabled=bool(gemini_raw.get("enabled", False)),
        timeout_seconds=float(gemini_raw.get("timeout_seconds", 30.0)),
        max_retries=int(gemini_raw.get("max_retries", 3)),
        max_concurrency=int(gemini_raw.get("max_concurrency", 5)),
    )

    return PipelineConfig(
        clear_fail=thresholds["clear_fail"],
        clear_pass=thresholds["clear_pass"],
        checks=[
            CheckConfig(
                id=c["id"],
                enabled=c["enabled"],
                early_exit_on_fail=c.get("early_exit_on_fail", False),
                params=c.get("params", {}),
            )
            for c in raw.get("checks", [])
        ],
        gemini=gemini,
    )


def load_document_schemas(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "document_schemas.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
