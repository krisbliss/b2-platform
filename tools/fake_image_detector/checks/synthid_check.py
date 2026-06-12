from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import (
    CheckResult,
    GL9_FLAG_LIKELY_AI_GENERATED,
    NormalizedSignals,
    SYNTHID_AI_GENERATED_THRESHOLD,
)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _first_numeric(data: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return _clamp01(data.get(key), default=default)
    return _clamp01(default)


def _first_bool(data: dict[str, Any], keys: tuple[str, ...], default: bool = False) -> bool:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
    return default


class VertexSynthIDCheck(BaseCheck):
    check_id = "synthid"

    def __init__(
        self,
        params: dict | None = None,
        *,
        project: str | None = None,
        location: str | None = None,
        endpoint_id: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        params = params or {}
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._endpoint_id = endpoint_id or params.get("endpoint_id") or os.environ.get("VERTEX_SYNTHID_ENDPOINT_ID")
        self._timeout_seconds = float(params.get("timeout_seconds", timeout_seconds))
        self._max_retries = max(1, int(params.get("max_retries", max_retries)))

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        if not self._project or not self._endpoint_id:
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                signals={
                    "reason": "missing project or endpoint",
                    "project_set": bool(self._project),
                    "endpoint_set": bool(self._endpoint_id),
                },
            )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return CheckResult(
                check=self.check_id,
                passed=True,
                fake_score=0.5,
                confidence=0.0,
                flags=["DETECTOR_UNAVAILABLE"],
                signals={"error": "SynthID check timed out", "fallback": "neutral"},
                error="SynthID check timed out",
            )

    def _run_sync(self, image_bytes: bytes) -> CheckResult:
        try:
            from google.cloud import aiplatform_v1
            from google.protobuf.json_format import MessageToDict
        except Exception as exc:
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                signals={"reason": "aiplatform import unavailable", "error": str(exc)},
                error=str(exc),
            )

        endpoint = f"projects/{self._project}/locations/{self._location}/endpoints/{self._endpoint_id}"
        api_endpoint = f"{self._location}-aiplatform.googleapis.com"
        instance = {
            "image_bytes": base64.b64encode(image_bytes).decode("ascii"),
            "mime_type": "image/jpeg",
        }

        last_error: str | None = None
        for attempt in range(self._max_retries):
            try:
                client = aiplatform_v1.PredictionServiceClient(
                    client_options={"api_endpoint": api_endpoint}
                )
                response = client.predict(endpoint=endpoint, instances=[instance])

                prediction_obj = None
                if getattr(response, "predictions", None):
                    prediction_obj = response.predictions[0]

                if prediction_obj is None:
                    raise ValueError("empty SynthID prediction payload")

                prediction = MessageToDict(prediction_obj)
                if not isinstance(prediction, dict):
                    raise ValueError("SynthID prediction must be an object")

                synth_score = _first_numeric(
                    prediction,
                    ("watermark_likelihood", "synthid_score", "score", "synthetic_probability"),
                )
                confidence = _first_numeric(prediction, ("confidence",), default=synth_score)
                detected = _first_bool(
                    prediction,
                    ("watermark_detected", "is_synthetic", "detected"),
                    default=synth_score >= 0.5,
                )

                flags = ["SYNTHID_WATERMARK_DETECTED"] if detected else ["CLEAN"]
                if synth_score >= SYNTHID_AI_GENERATED_THRESHOLD:
                    flags.append(GL9_FLAG_LIKELY_AI_GENERATED)

                return CheckResult(
                    check=self.check_id,
                    passed=not detected,
                    fake_score=round(synth_score, 3),
                    confidence=round(confidence, 3),
                    flags=flags,
                    signals={"prediction": prediction, "detected": detected},
                    normalized_signals=NormalizedSignals(
                        category="synthetic",
                        confidence=round(confidence, 3),
                        indicators=flags,
                        synthetic_score=round(synth_score, 3),
                    ),
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt == self._max_retries - 1:
                    break

        return CheckResult(
            check=self.check_id,
            passed=True,
            fake_score=0.5,
            confidence=0.0,
            flags=["DETECTOR_UNAVAILABLE"],
            signals={"error": last_error or "unknown error", "fallback": "neutral"},
            error=last_error or "unknown error",
        )
