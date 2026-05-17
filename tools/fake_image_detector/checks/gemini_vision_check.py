from __future__ import annotations

import asyncio
import json
import os
import re

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import CheckResult, NormalizedSignals

_PHOTO_PROMPT = """You are a fraud-detection assistant. Analyze this image and assess whether it is a genuine, original photograph submitted by a real person, or whether it is deceptive in any of the following ways:

1. AI-GENERATED or SYNTHETIC — produced by a GAN, diffusion model, or other generative system.
2. DIGITALLY MANIPULATED — a real photo that has been edited, composited, or had elements added/removed.
3. STAGED or STOCK — a professional shoot, stock image, or heavily posed photo unlikely to be a genuine personal submission.
4. BACKGROUND INCONSISTENCY — background does not match the subject (composited, greenscreen, mismatched lighting).
5. LIGHTING or SHADOW INCONSISTENCY — light sources or shadows are inconsistent across the image.

Do NOT make a final verdict. Report only the signals you observe.

Respond ONLY with valid JSON matching this schema:
{
  "is_deceptive": <bool, true if the image is AI-generated, manipulated, staged, or otherwise not a genuine personal photo>,
  "fake_likelihood": <float 0.0-1.0, probability the image is deceptive>,
  "confidence": <float 0.0-1.0, how certain you are about your assessment>,
  "signals": [<list of short specific observed-indicator strings, e.g. "skin texture too smooth", "background composited">],
  "flags": [<zero or more from: GAN_ARTIFACTS, DIFFUSION_ARTIFACTS, EDITING_DETECTED,
             INCONSISTENT_LIGHTING, UNNATURAL_TEXTURE, BACKGROUND_INCONSISTENCY,
             STOCK_PHOTO_INDICATORS, STAGING_ARTIFACTS, METADATA_MISMATCH, CLEAN>]
}
Do not include any text outside the JSON object."""

_DOCUMENT_PROMPT_TEMPLATE = """\
You are a fraud-detection assistant. This image has been identified as a {doc_type}{country_clause}.
{extracted_section}
Assess whether this appears to be a GENUINE, AUTHENTIC document or whether it is deceptive in any of the following ways:

1. FORGED or FABRICATED — a printed template, photoshop creation, or entirely made-up document.
2. PHOTO OF A PHOTO — a photograph taken of another physical document or screen.
3. DIGITALLY MANIPULATED — an authentic document with altered fields (name, date, number).
4. TEMPLATE DETECTED — produced from an online template without official security features.
5. INCONSISTENT SECURITY FEATURES — missing expected holograms, watermarks, or official markings.

Do NOT make a final verdict. Report only the signals you observe.

Respond ONLY with valid JSON matching this schema:
{{
  "is_deceptive": <bool, true if the document appears forged, altered, or not genuine>,
  "fake_likelihood": <float 0.0-1.0, probability the document is not genuine>,
  "confidence": <float 0.0-1.0, how certain you are about your assessment>,
  "signals": [<list of short specific observed-indicator strings, e.g. "no visible hologram", "font inconsistency on expiry date">],
  "flags": [<zero or more from: FORGED_DOCUMENT, PHOTO_OF_PHOTO, EDITING_DETECTED,
             TEMPLATE_DETECTED, INCONSISTENT_SECURITY_FEATURES, CLEAN>]
}}
Do not include any text outside the JSON object.\
"""

# Strip optional markdown code fences Gemini sometimes wraps around JSON
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9 _-]+")


def _sanitize_doc_type(value: object) -> str:
    if value is None:
        return "document"
    cleaned = _SAFE_TOKEN_RE.sub("", str(value)).strip().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:64] if cleaned else "document"


def _sanitize_country(value: object) -> str:
    if value is None:
        return ""
    cleaned = _SAFE_TOKEN_RE.sub("", str(value)).strip().upper()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned[:3]


def _sniff_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class GeminiVisionCheck(BaseCheck):
    check_id = "gemini_vision"

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(1, max_retries)

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes, context),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return self._error_result("Gemini vision check timed out", "CHECK_TIMEOUT")

    def _error_result(self, error: str, flag: str = "CHECK_RUNTIME_ERROR") -> CheckResult:
        return CheckResult(
            check=self.check_id,
            passed=False,
            fake_score=1.0,
            confidence=1.0,
            flags=[flag],
            signals={"error": error},
            skipped=False,
            error=error,
        )

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            from google import genai
            from google.genai import types as gentypes
        except ImportError as e:
            return self._error_result(str(e))

        if not self._project:
            return self._error_result("GOOGLE_CLOUD_PROJECT not set")

        doc_type = context.get("doc_type")
        country = context.get("country")

        if doc_type:
            safe_doc_type = _sanitize_doc_type(doc_type)
            safe_country = _sanitize_country(country)
            country_clause = f" from {safe_country}" if safe_country else ""
            extracted = context.get("extracted_fields", {})
            if extracted:
                lines = "\n".join(f"  - {k}: {v}" for k, v in extracted.items())
                extracted_section = f"\nExtracted fields:\n{lines}\n"
            else:
                extracted_section = ""
            prompt = _DOCUMENT_PROMPT_TEMPLATE.format(
                doc_type=safe_doc_type,
                country_clause=country_clause,
                extracted_section=extracted_section,
            )
        else:
            safe_doc_type = None
            safe_country = ""
            prompt = _PHOTO_PROMPT

        model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

        raw = ""
        for attempt in range(self._max_retries):
            try:
                client = genai.Client(
                    vertexai=True, project=self._project, location=self._location
                )
                image_part = gentypes.Part.from_bytes(
                    data=image_bytes, mime_type=_sniff_mime(image_bytes)
                )
                response = client.models.generate_content(
                    model=model, contents=[image_part, prompt]
                )
                raw = response.text
                break
            except Exception as e:
                is_last_attempt = attempt == self._max_retries - 1
                if is_last_attempt:
                    return self._error_result(str(e))
                continue

        try:
            match = _JSON_RE.search(raw)
            if not match:
                raise ValueError("no JSON object in response")
            data = json.loads(match.group())
        except Exception as e:
            raw_snippet = raw[:120] if raw else ""
            return self._error_result(
                f"JSON parse error: {e}; raw_snippet={raw_snippet!r}",
                "GEMINI_PARSE_ERROR",
            )

        fake_likelihood = max(0.0, min(1.0, float(data.get("fake_likelihood", 0.0))))
        gemini_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        is_deceptive = bool(data.get("is_deceptive", data.get("is_synthetic", False)))
        flags = [str(f) for f in data.get("flags", [])]
        signals = {
            "doc_type": safe_doc_type if doc_type else None,
            "country": safe_country if doc_type else "",
            "is_deceptive": is_deceptive,
            "signals": data.get("signals", []),
        }

        return CheckResult(
            check=self.check_id,
            passed=not is_deceptive,
            fake_score=round(fake_likelihood, 3),
            confidence=round(gemini_confidence, 3),
            flags=flags,
            signals=signals,
            normalized_signals=NormalizedSignals(
                category="document_authenticity" if doc_type else "synthetic",
                confidence=round(gemini_confidence, 3),
                indicators=flags or ["CLEAN"],
                document_type=safe_doc_type if doc_type else None,
                country_code=safe_country if doc_type else None,
                synthetic_score=round(fake_likelihood, 3) if not doc_type else None,
                manipulation_score=round(fake_likelihood, 3),
                staging_score=round(fake_likelihood, 3) if any(
                    f in {"STAGING_ARTIFACTS", "STOCK_PHOTO_INDICATORS"} for f in flags
                ) else None,
            ),
        )
