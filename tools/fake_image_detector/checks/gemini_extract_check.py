from __future__ import annotations

import asyncio
import json
import os
import re

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import CheckResult

# Fields to extract per document type, based on what each document typically contains.
_EXTRACT_FIELDS: dict[str, list[str]] = {
    "passport": ["full_name", "document_number", "date_of_birth", "expiry_date", "nationality"],
    "national_id": ["full_name", "id_number", "date_of_birth", "expiry_date"],
    "birth_certificate": ["full_name", "date_of_birth", "place_of_birth", "signed_by"],
    "death_certificate": ["full_name", "date_of_death", "place_of_death", "signed_by"],
    "driving_license": ["full_name", "licence_number", "date_of_birth", "expiry_date"],
    "bank_statement": ["account_holder", "iban", "account_number"],
}

_EXTRACT_PROMPT_TEMPLATE = """\
This image contains a {doc_type}. Extract the following fields exactly as they appear in the document:

{field_list}

Respond ONLY with a valid JSON object. Use null for any field that is not visible or not present.
Example: {{"full_name": "Max Mustermann", "date_of_birth": "1985-03-15"}}
Do not include any text outside the JSON object.\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _sniff_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class GeminiExtractCheck(BaseCheck):
    check_id = "gemini_extract"

    def __init__(self, params: dict | None = None, project: str | None = None, location: str | None = None):
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes, context)

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        doc_type = context.get("doc_type")
        if not doc_type:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        fields = _EXTRACT_FIELDS.get(doc_type)
        if not fields:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        try:
            from google import genai
            from google.genai import types as gentypes
        except ImportError as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        if not self._project:
            return CheckResult(
                check=self.check_id, passed=True, confidence=0.0,
                skipped=True, error="GOOGLE_CLOUD_PROJECT not set",
            )

        field_list = "\n".join(f"- {f}" for f in fields)
        prompt = _EXTRACT_PROMPT_TEMPLATE.format(
            doc_type=doc_type.replace("_", " "),
            field_list=field_list,
        )
        model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

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
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            match = _JSON_RE.search(raw)
            if not match:
                raise ValueError("no JSON object in response")
            extracted = json.loads(match.group())
        except Exception as e:
            return CheckResult(
                check=self.check_id, passed=True, confidence=0.0,
                skipped=True, error=f"JSON parse error: {e}",
            )

        # Drop null values; store the rest for downstream checks and the auth prompt
        extracted_fields = {k: v for k, v in extracted.items() if v is not None}
        context["extracted_fields"] = extracted_fields

        return CheckResult(
            check=self.check_id,
            passed=True,
            confidence=0.0,
            skipped=True,
            signals={"doc_type": doc_type, "extracted": extracted_fields},
        )
