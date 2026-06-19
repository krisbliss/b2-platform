from __future__ import annotations

import base64
import importlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_CONSISTENCY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "certificate": {
            "type": "object",
            "properties": {
                "full_name":           {"type": "string", "nullable": True},
                "date_of_death":       {"type": "string", "nullable": True},
                "place_of_death":      {"type": "string", "nullable": True},
                "age_at_death":        {"type": "integer", "nullable": True},
                "cause_of_death":      {"type": "string", "nullable": True},
                "certificate_number":  {"type": "string", "nullable": True},
                "issuing_authority":   {"type": "string", "nullable": True},
                "registration_date":   {"type": "string", "nullable": True},
                "other_visible_details": {"type": "object"},
            },
            "required": [
                "full_name", "date_of_death", "place_of_death", "age_at_death",
                "cause_of_death", "certificate_number", "issuing_authority",
                "registration_date", "other_visible_details",
            ],
        },
        "consistency_score":  {"type": "number"},
        "consistency_label":  {"type": "string", "enum": ["high", "moderate", "low"]},
        "confidence":         {"type": "number"},
        "matches":            {"type": "array", "items": {"type": "string"}},
        "mismatches":         {"type": "array", "items": {"type": "string"}},
        "uncertain_points":   {"type": "array", "items": {"type": "string"}},
        "summary":            {"type": "string"},
    },
    "required": [
        "certificate", "consistency_score", "consistency_label", "confidence",
        "matches", "mismatches", "uncertain_points", "summary",
    ],
}

_CONSISTENCY_PROMPT = """You are comparing a chat history against a death certificate image.

Use the chat history and the image together, but make only one model call.

Your tasks:
1. Extract the visible facts from the death certificate image.
2. Compare those facts against the chat history.
3. Produce a narrative consistency score where 1.0 means the chat history and certificate are highly consistent, and 0.0 means they strongly conflict.

Rules:
- Use only information visible in the image and explicitly present in the chat history.
- Do not invent missing facts.
- If a field is unreadable, use null.
- Treat the chat history as the source of narrative claims and the image as the source of certificate facts.

Respond ONLY with structured data matching this schema:
{
  "certificate": {
    "full_name": string|null,
    "date_of_death": string|null,
    "place_of_death": string|null,
    "age_at_death": integer|null,
    "cause_of_death": string|null,
    "certificate_number": string|null,
    "issuing_authority": string|null,
    "registration_date": string|null,
    "other_visible_details": object
  },
  "consistency_score": number,
  "consistency_label": "high"|"moderate"|"low",
  "confidence": number,
  "matches": [string],
  "mismatches": [string],
  "uncertain_points": [string],
  "summary": string
}

Do not include any text outside the structured response."""


def _sniff_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _render_chat_history(chat_history: str | Sequence[Any]) -> str:
    if isinstance(chat_history, str):
        return chat_history.strip()

    lines: list[str] = []
    for item in chat_history:
        if isinstance(item, Mapping):
            role = str(item.get("role", "message")).strip() or "message"
            content = item.get("content", "")
            if isinstance(content, (list, tuple)):
                content = " ".join(str(part) for part in content)
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(item))

    return "\n".join(lines).strip()


def _load_gemini_client() -> tuple[Any, Any]:
    try:
        genai    = importlib.import_module("google.genai")
        gentypes = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for death certificate consistency analysis."
        ) from exc
    return genai, gentypes


def _parse_json_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError("no JSON object in model response")
        raw = match.group()

    match = _JSON_RE.search(raw)
    if match:
        match_text = match.group()
    elif raw.startswith("{") and raw.endswith("}"):
        match_text = raw
    else:
        raise ValueError("no JSON object in model response")

    data = json.loads(match_text)
    if not isinstance(data, dict):
        raise ValueError("model response JSON must be an object")
    return data


def _normalize_structured_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed

    raw_text = getattr(response, "text", "") or ""
    return _parse_json_response(raw_text)


def analyze_death_certificate_consistency(
    chat_history: str | Sequence[Any],
    image_bytes: bytes,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract death certificate facts and score narrative consistency in one Gemini call."""
    transcript = _render_chat_history(chat_history)
    if not transcript:
        raise ValueError("chat_history must not be empty")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    gemini_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required for Gemini calls")

    gemini_model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    genai, gentypes = _load_gemini_client()
    client     = genai.Client(api_key=gemini_api_key)
    image_part = gentypes.Part.from_bytes(
        data=bytes(image_bytes),
        mime_type=_sniff_mime(image_bytes),
    )

    prompt = f"{_CONSISTENCY_PROMPT}\n\nChat history:\n{transcript}\n"

    config = gentypes.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema=_CONSISTENCY_RESPONSE_SCHEMA,
        temperature=0,
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=[image_part, prompt],
        config=config,
    )

    parsed = _normalize_structured_response(response)

    certificate = parsed.get("certificate") or {}
    if not isinstance(certificate, dict):
        certificate = {}

    return {
        "certificate":       certificate,
        "consistency_score": round(_clamp01(parsed.get("consistency_score")), 3),
        "consistency_label": str(parsed.get("consistency_label", "moderate")),
        "confidence":        round(_clamp01(parsed.get("confidence")), 3),
        "matches":           [str(i) for i in parsed.get("matches", [])],
        "mismatches":        [str(i) for i in parsed.get("mismatches", [])],
        "uncertain_points":  [str(i) for i in parsed.get("uncertain_points", [])],
        "summary":           str(parsed.get("summary", "")),
        "model":             gemini_model,
    }


def analyze_death_certificate_consistency_base64(
    chat_history: str | Sequence[Any],
    image_b64: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper — accepts base64-encoded image instead of raw bytes."""
    return analyze_death_certificate_consistency(
        chat_history,
        base64.b64decode(image_b64),
        api_key=api_key,
        model=model,
    )
