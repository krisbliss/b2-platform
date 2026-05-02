"""Foundation types for envelope-related message metadata."""

import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Any


class InputType(StrEnum):
    """Identifies the kind of user input contained in an envelope."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    LOCATION = "location"


class SubmissionType(StrEnum):
    """Categorizes why a submission was provided to the system."""

    INTAKE = "intake"
    VERIFICATION_EVIDENCE = "verification_evidence"
    MONITORING_CHECKIN = "monitoring_checkin"
    UNKNOWN = "unknown"


class LocationSource(StrEnum):
    """Describes where location context was derived from, if present."""

    DEVICE = "device"
    PROMPT = "prompt"
    PHONE_PREFIX = "phone_prefix"
    NONE = "none"


class LogEventType(StrEnum):
    """Names the core lifecycle events emitted while handling an envelope."""

    MESSAGE_RECEIVED = "message_received"
    MESSAGE_NORMALIZED = "message_normalized"
    AGENT_PROMPT_CREATED = "agent_prompt_created"
    RESPONSE_SENT = "response_sent"
    ERROR = "error"

# hardcode these fields from being shared to agent
_AGENT_PROMPT_FORBIDDEN_KEYS = {
    "channel",
    "channel_user_id",
    "coordinates",
    "gps",
    "lat",
    "latitude",
    "lng",
    "longitude",
    "media_url",
    "phone",
    "phone_number",
    "raw_channel_user_id",
    "raw_phone_number",
    "session_id",
}
_PHONE_NUMBER_PATTERN = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def make_session_id(channel_user_id: str, channel: str) -> str:
    """Returns a daily rotating pseudonymous session identifier."""

    return _make_session_id_for_date(channel_user_id, channel, datetime.now(timezone.utc).date())


def _make_session_id_for_date(channel_user_id: str, channel: str, day: date) -> str:
    """Returns the SHA-256 hex digest for a user/channel/day tuple."""

    payload = "\0".join((channel, day.isoformat(), channel_user_id))
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class LocationContext:
    """Stores only safe approximate location metadata and never coordinates."""

    country_code: str | None = None
    region: str | None = None
    city: str | None = None
    source: LocationSource = LocationSource.NONE
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be between 0.0 and 1.0 inclusive"
            raise ValueError(msg)

    def to_safe_dict(self) -> dict[str, str | float | None]:
        """Returns the permitted approximate location fields for serialization."""

        return {
            "country_code": self.country_code,
            "region": self.region,
            "city": self.city,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CanonicalMessage:
    """Normalized message envelope shared across channel-specific adapters."""

    session_id: str
    channel: str
    input_type: InputType
    text_content: str | None = None
    media_url: str | None = None
    language_hint: str | None = None
    location_context: LocationContext | None = None
    session_context: dict[str, Any] = field(default_factory=dict)
    submission_type: SubmissionType = SubmissionType.UNKNOWN
    prior_context: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_agent_prompt(self) -> dict[str, Any]:
        """Returns a privacy-safe prompt payload for agent execution."""

        prompt: dict[str, Any] = {"input_type": self.input_type}

        if self.text_content is not None:
            prompt["text_content"] = self.text_content
        if self.language_hint is not None:
            prompt["language_hint"] = self.language_hint
        if self.location_context is not None:
            prompt["location_context"] = self.location_context.to_safe_dict()
        if self.submission_type is not SubmissionType.UNKNOWN:
            prompt["submission_type"] = self.submission_type
        if self.prior_context and _is_agent_prompt_safe(self.prior_context):
            prompt["prior_context"] = deepcopy(self.prior_context)
        if self.session_context and _is_agent_prompt_safe(self.session_context):
            prompt["session_context"] = deepcopy(self.session_context)

        return prompt


def _is_agent_prompt_safe(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _AGENT_PROMPT_FORBIDDEN_KEYS:
                return False
            if not _is_agent_prompt_safe(child):
                return False
        return True

    if isinstance(value, list | tuple):
        return all(_is_agent_prompt_safe(item) for item in value)

    if isinstance(value, str):
        return not _contains_phone_number(value)

    return True


def _contains_phone_number(value: str) -> bool:
    return any(
        len(re.sub(r"\D", "", match.group(0))) >= 10
        for match in _PHONE_NUMBER_PATTERN.finditer(value)
    )
