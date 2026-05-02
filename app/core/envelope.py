"""Foundation types for envelope-related message metadata."""

from dataclasses import dataclass
from enum import StrEnum


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
