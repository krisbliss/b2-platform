"""Foundation enums for envelope-related message metadata."""

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
