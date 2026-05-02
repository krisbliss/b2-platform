from app.core.envelope import InputType, LocationSource, LogEventType, SubmissionType


def test_input_type_values_are_lowercase_strings() -> None:
    assert InputType.TEXT == "text"
    assert InputType.IMAGE == "image"
    assert InputType.AUDIO == "audio"
    assert InputType.DOCUMENT == "document"
    assert InputType.LOCATION == "location"


def test_submission_type_values_are_lowercase_strings() -> None:
    assert SubmissionType.INTAKE == "intake"
    assert SubmissionType.VERIFICATION_EVIDENCE == "verification_evidence"
    assert SubmissionType.MONITORING_CHECKIN == "monitoring_checkin"
    assert SubmissionType.UNKNOWN == "unknown"


def test_location_source_values_are_lowercase_strings() -> None:
    assert LocationSource.DEVICE == "device"
    assert LocationSource.PROMPT == "prompt"
    assert LocationSource.PHONE_PREFIX == "phone_prefix"
    assert LocationSource.NONE == "none"


def test_log_event_type_values_are_lowercase_strings() -> None:
    assert LogEventType.MESSAGE_RECEIVED == "message_received"
    assert LogEventType.MESSAGE_NORMALIZED == "message_normalized"
    assert LogEventType.AGENT_PROMPT_CREATED == "agent_prompt_created"
    assert LogEventType.RESPONSE_SENT == "response_sent"
    assert LogEventType.ERROR == "error"
