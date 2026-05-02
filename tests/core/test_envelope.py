from dataclasses import fields

import pytest

from app.core.envelope import (
    InputType,
    LocationContext,
    LocationSource,
    LogEventType,
    SubmissionType,
)


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


def test_location_context_defaults() -> None:
    context = LocationContext()

    assert context.country_code is None
    assert context.region is None
    assert context.city is None
    assert context.source is LocationSource.NONE
    assert context.confidence == 0.0


def test_location_context_accepts_valid_confidence() -> None:
    context = LocationContext(
        country_code="US",
        region="CA",
        city="San Francisco",
        source=LocationSource.DEVICE,
        confidence=0.75,
    )

    assert context.confidence == 0.75


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_location_context_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0 inclusive"):
        LocationContext(confidence=confidence)


def test_location_context_has_no_coordinate_fields() -> None:
    field_names = {field.name for field in fields(LocationContext)}

    forbidden = {"lat", "lng", "latitude", "longitude", "coordinates", "gps", "raw_location"}
    assert field_names.isdisjoint(forbidden)


def test_location_context_safe_dict_excludes_coordinate_keys() -> None:
    payload = LocationContext(source=LocationSource.PROMPT, confidence=1.0).to_safe_dict()

    assert payload == {
        "country_code": None,
        "region": None,
        "city": None,
        "source": LocationSource.PROMPT,
        "confidence": 1.0,
    }
    forbidden = {"lat", "lng", "latitude", "longitude", "coordinates", "gps", "raw_location"}
    assert set(payload).isdisjoint(forbidden)
