from dataclasses import fields
from datetime import date, datetime
import re

import pytest

from app.core.envelope import (
    CanonicalMessage,
    InputType,
    LocationContext,
    LocationSource,
    LogEventType,
    SubmissionType,
    _make_session_id_for_date,
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


def test_session_id_is_consistent_within_a_day() -> None:
    day = date(2026, 5, 2)

    first = _make_session_id_for_date("user-123", "sms", day)
    second = _make_session_id_for_date("user-123", "sms", day)

    assert first == second


def test_session_id_rotates_daily() -> None:
    first = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))
    second = _make_session_id_for_date("user-123", "sms", date(2026, 5, 3))

    assert first != second


def test_session_id_differs_by_channel() -> None:
    sms = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))
    whatsapp = _make_session_id_for_date("user-123", "whatsapp", date(2026, 5, 2))

    assert sms != whatsapp


def test_session_id_is_sha256_hex_digest() -> None:
    session_id = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))

    assert len(session_id) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", session_id)


def test_session_id_does_not_include_raw_user_id() -> None:
    raw_user_id = "user-123"

    session_id = _make_session_id_for_date(raw_user_id, "sms", date(2026, 5, 2))

    assert raw_user_id not in session_id


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


def test_canonical_message_defaults_to_unknown_submission_type() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    assert message.submission_type is SubmissionType.UNKNOWN


def test_canonical_message_session_context_does_not_share_mutable_state() -> None:
    first = CanonicalMessage(session_id="session-1", channel="sms", input_type=InputType.TEXT)
    second = CanonicalMessage(session_id="session-2", channel="sms", input_type=InputType.TEXT)

    first.session_context["key"] = "value"

    assert second.session_context == {}


def test_canonical_message_prior_context_does_not_share_mutable_state() -> None:
    first = CanonicalMessage(session_id="session-1", channel="sms", input_type=InputType.TEXT)
    second = CanonicalMessage(session_id="session-2", channel="sms", input_type=InputType.TEXT)

    first.prior_context.append({"role": "user"})

    assert second.prior_context == []


def test_canonical_message_timestamp_is_timezone_aware() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    assert message.timestamp.tzinfo is not None
    assert message.timestamp.utcoffset() is not None
    assert datetime.now(message.timestamp.tzinfo).utcoffset() == message.timestamp.utcoffset()


def test_agent_prompt_includes_text_content_for_text_messages() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        text_content="I need help with an application.",
    )

    prompt = message.to_agent_prompt()

    assert prompt["text_content"] == "I need help with an application."


def test_agent_prompt_excludes_session_id() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "session_id" not in prompt


def test_agent_prompt_excludes_channel() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "channel" not in prompt


def test_agent_prompt_excludes_media_url() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.IMAGE,
        media_url="https://example.test/image.jpg",
    )

    prompt = message.to_agent_prompt()

    assert "media_url" not in prompt


def test_agent_prompt_includes_safe_location_fields_when_location_context_exists() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        location_context=LocationContext(
            country_code="US",
            region="CA",
            city="Oakland",
            source=LocationSource.PROMPT,
            confidence=0.8,
        ),
    )

    prompt = message.to_agent_prompt()

    assert prompt["location_context"] == {
        "country_code": "US",
        "region": "CA",
        "city": "Oakland",
        "source": LocationSource.PROMPT,
        "confidence": 0.8,
    }


def test_agent_prompt_excludes_coordinate_like_keys() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        location_context=LocationContext(
            country_code="US",
            source=LocationSource.PROMPT,
            confidence=0.9,
        ),
        session_context={"lat": 37.8, "safe": "value"},
    )

    prompt = message.to_agent_prompt()
    forbidden = {"coordinates", "latitude", "longitude", "lat", "lng", "gps"}

    assert set(prompt).isdisjoint(forbidden)
    assert set(prompt["location_context"]).isdisjoint(forbidden)
    assert "session_context" not in prompt


def test_agent_prompt_handles_missing_location_context_cleanly() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "location_context" not in prompt


def test_agent_prompt_does_not_mutate_original_message() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        session_context={"eligible_programs": ["housing"]},
    )

    prompt = message.to_agent_prompt()
    prompt["session_context"]["eligible_programs"].append("food")

    assert message.session_context == {"eligible_programs": ["housing"]}
