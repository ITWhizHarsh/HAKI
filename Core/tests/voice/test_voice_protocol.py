"""Focused unit tests for the strict v1 local voice socket protocol."""

from __future__ import annotations

import struct
from copy import deepcopy

import pytest

from core.ipc.voice_protocol import (
    EVENT_ACK,
    PCM_CHUNK,
    PLAYBACK_CANCELLED,
    PLAYBACK_CONFIRMED,
    TRANSCRIPT_EVENT,
    VoiceProtocolError,
    VoiceProtocolSession,
    decode_jsonl,
    decode_pcm_chunk,
    encode_jsonl,
    encode_pcm_chunk,
    protocol_fixture,
    validate_message,
)


def test_accepts_ordered_partial_then_final_transcript_events() -> None:
    """A turn permits partial text followed by one final text event."""
    partial = protocol_fixture("transcript_partial")
    final = protocol_fixture("transcript_final")
    session = VoiceProtocolSession(partial["session_id"])

    accepted_partial = session.accept_jsonl(encode_jsonl(partial))
    accepted_final = session.accept_jsonl(encode_jsonl(final))

    assert accepted_partial.message_type == TRANSCRIPT_EVENT
    assert accepted_partial.data["is_final"] is False
    assert accepted_final.data["is_final"] is True
    assert accepted_final.data["text"] == "Kal meeting reschedule kar do"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda message: message.update(unexpected="extension"), "unknown_field"),
        (lambda message: message.update(samples_b64="not-permitted"), "microphone_payload_forbidden"),
        (lambda message: message.update(turn_id="turn_7"), "malformed_id:turn_id"),
        (lambda message: message.update(language="fr"), "invalid_language"),
        (lambda message: message.update(is_final="false"), "invalid_finality"),
        (lambda message: message.update(version=2), "protocol_version_incompatible"),
    ],
)
def test_rejects_strict_schema_and_version_failures(mutate, reason: str) -> None:
    """Unknown/audio fields and malformed transcript values have stable reasons."""
    message = protocol_fixture("transcript_partial")
    mutate(message)

    with pytest.raises(VoiceProtocolError, match=reason) as error:
        validate_message(message)

    assert error.value.reason == reason


def test_rejects_out_of_order_duplicate_final_and_stale_session_events() -> None:
    """Turn ordering and session ownership cannot be bypassed after acceptance."""
    partial = protocol_fixture("transcript_partial")
    final = protocol_fixture("transcript_final")
    session = VoiceProtocolSession(partial["session_id"])
    session.accept(partial)

    out_of_order = deepcopy(final)
    out_of_order["event_seq"] += 1
    with pytest.raises(VoiceProtocolError, match="invalid_event_sequence"):
        session.accept(out_of_order)

    session.accept(final)
    duplicate_final = deepcopy(final)
    duplicate_final["event_id"] = "00000000-0000-4000-8000-000000000007"
    duplicate_final["event_seq"] += 1
    with pytest.raises(VoiceProtocolError, match="duplicate_final"):
        session.accept(duplicate_final)

    stale = protocol_fixture("transcript_partial")
    stale["session_id"] = "00000000-0000-4000-8000-000000000008"
    with pytest.raises(VoiceProtocolError, match="stale_session"):
        session.accept(stale)


def test_pcm_output_uses_exact_length_prefixed_binary_framing() -> None:
    """PCM output permits only exact metadata/prefix/payload length agreement."""
    metadata = protocol_fixture("pcm_chunk_metadata")
    payload = b"\x00\x00\x01\x00"

    frame = encode_pcm_chunk(metadata, payload)
    decoded_metadata, decoded_payload = decode_pcm_chunk(frame)

    assert decoded_metadata.message_type == PCM_CHUNK
    assert decoded_payload == payload

    header = encode_jsonl(metadata)
    mismatched_prefix = header + struct.pack(">I", len(payload) + 2) + payload
    with pytest.raises(VoiceProtocolError, match="invalid_binary_length"):
        decode_pcm_chunk(mismatched_prefix)

    wrong_metadata_length = deepcopy(metadata)
    wrong_metadata_length["byte_length"] = 2
    with pytest.raises(VoiceProtocolError, match="invalid_binary_length"):
        encode_pcm_chunk(wrong_metadata_length, payload)


def test_playback_terminal_events_are_exactly_once_per_sentence() -> None:
    """A confirmed, cancelled, or failed sentence cannot become terminal twice."""
    confirmed = protocol_fixture("playback_confirmed")
    session = VoiceProtocolSession(confirmed["session_id"])
    session.accept(confirmed)

    cancelled = {
        "version": 1,
        "type": PLAYBACK_CANCELLED,
        "event_id": "00000000-0000-4000-8000-000000000009",
        "session_id": confirmed["session_id"],
        "turn_id": confirmed["turn_id"],
        "sentence_id": confirmed["sentence_id"],
    }
    with pytest.raises(VoiceProtocolError, match="duplicate_playback_terminal"):
        session.accept(cancelled)


def test_ack_schema_allows_only_the_documented_fields() -> None:
    """Acknowledgements are strict JSONL records without an implicit extension bag."""
    ack = {
        "version": 1,
        "type": EVENT_ACK,
        "event_id": "00000000-0000-4000-8000-000000000003",
        "status": "discarded",
        "reason": "stale_session",
    }
    assert decode_jsonl(encode_jsonl(ack)).data == ack

    ack["audio"] = "prohibited"
    with pytest.raises(VoiceProtocolError, match="microphone_payload_forbidden"):
        validate_message(ack)


def test_rejects_pcm_metadata_with_microphone_payload_or_invalid_sample_bytes() -> None:
    """Output metadata is not a loophole for microphone content or malformed PCM."""
    metadata = protocol_fixture("pcm_chunk_metadata")
    metadata["pcm"] = "input-audio"
    with pytest.raises(VoiceProtocolError, match="microphone_payload_forbidden"):
        validate_message(metadata)

    malformed_length = protocol_fixture("pcm_chunk_metadata")
    malformed_length["byte_length"] = 3
    with pytest.raises(VoiceProtocolError, match="invalid_binary_length"):
        validate_message(malformed_length)


def test_rejects_unordered_pcm_chunk_sequences_in_a_session() -> None:
    """Renderer chunks for one sentence retain monotonic, gap-free ordering."""
    metadata = protocol_fixture("pcm_chunk_metadata")
    session = VoiceProtocolSession(metadata["session_id"])
    session.accept(metadata)

    skipped = deepcopy(metadata)
    skipped["sequence"] = 2
    with pytest.raises(VoiceProtocolError, match="invalid_pcm_sequence"):
        session.accept(skipped)

    next_chunk = deepcopy(metadata)
    next_chunk["sequence"] = 1
    assert session.accept(next_chunk).message_type == PCM_CHUNK


def test_terminal_confirmed_fixture_has_complete_strict_schema() -> None:
    """The embedded shared terminal fixture remains a valid contract example."""
    confirmed = protocol_fixture("playback_confirmed")
    assert confirmed["type"] == PLAYBACK_CONFIRMED
    assert validate_message(confirmed).as_dict() == confirmed
