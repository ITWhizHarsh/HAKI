"""
Unit and property-based tests for the Note model and NoteSerializer.

Feature: haki-personal-ai-assistant
Requirements: 7.8

Test catalogue
--------------
Unit tests:
  1.  Serialize produces a well-structured Markdown string with YAML front matter
  2.  Deserialize reconstructs all fields from the serialized form
  3.  Round-trip: deserialize(serialize(note)) == note
  4.  Optional fields (superseded_by, learned_session) serialise as null / deserialise to None
  5.  Tags and topics round-trip as lists
  6.  All three NoteSource values serialise and deserialise correctly
  7.  private=True / private=False both round-trip
  8.  Missing required fields raise NoteSerializationError
  9.  Malformed YAML raises NoteSerializationError
 10.  Missing opening delimiter raises NoteSerializationError
 11.  Missing closing delimiter raises NoteSerializationError
 12.  Invalid source value raises NoteSerializationError
 13.  Body with multiple lines and trailing newlines round-trips cleanly
 14.  Note model defaults are sane

Property-based tests:
 21. Note serialization round trip (Property 21, Hypothesis)
     Validates: Requirements 7.8
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.memory import Chunk, Note, NoteSerializationError, NoteSerializer, NoteSource


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def serializer() -> NoteSerializer:
    return NoteSerializer()


def _make_note(**kwargs) -> Note:
    """Convenience factory with sensible UTC timestamps."""
    defaults = dict(
        id="2024-06-01T12-03-22-a1b2",
        created=datetime(2024, 6, 1, 12, 3, 22, tzinfo=timezone.utc),
        updated=datetime(2024, 6, 1, 12, 3, 22, tzinfo=timezone.utc),
        source=NoteSource.USER_STATED,
        tags=["exam", "networks"],
        topics=["computer-networks", "midterm"],
        superseded_by=None,
        private=False,
        learned_session=None,
        body="Midterm for Computer Networks is on June 14.",
    )
    defaults.update(kwargs)
    return Note(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Serialize structure
# ---------------------------------------------------------------------------


def test_serialize_starts_with_front_matter_delimiter(serializer: NoteSerializer) -> None:
    note = _make_note()
    result = serializer.serialize(note)
    assert result.startswith("---\n"), "Serialized note must begin with '---'"


def test_serialize_contains_all_required_fields(serializer: NoteSerializer) -> None:
    note = _make_note()
    result = serializer.serialize(note)
    for field in ("id:", "created:", "updated:", "source:", "tags:", "topics:",
                  "superseded_by:", "private:", "learned_session:"):
        assert field in result, f"Serialized note must contain field '{field}'"


def test_serialize_body_appears_after_closing_delimiter(serializer: NoteSerializer) -> None:
    body = "Midterm for Computer Networks is on June 14."
    note = _make_note(body=body)
    result = serializer.serialize(note)
    # Body must appear after the second '---'
    _, _, after_close = result.partition("\n---\n")
    assert body in after_close, "Note body must appear after the closing '---'"


# ---------------------------------------------------------------------------
# Test 2: Deserialize reconstructs fields
# ---------------------------------------------------------------------------


def test_deserialize_reconstructs_id(serializer: NoteSerializer) -> None:
    note = _make_note()
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.id == note.id


def test_deserialize_reconstructs_timestamps(serializer: NoteSerializer) -> None:
    note = _make_note()
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.created == note.created
    assert parsed.updated == note.updated


def test_deserialize_reconstructs_source(serializer: NoteSerializer) -> None:
    note = _make_note(source=NoteSource.LEARNED)
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.source == NoteSource.LEARNED


def test_deserialize_reconstructs_body(serializer: NoteSerializer) -> None:
    body = "Midterm for Computer Networks is on June 14."
    note = _make_note(body=body)
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.body == body


# ---------------------------------------------------------------------------
# Test 3: Full round-trip equality
# ---------------------------------------------------------------------------


def test_round_trip_full_note(serializer: NoteSerializer) -> None:
    note = _make_note(
        tags=["exam", "networks"],
        topics=["computer-networks"],
        private=False,
        learned_session=None,
        superseded_by=None,
    )
    assert serializer.deserialize(serializer.serialize(note)) == note


def test_round_trip_learned_note(serializer: NoteSerializer) -> None:
    note = _make_note(
        source=NoteSource.LEARNED,
        learned_session="2024-06-01T12-00",
    )
    assert serializer.deserialize(serializer.serialize(note)) == note


def test_round_trip_private_note(serializer: NoteSerializer) -> None:
    note = _make_note(private=True)
    assert serializer.deserialize(serializer.serialize(note)) == note


def test_round_trip_superseded_note(serializer: NoteSerializer) -> None:
    note = _make_note(superseded_by="2024-06-02T09-00-00-c3d4")
    assert serializer.deserialize(serializer.serialize(note)) == note


# ---------------------------------------------------------------------------
# Test 4: Optional fields serialise as null / deserialise to None
# ---------------------------------------------------------------------------


def test_superseded_by_null_serializes_as_null(serializer: NoteSerializer) -> None:
    note = _make_note(superseded_by=None)
    result = serializer.serialize(note)
    assert "superseded_by: null" in result


def test_learned_session_null_serializes_as_null(serializer: NoteSerializer) -> None:
    note = _make_note(learned_session=None)
    result = serializer.serialize(note)
    assert "learned_session: null" in result


def test_superseded_by_none_deserializes_to_none(serializer: NoteSerializer) -> None:
    note = _make_note(superseded_by=None)
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.superseded_by is None


def test_learned_session_none_deserializes_to_none(serializer: NoteSerializer) -> None:
    note = _make_note(learned_session=None)
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.learned_session is None


# ---------------------------------------------------------------------------
# Test 5: Tags and topics round-trip
# ---------------------------------------------------------------------------


def test_tags_empty_list_round_trips(serializer: NoteSerializer) -> None:
    note = _make_note(tags=[])
    assert serializer.deserialize(serializer.serialize(note)).tags == []


def test_topics_multiple_values_round_trip(serializer: NoteSerializer) -> None:
    note = _make_note(topics=["rag", "vector-search", "memory"])
    parsed = serializer.deserialize(serializer.serialize(note))
    assert parsed.topics == ["rag", "vector-search", "memory"]


# ---------------------------------------------------------------------------
# Test 6: All NoteSource values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", list(NoteSource))
def test_all_note_source_values_round_trip(serializer: NoteSerializer, source: NoteSource) -> None:
    note = _make_note(source=source)
    assert serializer.deserialize(serializer.serialize(note)).source == source


# ---------------------------------------------------------------------------
# Test 7: private flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("private", [True, False])
def test_private_flag_round_trips(serializer: NoteSerializer, private: bool) -> None:
    note = _make_note(private=private)
    assert serializer.deserialize(serializer.serialize(note)).private == private


# ---------------------------------------------------------------------------
# Test 8: Missing required fields raise NoteSerializationError
# ---------------------------------------------------------------------------


def test_missing_id_raises(serializer: NoteSerializer) -> None:
    md = (
        "---\n"
        "created: 2024-06-01T12:03:22Z\n"
        "updated: 2024-06-01T12:03:22Z\n"
        "source: user_stated\n"
        "---\n\nBody.\n"
    )
    with pytest.raises(NoteSerializationError, match="id"):
        serializer.deserialize(md)


def test_missing_source_raises(serializer: NoteSerializer) -> None:
    md = (
        "---\n"
        "id: abc\n"
        "created: 2024-06-01T12:03:22Z\n"
        "updated: 2024-06-01T12:03:22Z\n"
        "---\n\nBody.\n"
    )
    with pytest.raises(NoteSerializationError, match="source"):
        serializer.deserialize(md)


# ---------------------------------------------------------------------------
# Test 9: Malformed YAML
# ---------------------------------------------------------------------------


def test_malformed_yaml_raises(serializer: NoteSerializer) -> None:
    md = "---\n: bad: yaml: [\n---\n\nBody.\n"
    with pytest.raises(NoteSerializationError):
        serializer.deserialize(md)


# ---------------------------------------------------------------------------
# Test 10 & 11: Missing delimiters
# ---------------------------------------------------------------------------


def test_missing_opening_delimiter_raises(serializer: NoteSerializer) -> None:
    md = "id: abc\ncreated: 2024-06-01T12:03:22Z\n"
    with pytest.raises(NoteSerializationError, match="---"):
        serializer.deserialize(md)


def test_missing_closing_delimiter_raises(serializer: NoteSerializer) -> None:
    md = "---\nid: abc\ncreated: 2024-06-01T12:03:22Z\n"
    with pytest.raises(NoteSerializationError, match="---"):
        serializer.deserialize(md)


# ---------------------------------------------------------------------------
# Test 12: Invalid source value
# ---------------------------------------------------------------------------


def test_invalid_source_value_raises(serializer: NoteSerializer) -> None:
    md = (
        "---\n"
        "id: abc\n"
        "created: 2024-06-01T12:03:22Z\n"
        "updated: 2024-06-01T12:03:22Z\n"
        "source: alien_invented\n"
        "---\n\nBody.\n"
    )
    with pytest.raises(NoteSerializationError, match="source"):
        serializer.deserialize(md)


# ---------------------------------------------------------------------------
# Test 13: Multi-line body
# ---------------------------------------------------------------------------


def test_multiline_body_round_trips(serializer: NoteSerializer) -> None:
    body = "Line one.\nLine two.\nLine three."
    note = _make_note(body=body)
    assert serializer.deserialize(serializer.serialize(note)).body == body


def test_body_with_internal_yaml_like_content_round_trips(serializer: NoteSerializer) -> None:
    """Body text that looks like YAML must not be interpreted as front matter."""
    body = "key: value\n- item1\n- item2"
    note = _make_note(body=body)
    assert serializer.deserialize(serializer.serialize(note)).body == body


# ---------------------------------------------------------------------------
# Test 14: Note model defaults
# ---------------------------------------------------------------------------


def test_note_default_source_is_user_stated() -> None:
    note = Note()
    assert note.source == NoteSource.USER_STATED


def test_note_default_private_is_false() -> None:
    note = Note()
    assert note.private is False


def test_note_default_superseded_by_is_none() -> None:
    note = Note()
    assert note.superseded_by is None


def test_note_default_learned_session_is_none() -> None:
    note = Note()
    assert note.learned_session is None


def test_note_default_tags_and_topics_are_empty() -> None:
    note = Note()
    assert note.tags == []
    assert note.topics == []


def test_chunk_fields() -> None:
    chunk = Chunk(note_id="abc", chunk_index=0, text="hello")
    assert chunk.note_id == "abc"
    assert chunk.chunk_index == 0
    assert chunk.text == "hello"
    assert chunk.embedding is None


# ---------------------------------------------------------------------------
# Property-based test 21: Note serialization round trip (Hypothesis)
#
# Feature: haki-personal-ai-assistant
# Property 21: Note serialization round trip
# Validates: Requirements 7.8
# ---------------------------------------------------------------------------

_NOTE_ID_STRATEGY = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-T:"),
    min_size=1,
    max_size=40,
)

_DATETIME_STRATEGY = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2099, 12, 31),
    timezones=st.just(timezone.utc),
)

_TAG_STRATEGY = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=20,
    ),
    min_size=0,
    max_size=5,
)

_BODY_STRATEGY = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),   # exclude surrogates
        blacklist_characters="\x00",    # exclude null bytes
    ),
    min_size=0,
    max_size=500,
)

_NOTE_SOURCE_STRATEGY = st.sampled_from(list(NoteSource))

_LEARNED_SESSION_STRATEGY = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-T:"),
        min_size=1,
        max_size=20,
    ),
)


@given(
    note_id=_NOTE_ID_STRATEGY,
    created=_DATETIME_STRATEGY,
    updated=_DATETIME_STRATEGY,
    source=_NOTE_SOURCE_STRATEGY,
    tags=_TAG_STRATEGY,
    topics=_TAG_STRATEGY,
    private=st.booleans(),
    learned_session=_LEARNED_SESSION_STRATEGY,
    body=_BODY_STRATEGY,
)
@settings(max_examples=100)
def test_property_note_serialization_round_trip(
    note_id: str,
    created: datetime,
    updated: datetime,
    source: NoteSource,
    tags: list[str],
    topics: list[str],
    private: bool,
    learned_session: str | None,
    body: str,
) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 21: Note serialization round trip

    For any valid Note, serializing and then deserializing must produce an
    identical Note.  This guarantees that no information is lost when notes
    are written to the Obsidian vault and read back.

    Validates: Requirements 7.8
    """
    # learned_session is only meaningful when source=LEARNED, but the model
    # accepts it regardless — test round-trip for all combinations.
    note = Note(
        id=note_id,
        created=created,
        updated=updated,
        source=source,
        tags=tags,
        topics=topics,
        superseded_by=None,
        private=private,
        learned_session=learned_session,
        body=body,
    )
    s = NoteSerializer()
    reconstructed = s.deserialize(s.serialize(note))
    assert reconstructed == note, (
        f"Round-trip failed.\nOriginal:     {note!r}\nReconstructed:{reconstructed!r}"
    )
