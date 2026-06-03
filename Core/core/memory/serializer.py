"""
Obsidian-compatible Markdown serializer and parser for Note objects.

A note is stored as a plain Markdown file with a YAML front-matter block
delimited by ``---`` lines, followed by the note body.  The format is
intentionally human-readable and compatible with Obsidian and any other
standard Markdown tool.

Example on-disk format (Req 7.8):

    ---
    id: 2024-06-01T12-03-22-a1b2
    created: 2024-06-01T12:03:22Z
    updated: 2024-06-01T12:03:22Z
    source: user_stated
    tags: [exam, networks]
    topics: [computer-networks, midterm]
    superseded_by: null
    private: false
    learned_session: null
    ---
    Midterm for Computer Networks is on June 14.

Design: Data Models (Note), Memory, RAG & Learning.
Requirements: 7.8.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml

from .models import Note, NoteSource

# Sentinel used in YAML to represent Python None
_YAML_NULL = "null"


class NoteSerializationError(ValueError):
    """Raised when a Markdown string cannot be parsed into a Note."""


class NoteSerializer:
    """
    Converts between ``Note`` objects and Obsidian-compatible Markdown strings.

    The serializer is a strict round-trip:

        deserialize(serialize(note)) == note

    Both ``serialize`` and ``deserialize`` are stateless; a single shared
    instance or multiple instances are interchangeable.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serialize(self, note: Note) -> str:
        """
        Convert a ``Note`` to an Obsidian-compatible Markdown string.

        The result starts with ``---``, followed by the YAML front matter
        block, another ``---``, a blank line, and then the note body.

        Parameters
        ----------
        note:
            The ``Note`` instance to serialize.

        Returns
        -------
        str
            A complete Markdown string ready to be written to disk.
        """
        front_matter = self._note_to_front_matter(note)
        # PyYAML's dump adds a trailing newline, strip it for clean output.
        yaml_str = yaml.dump(
            front_matter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip("\n")

        # Reconstruct body: if it has trailing newlines strip them to keep
        # the file clean, but always separate body from front matter with
        # exactly one blank line.
        body = note.body.rstrip("\n")
        return f"---\n{yaml_str}\n---\n\n{body}"

    def deserialize(self, markdown: str) -> Note:
        """
        Parse an Obsidian-compatible Markdown string back into a ``Note``.

        Parameters
        ----------
        markdown:
            A string containing a ``---``-delimited YAML front matter block
            followed by the note body.

        Returns
        -------
        Note
            A fully populated ``Note`` instance.

        Raises
        ------
        NoteSerializationError
            If the string is missing the front-matter delimiters, if the YAML
            is malformed, or if any required field is absent or has the wrong
            type.
        """
        fm_dict, body = self._split_markdown(markdown)
        return self._front_matter_to_note(fm_dict, body)

    # ------------------------------------------------------------------
    # Private helpers — serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """Format a UTC datetime as an ISO-8601 string, preserving microseconds."""
        # Always output UTC; include microseconds when they are non-zero so the
        # round-trip is lossless.
        base = dt.strftime("%Y-%m-%dT%H:%M:%S")
        if dt.microsecond:
            base += f".{dt.microsecond:06d}"
        return base + "Z"

    @staticmethod
    def _note_to_front_matter(note: Note) -> dict[str, Any]:
        """Build the YAML front-matter dictionary from a Note."""
        return {
            "id": note.id,
            "created": NoteSerializer._format_datetime(note.created),
            "updated": NoteSerializer._format_datetime(note.updated),
            "source": note.source.value,
            "tags": list(note.tags),
            "topics": list(note.topics),
            "superseded_by": note.superseded_by,   # None serializes as "null"
            "private": note.private,
            "learned_session": note.learned_session,  # None serializes as "null"
        }

    # ------------------------------------------------------------------
    # Private helpers — deserialization
    # ------------------------------------------------------------------

    @staticmethod
    def _split_markdown(markdown: str) -> tuple[dict[str, Any], str]:
        """
        Split a Markdown string into (front_matter_dict, body_str).

        Raises NoteSerializationError on structural problems.
        """
        text = markdown.lstrip("\n")

        if not text.startswith("---"):
            raise NoteSerializationError(
                "Markdown note must begin with a '---' front-matter delimiter."
            )

        # Find the closing delimiter.  We skip the first character to avoid
        # matching the opening delimiter itself.
        lines = text.splitlines(keepends=True)
        closing_idx: int | None = None
        for i, line in enumerate(lines[1:], start=1):
            if line.rstrip("\n\r") == "---":
                closing_idx = i
                break

        if closing_idx is None:
            raise NoteSerializationError(
                "Markdown note is missing the closing '---' front-matter delimiter."
            )

        yaml_str = "".join(lines[1:closing_idx])
        # Body starts after the closing delimiter line (and an optional blank line).
        body_lines = lines[closing_idx + 1 :]
        # Drop exactly one leading blank line if present (standard Obsidian format).
        if body_lines and body_lines[0].strip() == "":
            body_lines = body_lines[1:]
        body = "".join(body_lines).rstrip("\n")

        try:
            fm_dict = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:
            raise NoteSerializationError(
                f"Malformed YAML in note front matter: {exc}"
            ) from exc

        if not isinstance(fm_dict, dict):
            raise NoteSerializationError(
                f"Front matter must be a YAML mapping, got {type(fm_dict).__name__}."
            )

        return fm_dict, body

    @staticmethod
    def _require(fm: dict[str, Any], key: str) -> Any:
        """Return fm[key] or raise NoteSerializationError."""
        if key not in fm:
            raise NoteSerializationError(
                f"Required field '{key}' is missing from the note front matter."
            )
        return fm[key]

    @classmethod
    def _front_matter_to_note(cls, fm: dict[str, Any], body: str) -> Note:
        """Construct a Note from a parsed front-matter dict and body text."""
        # id
        note_id = str(cls._require(fm, "id"))

        # timestamps — accept ISO-8601 strings or Python datetime objects
        # (yaml.safe_load may return datetime objects for bare timestamps)
        created = cls._parse_datetime(cls._require(fm, "created"), "created")
        updated = cls._parse_datetime(cls._require(fm, "updated"), "updated")

        # source
        raw_source = cls._require(fm, "source")
        try:
            source = NoteSource(raw_source)
        except ValueError:
            valid = [s.value for s in NoteSource]
            raise NoteSerializationError(
                f"Invalid source value '{raw_source}'. Must be one of {valid}."
            )

        # tags / topics — accept None (yaml null) or a list
        tags = cls._parse_string_list(fm.get("tags"), "tags")
        topics = cls._parse_string_list(fm.get("topics"), "topics")

        # superseded_by — null/None means not superseded
        raw_sup = fm.get("superseded_by")
        superseded_by: str | None = None if (raw_sup is None or raw_sup == _YAML_NULL) else str(raw_sup)

        # private — accept bool or string "true"/"false"
        raw_private = fm.get("private", False)
        private = cls._parse_bool(raw_private, "private")

        # learned_session — null/None means not a learned note
        raw_ls = fm.get("learned_session")
        learned_session: str | None = None if (raw_ls is None or raw_ls == _YAML_NULL) else str(raw_ls)

        return Note(
            id=note_id,
            created=created,
            updated=updated,
            source=source,
            tags=tags,
            topics=topics,
            superseded_by=superseded_by,
            private=private,
            learned_session=learned_session,
            body=body,
        )

    # ------------------------------------------------------------------
    # Type-coercion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_datetime(value: Any, field_name: str) -> datetime:
        """Parse an ISO-8601 string or a yaml datetime object into a UTC datetime."""
        if isinstance(value, datetime):
            # yaml.safe_load may parse bare timestamps into aware/naive datetimes.
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            # Support both 'Z' suffix and '+00:00' offset.
            normalized = value.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise NoteSerializationError(
                    f"Field '{field_name}' has an invalid datetime value '{value}': {exc}"
                ) from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        raise NoteSerializationError(
            f"Field '{field_name}' must be an ISO-8601 string, got {type(value).__name__}."
        )

    @staticmethod
    def _parse_string_list(value: Any, field_name: str) -> list[str]:
        """Return a list[str] from a YAML list, None, or empty value."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        raise NoteSerializationError(
            f"Field '{field_name}' must be a YAML list or null, "
            f"got {type(value).__name__}."
        )

    @staticmethod
    def _parse_bool(value: Any, field_name: str) -> bool:
        """Coerce a boolean or string representation to a Python bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
        raise NoteSerializationError(
            f"Field '{field_name}' must be a boolean (true/false), "
            f"got '{value}'."
        )
