"""
FastPassExtractor — deterministic, LLM-free entity/fact extraction.

Part of the HAKI Brain Memory Processing Pipeline. The Fast Pass runs
before any LLM is invoked, applying spaCy NER and regex rules to extract
entities from a Source_File in milliseconds at zero GPU cost.

See: .kiro/specs/haki-brain-memory-processing-pipeline/design.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FastPassStatus(str, Enum):
    """Outcome of a Fast Pass extraction attempt."""

    SUCCESS = "success"        # >=1 entity extracted
    NO_ENTITIES = "no_entities"  # nothing found (trigger Heavy Pass)
    ERROR = "error"            # exception during extraction (trigger Heavy Pass)


@dataclass
class Entity:
    """A single extracted entity with its surface form, label, and offsets."""

    text: str    # surface form ("Harsh Kumar")
    label: str   # spaCy label ("PERSON") or regex category ("EMAIL", "URL", etc.)
    start: int   # char offset in content
    end: int     # char offset in content


@dataclass
class FastPassResult:
    """Result of running FastPassExtractor.extract() on a Source_File."""

    status: FastPassStatus
    entities: list[Entity] = field(default_factory=list)
    raw_markdown: str = ""              # pre-formatted note body per entity
    error_msg: Optional[str] = None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FastPassExtractor:
    """
    CPU-only entity extractor. Loaded once at HAKIBrain.init() and reused.

    Extraction order:
      1. spaCy en_core_web_sm — PERSON, ORG, GPE, DATE, EVENT, PRODUCT
      2. Regex patterns       — EMAIL, URL, PHONE, HAKI_MARKER
      3. Hindi/Hinglish       — regex fallback for common markers
         when spaCy yields nothing AND content appears non-English

    spaCy is loaded lazily on first call to `extract()` so that importing
    this module never fails even if spaCy or the en_core_web_sm model is
    unavailable — in that case the extractor silently falls back to
    regex-only mode.
    """

    _ENTITY_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT"}

    _REGEX_PATTERNS: dict[str, "re.Pattern[str]"] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
        "URL": re.compile(r"https?://[^\s]+"),
        "PHONE": re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"),
        "HAKI_MARKER": re.compile(r"#haki\:[a-zA-Z0-9_\-]+", re.IGNORECASE),
    }

    # Common Hindi/Hinglish entity markers (name, place, company)
    _HINDI_FALLBACK_RE = re.compile(
        r"\b(?:naam|नाम|jagah|जगह|company|सिंह|कुमार|sharma|verma)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._nlp = None  # lazy-loaded on first call; False means "unavailable"

    def _load_spacy(self) -> None:
        """Lazily load the spaCy en_core_web_sm model.

        Sets `self._nlp` to the loaded pipeline on success, or to `False`
        (sentinel for "unavailable") if spaCy or the model cannot be loaded.
        Safe to call multiple times — a no-op after the first call.
        """
        if self._nlp is not None:
            return
        try:
            import spacy  # type: ignore[import]

            self._nlp = spacy.load("en_core_web_sm")
            logger.info("[FastPassExtractor] spaCy en_core_web_sm loaded")
        except (ImportError, OSError) as exc:
            logger.warning(
                "[FastPassExtractor] spaCy unavailable: %s — regex-only mode", exc
            )
            self._nlp = False  # sentinel: spaCy unavailable, use regex only

    def extract(self, content: str, filename: str) -> FastPassResult:
        """
        Extract entities from *content*.

        Returns FastPassResult with status SUCCESS when >=1 unique entity is
        found, NO_ENTITIES when the content yields nothing, or ERROR (with a
        diagnostic message) if an exception is raised during extraction.
        """
        try:
            self._load_spacy()
            entities: list[Entity] = []

            # 1. spaCy NER (only if spaCy loaded successfully)
            if self._nlp:
                doc = self._nlp(content[:100_000])  # cap to avoid OOM on huge files
                for ent in doc.ents:
                    if ent.label_ in self._ENTITY_LABELS:
                        entities.append(
                            Entity(
                                text=ent.text,
                                label=ent.label_,
                                start=ent.start_char,
                                end=ent.end_char,
                            )
                        )

            # 2. Regex patterns (run regardless of spaCy result)
            for label, pattern in self._REGEX_PATTERNS.items():
                for m in pattern.finditer(content):
                    entities.append(
                        Entity(text=m.group(), label=label, start=m.start(), end=m.end())
                    )

            # 3. Hindi/Hinglish fallback: only if nothing found so far
            if not entities:
                m = self._HINDI_FALLBACK_RE.search(content)
                if m:
                    entities.append(
                        Entity(
                            text=m.group(),
                            label="HINGLISH_ENTITY",
                            start=m.start(),
                            end=m.end(),
                        )
                    )

            # Deduplicate by (text, label)
            seen: set[tuple[str, str]] = set()
            unique: list[Entity] = []
            for e in entities:
                key = (e.text.strip().lower(), e.label)
                if key not in seen:
                    seen.add(key)
                    unique.append(e)

            if not unique:
                return FastPassResult(status=FastPassStatus.NO_ENTITIES)

            return FastPassResult(
                status=FastPassStatus.SUCCESS,
                entities=unique,
                raw_markdown=_entities_to_markdown(unique, filename),
            )

        except Exception as exc:
            logger.error("[FastPassExtractor] Error extracting %s: %s", filename, exc)
            return FastPassResult(
                status=FastPassStatus.ERROR,
                error_msg=str(exc),
            )


def _entities_to_markdown(entities: list[Entity], filename: str) -> str:
    """Format extracted entities as a pre-formatted markdown note body."""
    lines = [f"- **{e.label}**: {e.text}" for e in entities]
    return f"Extracted from `{filename}`:\n\n" + "\n".join(lines)
