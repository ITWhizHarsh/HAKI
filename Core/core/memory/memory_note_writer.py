"""
MemoryNoteWriter — writes Memory_Notes atomically to wiki/ and validates links.

Part of the HAKI Brain Memory Processing Pipeline. Writes one Memory_Note per
call using atomic temp-then-rename writes, validates all link targets before
writing, and updates ChromaDB after a successful write.

See: .kiro/specs/haki-brain-memory-processing-pipeline/design.md
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.fast_pass import Entity
    from core.memory.heavy_pass import HeavyPassResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MemoryNotePath:
    """Identifies a Memory_Note written to Wiki_Folder."""

    path: Path       # absolute filesystem path to the written .md file
    title: str       # note title (filename stem), e.g. "harsh-kumar_20250101_120000"
    wiki_link: str   # "[[concept_slug_YYYYMMDD_HHMMSS]]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert *text* to a safe, lowercase, hyphenated filename slug.

    - Lowercases and strips surrounding whitespace.
    - Removes any character that is not a word character, whitespace, or hyphen.
    - Collapses runs of whitespace/underscore into a single hyphen.
    - Truncates to 60 characters and strips leading/trailing hyphens.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60].strip("-")


def _dedupe_links(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Filter *links* — a list of `(link_type, target)` pairs in the order they
    would be written into a single Memory_Note — so that no `(link_type,
    target)` pair appears more than once (Req 11.3).

    The first occurrence of each `(link_type, target)` pair is kept; any
    later occurrence of the same pair is skipped and logged as a diagnostic.
    Links of the same target but a *different* type (e.g. a provenance link
    and an evolutionary link both pointing at the same note name) are not
    considered duplicates and are both kept.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for link_type, target in links:
        key = (link_type, target)
        if key in seen:
            logger.warning(
                "[MemoryNoteWriter] Skipping duplicate %s link to [[%s]] — "
                "an identical link already exists in this Memory_Note",
                link_type,
                target,
            )
            continue
        seen.add(key)
        deduped.append((link_type, target))
    return deduped


def validate_wiki_link(target: str, vault_root: Path) -> bool:
    """
    Return True if *target* (the note name without the surrounding `[[ ]]`)
    resolves to an existing `.md` file within the Vault.

    Search order (Req 11.2, 11.5):
      1. vault_root / wiki / target.md
      2. vault_root / target.md
    """
    for candidate in (
        vault_root / "wiki" / f"{target}.md",
        vault_root / f"{target}.md",
    ):
        if candidate.exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Memory_Note templates
# ---------------------------------------------------------------------------

_FAST_PASS_TEMPLATE = """---
title: "{title}"
created: "{iso8601}"
pass: "fast"
sources:
  - "[[{vault_rel}]]"
tags:
  - {entity_type_lower}
---

## {concept_slug}

- **Entity type:** {entity_label}
- **Extracted from:** [[{source_filename}]]
- **Date:** {run_date}

---
*Source: [[{vault_rel}]]*
"""


_HEAVY_PASS_TEMPLATE = """---
title: "{title}"
created: "{iso8601}"
pass: "heavy"
sources:
  - "[[{vault_rel}]]"
evolved_from: "{evolved_from}"
---

## {concept_slug}

{llm_content}

- **Extracted from:** [[{source_filename}]]
- **Date:** {run_date}

---
*Source: [[{vault_rel}]]*
{evo_link}"""


_HEAVY_PASS_TEMPLATE_NO_EVOLUTION = """---
title: "{title}"
created: "{iso8601}"
pass: "heavy"
sources:
  - "[[{vault_rel}]]"
---

## {concept_slug}

{llm_content}

- **Extracted from:** [[{source_filename}]]
- **Date:** {run_date}

---
*Source: [[{vault_rel}]]*
"""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class MemoryNoteWriter:
    """
    Writes one Memory_Note per call to Wiki_Folder using atomic temp-then-rename
    writes, validates all link targets before writing, and updates ChromaDB after
    a successful write.

    Parameters
    ----------
    vault_root : Path
        Absolute path to the Vault root (read from `HAKI_OBSIDIAN_VAULT`).
    chroma_collection : optional
        The `haki_wiki` ChromaDB collection used to embed newly written notes.
        May be `None` (e.g. in tests) — `_embed_note()` becomes a no-op.
    """

    def __init__(self, vault_root: Path, chroma_collection=None) -> None:
        self._vault = vault_root
        self._wiki = vault_root / "wiki"
        self._chroma = chroma_collection

    # ------------------------------------------------------------------
    # Fast Pass write
    # ------------------------------------------------------------------

    def write_fast_pass_note(
        self,
        entity: "Entity",
        source_filename: str,
        source_vault_rel: str,
        run_date: str,
    ) -> Optional[MemoryNotePath]:
        """
        Write one Memory_Note for a single entity extracted by the Fast Pass.

        Parameters
        ----------
        entity : Entity
            The extracted entity (text + label) this note documents.
        source_filename : str
            The Source_File's bare filename, used in the "Extracted from" line.
        source_vault_rel : str
            The Source_File's path relative to `vault_root`, used to build the
            Provenance_Link and validated to resolve within the Vault before
            any write is attempted (Req 3.1, 3.3, 3.4).
        run_date : str
            Processing run date in YYYY-MM-DD format.

        Returns
        -------
        MemoryNotePath on success, or None if the provenance link does not
        resolve (note creation is skipped and a diagnostic is logged) or if
        the write itself fails.
        """
        vault_rel = source_vault_rel.replace("\\", "/")

        # Provenance-link validation before write (Req 3.3, 3.4): the
        # Source_File path must resolve to an existing file within the Vault.
        if not (self._vault / Path(vault_rel)).exists():
            logger.warning(
                "[MemoryNoteWriter] Provenance target does not exist: %s — "
                "skipping Memory_Note creation for entity %r",
                vault_rel,
                entity.text,
            )
            return None

        slug = _slugify(entity.text)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = f"{slug}_{ts}"

        content = _FAST_PASS_TEMPLATE.format(
            title=title,
            iso8601=datetime.now(timezone.utc).isoformat(),
            vault_rel=vault_rel,
            entity_type_lower=entity.label.lower(),
            concept_slug=slug,
            entity_label=entity.label,
            source_filename=source_filename,
            run_date=run_date,
        )

        return self._write_atomic(title, content)

    # ------------------------------------------------------------------
    # Heavy Pass write
    # ------------------------------------------------------------------

    def write_heavy_pass_note(
        self,
        result: "HeavyPassResult",
        source_filename: str,
        source_vault_rel: str,
        run_date: str,
    ) -> Optional[MemoryNotePath]:
        """
        Write one Memory_Note produced by the Heavy Pass.

        Parameters
        ----------
        result : HeavyPassResult
            The Bonsai-8B synthesis result: `memory_content`, and optionally
            `old_memory_note_name` (candidate Evolutionary_Link target).
        source_filename : str
            The Source_File's bare filename, used in the "Extracted from" line.
        source_vault_rel : str
            The Source_File's path relative to `vault_root`, used to build the
            Provenance_Link and validated to resolve within the Vault before
            any write is attempted (Req 3.2, 3.3, 3.4).
        run_date : str
            Processing run date in YYYY-MM-DD format, used to annotate the
            Evolutionary_Link (Req 5.3).

        Returns
        -------
        MemoryNotePath on success, or None if the provenance link does not
        resolve (note creation is skipped and a diagnostic is logged) or if
        the write itself fails.

        Notes
        -----
        The Evolutionary_Link is validated independently of the provenance
        link (Req 5.4): if `result.old_memory_note_name` does not resolve to
        an existing note in Wiki_Folder, the Memory_Note is still written —
        just without the Evolutionary_Link — and a diagnostic is logged
        (Req 5.5 covers the "no old note at all" case, which arrives here as
        `old_memory_note_name is None`).
        """
        vault_rel = source_vault_rel.replace("\\", "/")

        # Provenance-link validation before write (Req 3.3, 3.4): the
        # Source_File path must resolve to an existing file within the Vault.
        if not (self._vault / Path(vault_rel)).exists():
            logger.warning(
                "[MemoryNoteWriter] Provenance target does not exist: %s — "
                "skipping Memory_Note creation for %r",
                vault_rel,
                source_filename,
            )
            return None

        slug = _slugify(source_filename.rsplit(".", 1)[0])
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = f"{slug}_{ts}"

        # Evolutionary-link validation before write (Req 5.1, 5.4, 5.5): only
        # include the link if the target note actually exists in the Vault.
        evo_target = result.old_memory_note_name
        evo_link_text = ""
        evolved_from = ""
        if evo_target:
            if validate_wiki_link(evo_target, self._vault):
                evolved_from = evo_target
            else:
                logger.warning(
                    "[MemoryNoteWriter] Evolutionary link target does not "
                    "exist: %s — skipping Evolutionary_Link for %r",
                    evo_target,
                    title,
                )

        # Duplicate-link detection (Req 11.3): a note must never contain two
        # links of the same type targeting the same destination note. Build
        # the full set of candidate links for this note and dedupe by
        # (link_type, target) before any of them are rendered into the
        # template. With today's single-provenance/single-evolutionary shape
        # this is a no-op, but it protects future multi-source notes (or a
        # provenance and evolutionary link that happen to collide) from ever
        # producing a duplicate link.
        candidate_links = [("provenance", vault_rel)]
        if evolved_from:
            candidate_links.append(("evolutionary", evolved_from))
        deduped_links = set(_dedupe_links(candidate_links))

        if evolved_from and ("evolutionary", evolved_from) not in deduped_links:
            evolved_from = ""
        if evolved_from:
            evo_link_text = f"*Evolved from: [[{evolved_from}]] ({run_date})*\n"

        template = _HEAVY_PASS_TEMPLATE if evolved_from else _HEAVY_PASS_TEMPLATE_NO_EVOLUTION

        format_kwargs = dict(
            title=title,
            iso8601=datetime.now(timezone.utc).isoformat(),
            vault_rel=vault_rel,
            concept_slug=slug,
            llm_content=result.memory_content,
            source_filename=source_filename,
            run_date=run_date,
        )
        if evolved_from:
            format_kwargs["evolved_from"] = evolved_from
            format_kwargs["evo_link"] = evo_link_text

        content = template.format(**format_kwargs)

        return self._write_atomic(title, content)

    # ------------------------------------------------------------------
    # Atomic write / embedding
    # ------------------------------------------------------------------

    def _write_atomic(self, title: str, content: str) -> Optional[MemoryNotePath]:
        """
        Write *content* to `wiki/{title}.md` using a temp-file + fsync +
        rename sequence so a Memory_Note is never left half-written (Req 10.2).

        The temp file is created in the same directory as the final target so
        that `Path.rename()` is an atomic filesystem operation (same
        filesystem, no partial-write window). On any failure — temp file
        creation, write, fsync, or rename — the temp file is removed if it
        exists, a diagnostic is logged, and `None` is returned; no
        previously existing Vault file is touched.
        """
        self._wiki.mkdir(parents=True, exist_ok=True)
        target = self._wiki / f"{title}.md"
        tmp_path: Optional[Path] = None
        try:
            fd, tmp_str = tempfile.mkstemp(dir=self._wiki, suffix=".tmp", prefix=title)
            tmp_path = Path(tmp_str)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.rename(target)
            tmp_path = None
        except OSError as exc:
            logger.error("[MemoryNoteWriter] Failed to write %s: %s", title, exc)
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None

        note_path = MemoryNotePath(
            path=target, title=title, wiki_link=f"[[{title}]]"
        )
        self._embed_note(title, content, target)
        logger.info("[MemoryNoteWriter] Wrote %s", target.name)
        return note_path

    def _embed_note(self, title: str, content: str, note_path: Path) -> None:
        """
        Upsert the newly written Memory_Note into the `haki_wiki` ChromaDB
        collection so it is immediately searchable by the Heavy Pass's
        semantic query step. No-op when no `chroma_collection` was injected
        (e.g. in tests). A ChromaDB failure is logged as a diagnostic but
        never raised — the write to `wiki/` has already succeeded and must
        not be rolled back for an embedding failure.
        """
        if self._chroma is None:
            return
        try:
            doc_id = hashlib.sha256(str(note_path).encode()).hexdigest()[:16]
            self._chroma.upsert(
                ids=[doc_id],
                documents=[content[:4_000]],
                metadatas=[
                    {
                        "title": title,
                        "wiki_path": str(note_path),
                        "created": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )
        except Exception as exc:
            logger.warning(
                "[MemoryNoteWriter] ChromaDB embed failed for %s: %s", title, exc
            )
