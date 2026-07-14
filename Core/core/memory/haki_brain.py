"""
HAKI Brain — Modified LLM Wiki with 3-Folder Obsidian Pipeline.

Based on Andrej Karpathy's LLM Wiki pattern
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
but extended with a strict 3-folder pipeline and bidirectional traceability.

Folder Structure (inside the Obsidian vault root / HAKI_Brain/):
    raw/        — Inbox. Drop unprocessed files, PDFs, images, chat logs here.
    processed/  — Archive. Files moved here after HAKI ingests them from raw/.
    wiki/       — The Brain. HAKI writes/maintains clean Markdown pages here.

Ingestion Rules:
    1. Watch raw/ for new files.
    2. For each file: read → synthesise → write wiki/ Markdown.
    3. Immediately MOVE the source file from raw/ → processed/  (never delete).
    4. Every wiki/ page MUST contain a [[processed/filename]] wikilink in a
       "Sources" YAML frontmatter block — the Cross-Reference Rule.

Embeddings:
    - All wiki/ pages are embedded with paraphrase-multilingual-MiniLM-L12-v2
      and stored in ChromaDB (persistent local SSD).
    - Raw text is never sent to an API unless the user explicitly enables
      API embeddings for large documents.

Usage:
    brain = HAKIBrain(
        obsidian_vault_path=Path("/Users/you/Obsidian/HAKI_Brain"),
        llm_router=router,
        embeddings_engine=embeddings,
    )
    await brain.ingest_pending()        # process all files in raw/
    results = await brain.search("photosynthesis notes")
    await brain.remember_fact("My exam is on June 20")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.model_provider.embeddings_engine import EmbeddingsEngine
    from core.model_provider.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_DIR = "raw"
_PROCESSED_DIR = "processed"
_WIKI_DIR = "wiki"
_CONVERSATIONS_DIR = "conversations"

_SUPPORTED_TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".java", ".c", ".cpp",
    ".rs", ".go", ".json", ".yaml", ".yml", ".toml", ".csv",
    ".html", ".htm", ".xml", ".tex",
}
_SUPPORTED_PDF = {".pdf"}
_SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

_WIKI_COLLECTION_NAME = "haki_wiki"
_CHROMA_BATCH_SIZE = 50


def _read_float_env(key: str, default: float) -> float:
    """Read *key* from the environment as a float, falling back to *default*.

    Any missing, empty, or unparseable value silently falls back to
    *default* (with a warning logged) rather than raising — pipeline
    configuration must never prevent HAKIBrain from starting up.
    """
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "[HAKIBrain] Invalid value for %s=%r — using default %s", key, raw, default
        )
        return default

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WikiPage:
    """A page written to the wiki/ folder."""

    title: str                      # Filename stem (slugified)
    source_file: str                # Original filename in raw/ before move
    content: str                    # Full Markdown content
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_markdown(self) -> str:
        """Render the page as Obsidian-compatible Markdown with YAML frontmatter."""
        tags_yaml = "\n".join(f"  - {t}" for t in self.tags) if self.tags else "  []"
        frontmatter = f"""\
---
title: "{self.title}"
created: "{self.created_at}"
sources:
  - "[[processed/{self.source_file}]]"
tags:
{tags_yaml}
---

"""
        return frontmatter + self.content


@dataclass
class IngestionResult:
    """Result of processing one file from raw/."""

    source_file: str
    wiki_page_path: str | None = None
    success: bool = True
    error: str | None = None
    pass_used: str | None = None  # "fast" | "heavy" | None (no pass succeeded)


# ---------------------------------------------------------------------------
# LLM prompts for wiki synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are HAKI's knowledge engine. Your job is to synthesize raw content into
clean, well-structured Markdown wiki pages for an Obsidian knowledge base.

Rules:
1. Extract all key facts, concepts, entities, and relationships.
2. Write in a clear, concise style — bullet points for lists, ## headings for sections.
3. Preserve all important details; do not hallucinate or add information.
4. Generate 3-7 relevant tags at the end (hashtag format).
5. Output ONLY the Markdown body (no frontmatter — that is added separately).
6. The page should be self-contained and linkable.
"""

_SYNTHESIS_PROMPT = """\
Synthesize the following content into a clean Obsidian wiki page.

Source file: {filename}
Content type: {content_type}

---CONTENT START---
{content}
---CONTENT END---

Write a clean, structured Markdown wiki page about this content.
End your response with a line: TAGS: tag1, tag2, tag3
"""

_FACT_SYSTEM = """\
You are HAKI's memory assistant. Convert the user's statement into a concise,
searchable Markdown note with key facts extracted as bullet points.
Output ONLY the Markdown body, no frontmatter.
End with: TAGS: tag1, tag2
"""


# ---------------------------------------------------------------------------
# HAKI Brain
# ---------------------------------------------------------------------------


class HAKIBrain:
    """
    HAKI's persistent knowledge store — the LLM Wiki pattern for Obsidian.

    Manages the 3-folder pipeline (raw/ → processed/ + wiki/) and provides
    semantic search over all wiki pages via ChromaDB + multilingual embeddings.

    Parameters
    ----------
    obsidian_vault_path:
        Path to the HAKI_Brain folder inside your Obsidian vault.
        e.g. Path("/Users/you/Obsidian/HAKI_Brain")
    llm_router:
        The LLMRouter instance for wiki synthesis.
    embeddings_engine:
        The EmbeddingsEngine instance for semantic search.
    auto_watch_interval:
        Seconds between automatic checks of the raw/ folder for new files.
        Set to 0 to disable automatic watching.
    """

    def __init__(
        self,
        obsidian_vault_path: Path,
        llm_router: Optional["LLMRouter"] = None,
        embeddings_engine: Optional["EmbeddingsEngine"] = None,
        auto_watch_interval: float = 30.0,
    ) -> None:
        self._vault = obsidian_vault_path
        self._raw = obsidian_vault_path / _RAW_DIR
        self._processed = obsidian_vault_path / _PROCESSED_DIR
        self._wiki = obsidian_vault_path / _WIKI_DIR
        self._conversations = obsidian_vault_path / _CONVERSATIONS_DIR
        self._llm = llm_router
        self._embeddings = embeddings_engine
        self._watch_interval = auto_watch_interval

        # ChromaDB collection (lazy-initialised)
        self._chroma_collection = None

        # Dedicated single-thread executor for all ChromaDB/embedding calls.
        # sentence-transformers uses PyTorch which is NOT thread-safe when
        # called concurrently from different threads.  Serialising all DB ops
        # on one thread prevents deadlocks with TTS/STT thread pools.
        import concurrent.futures
        self._chroma_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="haki_chroma"
        )

        # Track which raw files are currently being processed (avoid re-entry)
        self._in_progress: set[str] = set()

        # Background watcher task
        self._watcher_task: asyncio.Task | None = None

        # --- Hybrid pipeline (Fast Pass / Heavy Pass) infrastructure ---
        from core.memory.fast_pass import FastPassExtractor
        from core.memory.process_tracker import ProcessTracker
        from core.memory.memory_note_writer import MemoryNoteWriter

        self._fast_pass = FastPassExtractor()
        self._process_tracker = ProcessTracker(
            db_path=obsidian_vault_path / ".haki" / "process_tracker.db"
        )
        # MemoryNoteWriter is (re)constructed in init(), once the ChromaDB
        # collection is ready, so it can be handed a real chroma_collection
        # instead of the None that's available at __init__ time. A first
        # instance is created here (chroma_collection=None) so callers that
        # skip init() in tests still get a usable, if embedding-less, writer.
        self._note_writer = MemoryNoteWriter(
            vault_root=obsidian_vault_path,
            chroma_collection=self._chroma_collection,
        )

        self._llm_timeout = _read_float_env(
            "HAKI_PIPELINE_LLM_TIMEOUT_SECS", default=30.0
        )
        self._low_memory_mb = _read_float_env(
            "HAKI_PIPELINE_LOW_MEMORY_THRESHOLD_MB", default=500.0
        )

        # Vault validity flag — set by init() after startup validation.
        # When False, the PipelineScheduler must not start and no Vault
        # modifications should be performed (Req 8.1, 8.2, 8.3, 8.5).
        self._vault_valid: bool = False

        # Live-turn deferral mechanism (Req 9.5, 9.8).
        # The event is *set* when no live turn is active (pipeline can proceed).
        # The event is *cleared* when a live turn is in progress (pipeline must wait).
        self._live_turn_event: asyncio.Event = asyncio.Event()
        self._live_turn_event.set()  # No live turn initially — pipeline can proceed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Validate the vault path and create the 3-folder structure if valid.
        Must be called before any other method.

        Sets ``self._vault_valid`` to ``True`` only when all startup checks
        pass.  When the flag is ``False``, callers (e.g. PipelineScheduler)
        MUST NOT start background jobs or perform any Vault modifications.
        """
        # ----- Vault path startup validation (Req 8.1, 8.2, 8.3, 8.5) -----
        vault_env = os.environ.get("HAKI_OBSIDIAN_VAULT")

        # Check 1: env var is set and non-empty
        if not vault_env or not vault_env.strip():
            logger.error(
                "[HAKIBrain] HAKI_OBSIDIAN_VAULT is not set or is empty. "
                "Pipeline startup aborted — no Vault modifications will be performed."
            )
            self._vault_valid = False
            return

        vault_env = vault_env.strip()

        # Check 2: path is absolute (starts with /)
        if not vault_env.startswith("/"):
            logger.error(
                "[HAKIBrain] HAKI_OBSIDIAN_VAULT is not an absolute path "
                "(got %r). Pipeline startup aborted — no Vault modifications "
                "will be performed.", vault_env
            )
            self._vault_valid = False
            return

        vault_path = Path(vault_env)

        # Check 3: path exists on disk
        if not vault_path.exists():
            logger.error(
                "[HAKIBrain] HAKI_OBSIDIAN_VAULT path does not exist: %s. "
                "Pipeline startup aborted — no Vault modifications will be "
                "performed.", vault_path
            )
            self._vault_valid = False
            return

        # Check 4: path is readable and writable
        if not os.access(vault_path, os.R_OK | os.W_OK):
            logger.error(
                "[HAKIBrain] HAKI_OBSIDIAN_VAULT path is not read/write "
                "accessible: %s. Pipeline startup aborted — no Vault "
                "modifications will be performed.", vault_path
            )
            self._vault_valid = False
            return

        # All checks passed — vault is valid.
        self._vault_valid = True
        logger.info("[HAKIBrain] Vault path validated: %s", vault_path)

        # ----- Create the 3-folder structure -----
        for folder in [self._raw, self._processed, self._wiki, self._conversations]:
            folder.mkdir(parents=True, exist_ok=True)
            logger.info("[HAKIBrain] Ensured folder exists: %s", folder)

        # Initialise ChromaDB collection
        if self._embeddings is not None:
            self._chroma_collection = self._embeddings.get_or_create_collection(
                name=_WIKI_COLLECTION_NAME,
                metadata={"description": "HAKI wiki pages — multilingual"},
            )
            logger.info("[HAKIBrain] ChromaDB collection ready: %s", _WIKI_COLLECTION_NAME)

            # Re-create the MemoryNoteWriter now that the real ChromaDB
            # collection is available, so Fast/Heavy Pass notes get embedded.
            from core.memory.memory_note_writer import MemoryNoteWriter

            self._note_writer = MemoryNoteWriter(
                vault_root=self._vault,
                chroma_collection=self._chroma_collection,
            )

    # ------------------------------------------------------------------
    # Live-turn deferral (Req 9.5, 9.8)
    # ------------------------------------------------------------------

    def acquire_live_turn(self) -> None:
        """Signal that a live conversational turn has started.

        While the live turn is held, the pipeline will defer all
        Vault-modifying operations (``_ingest_file``, ``_process_conversation_log``)
        until the turn completes (Req 9.5).
        """
        self._live_turn_event.clear()
        logger.debug("[HAKIBrain] Live turn acquired — pipeline deferred")

    def release_live_turn(self) -> None:
        """Signal that the live conversational turn has completed.

        Unblocks any waiting pipeline operations (Req 9.5).
        """
        self._live_turn_event.set()
        logger.debug("[HAKIBrain] Live turn released — pipeline may proceed")

    async def _wait_for_live_turn(self) -> bool:
        """Wait until no live turn is active, with a 10-minute timeout.

        Returns ``True`` if the caller may proceed (no live turn active or
        the live turn completed within the timeout window). Returns ``False``
        if the wait exceeds 10 minutes, meaning the caller should abort its
        deferred operation (Req 9.8).
        """
        if self._live_turn_event.is_set():
            # No live turn active — proceed immediately.
            return True

        logger.info(
            "[HAKIBrain] Live turn active — deferring Vault-modifying operation "
            "until turn completes (max 10 min)"
        )
        try:
            await asyncio.wait_for(self._live_turn_event.wait(), timeout=600.0)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "[HAKIBrain] Live-turn deferral timeout exceeded (10 min). "
                "Aborting deferred Vault-modifying operation."
            )
            return False

    def start_watching(self) -> None:
        """Start background task that auto-ingests files dropped in raw/."""
        if self._watch_interval > 0 and self._watcher_task is None:
            self._watcher_task = asyncio.get_event_loop().create_task(
                self._watch_loop(), name="haki_brain_watcher"
            )
            logger.info(
                "[HAKIBrain] Started raw/ watcher (interval=%.0fs)", self._watch_interval
            )

    def stop_watching(self) -> None:
        """Stop the background watcher."""
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            self._watcher_task = None

    # ------------------------------------------------------------------
    # Ingestion: raw/ → processed/ + wiki/
    # ------------------------------------------------------------------

    async def ingest_pending(self) -> list[IngestionResult]:
        """
        Process all files currently in raw/ and return results.

        For each file:
          1. Read content (text / PDF / image OCR).
          2. Synthesise a wiki page via the LLM.
          3. Write wiki/ Markdown with source wikilink (Cross-Reference Rule).
          4. Move source file raw/ → processed/.
          5. Embed the wiki page into ChromaDB.
        """
        # Second-layer safety: abort if vault validation failed (Req 8.3, 8.5).
        if not self._vault_valid:
            logger.warning(
                "[HAKIBrain] ingest_pending() called but vault is invalid — "
                "returning early without processing."
            )
            return []

        pending = [
            f for f in self._raw.iterdir()
            if f.is_file() and f.name not in self._in_progress
        ]

        if not pending:
            return []

        logger.info("[HAKIBrain] Found %d file(s) in raw/ to ingest", len(pending))
        results: list[IngestionResult] = []

        for file_path in pending:
            # Per-file isolation (Req 4.2, 10.1, 10.4): one file's unexpected
            # exception must never abort the rest of the batch. _ingest_file()
            # already catches its own errors and returns a failed
            # IngestionResult, but this try/except is a last-resort guard
            # against anything that slips past that (e.g. a bug in a helper).
            try:
                result = await self._ingest_file(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[HAKIBrain] Unexpected error ingesting %s: %s", file_path.name, exc
                )
                result = IngestionResult(
                    source_file=file_path.name,
                    success=False,
                    error=f"Unexpected error: {exc}",
                )
            results.append(result)

        # Run summary diagnostic (Req 4.2, 10.1, 10.4).
        processed = sum(1 for r in results if r.success)
        notes_created = sum(1 for r in results if r.success and r.wiki_page_path)
        fast_pass_successes = sum(1 for r in results if r.pass_used == "fast")
        heavy_pass_invocations = sum(1 for r in results if r.pass_used == "heavy")
        errors = sum(1 for r in results if not r.success)
        logger.info(
            "[HAKIBrain] Run summary: %d file(s) processed, %d note(s) created, "
            "%d fast-pass success(es), %d heavy-pass invocation(s), %d error(s)",
            processed, notes_created, fast_pass_successes, heavy_pass_invocations, errors,
        )

        return results

    async def ingest_file(self, file_path: Path) -> IngestionResult:
        """Ingest a single file into the brain (can be called directly)."""
        return await self._ingest_file(file_path)

    # ------------------------------------------------------------------
    # Conversation processing: conversations/ (never moved/modified)
    # ------------------------------------------------------------------

    async def process_pending_conversations(self) -> list[IngestionResult]:
        """
        Process all unprocessed conversation logs on or before yesterday.

        Unlike raw/ file ingestion, conversation logs in conversations/ are
        NEVER moved or deleted — they remain the human-readable, permanent
        chat history. Only ProcessTracker.mark_processed() tracks completion
        so a log is never reprocessed (Req 6.5).
        """
        # Second-layer safety: abort if vault validation failed (Req 8.3, 8.5).
        if not self._vault_valid:
            logger.warning(
                "[HAKIBrain] process_pending_conversations() called but vault is "
                "invalid — returning early without processing."
            )
            return []

        from datetime import date, timedelta

        cutoff = date.today() - timedelta(days=1)  # Req 6.4: exclude today's in-progress log
        filenames = self._process_tracker.get_unprocessed_conversations(
            cutoff, self._conversations
        )

        if not filenames:
            return []

        logger.info(
            "[HAKIBrain] Found %d unprocessed conversation log(s)", len(filenames)
        )
        results: list[IngestionResult] = []

        for filename in filenames:  # already chronological oldest-first (Req 6.6)
            try:
                result = await self._process_conversation_log(filename)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[HAKIBrain] Unexpected error processing conversation %s: %s",
                    filename, exc,
                )
                result = IngestionResult(
                    source_file=filename,
                    success=False,
                    error=f"Unexpected error: {exc}",
                )
            if result.success:
                # Req 6.5, 6.8: only mark processed on success, so a failed
                # log is retried on the next Conversation_Scheduler run.
                self._process_tracker.mark_processed(filename)
            results.append(result)

        # Run summary diagnostic (Req 6.8, mirrors ingest_pending()'s summary).
        processed = sum(1 for r in results if r.success)
        notes_created = sum(1 for r in results if r.success and r.wiki_page_path)
        fast_pass_successes = sum(1 for r in results if r.pass_used == "fast")
        heavy_pass_invocations = sum(1 for r in results if r.pass_used == "heavy")
        errors = sum(1 for r in results if not r.success)
        logger.info(
            "[HAKIBrain] Conversation run summary: %d log(s) processed, %d note(s) "
            "created, %d fast-pass success(es), %d heavy-pass invocation(s), "
            "%d error(s)",
            processed, notes_created, fast_pass_successes, heavy_pass_invocations, errors,
        )

        return results

    async def _process_conversation_log(self, filename: str) -> IngestionResult:
        """Process one conversations/{filename} log through the Fast Pass /
        Heavy Pass hybrid pipeline.

        This mirrors `_ingest_file()`'s pass logic exactly, except the log
        file itself is NEVER moved or modified (Req 6.3, 6.7) — conversations/
        is the permanent, human-readable chat history, unlike raw/.
        """
        # Req 9.5, 9.8: wait for any active live turn to complete before
        # performing Vault-modifying operations.
        if not await self._wait_for_live_turn():
            logger.warning(
                "[HAKIBrain] Skipping conversation processing for %s — "
                "live-turn deferral timeout exceeded", filename,
            )
            return IngestionResult(
                source_file=filename,
                success=False,
                error="Aborted: live-turn deferral timeout exceeded (10 min)",
            )

        log_path = self._conversations / filename

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return IngestionResult(
                source_file=filename,
                success=False,
                error=f"Read failed: {exc}",
            )

        if not content.strip():
            return IngestionResult(
                source_file=filename,
                success=False,
                error="Empty conversation log",
            )

        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        source_vault_rel = f"{_CONVERSATIONS_DIR}/{filename}"

        # 1. Fast Pass
        from core.memory.fast_pass import FastPassStatus

        fast_result = self._fast_pass.extract(content, filename)

        if fast_result.status == FastPassStatus.SUCCESS:
            note_paths = []
            write_failed = False
            for entity in fast_result.entities:
                note_path = self._note_writer.write_fast_pass_note(
                    entity, filename, source_vault_rel, run_date=run_date
                )
                if note_path is None:
                    write_failed = True
                    break
                note_paths.append(note_path)

            if write_failed or not note_paths:
                logger.error(
                    "[HAKIBrain] Fast Pass note write failed for conversation %s",
                    filename,
                )
                return IngestionResult(
                    source_file=filename,
                    success=False,
                    error="Fast Pass note write failed",
                    pass_used="fast",
                )

            return IngestionResult(
                source_file=filename,
                wiki_page_path=str(note_paths[0].path),
                success=True,
                pass_used="fast",
            )

        # 2. Heavy Pass (Fast Pass returned NO_ENTITIES or ERROR)
        if self._check_low_memory():
            logger.warning(
                "[HAKIBrain] Low memory detected — deferring Heavy Pass for "
                "conversation %s to next run", filename,
            )
            return IngestionResult(
                source_file=filename,
                success=False,
                error="Deferred: low memory",
            )

        from core.memory.heavy_pass import HeavyPassExtractor, HeavyPassStatus

        heavy_extractor = HeavyPassExtractor(
            chroma_collection=self._chroma_collection,
            llm_router=self._llm,
            timeout_secs=self._llm_timeout,
        )
        heavy_result = await heavy_extractor.extract(content, filename)

        if heavy_result.status != HeavyPassStatus.SUCCESS:
            logger.error(
                "[HAKIBrain] Heavy Pass failed for conversation %s (%s)",
                filename, heavy_result.status.value,
            )
            return IngestionResult(
                source_file=filename,
                success=False,
                error=heavy_result.error_msg or heavy_result.status.value,
            )

        note_path = self._note_writer.write_heavy_pass_note(
            heavy_result, filename, source_vault_rel, run_date=run_date
        )
        if note_path is None:
            logger.error(
                "[HAKIBrain] Heavy Pass note write failed for conversation %s",
                filename,
            )
            return IngestionResult(
                source_file=filename,
                success=False,
                error="Heavy Pass note write failed",
                pass_used="heavy",
            )

        return IngestionResult(
            source_file=filename,
            wiki_page_path=str(note_path.path),
            success=True,
            pass_used="heavy",
        )

    # ------------------------------------------------------------------
    # Memory: store a fact directly (no raw/ file needed)
    # ------------------------------------------------------------------

    async def remember_fact(
        self,
        text: str,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> WikiPage:
        """
        Synthesise and store a fact/preference directly into the wiki.

        Used when the user says something like "remember that my exam is June 20"
        — no file is dropped in raw/; the fact goes straight to wiki/.

        Parameters
        ----------
        text:
            The fact or note to remember.
        title:
            Optional wiki page title.  If None, generated from the text.
        tags:
            Optional tags list.  If None, extracted by the LLM.
        """
        # Generate a title if not provided
        if title is None:
            title = _slugify(text[:60])

        # Build wiki page content
        if self._llm is not None:
            markdown_body = await self._llm.chat(
                text,
                system_prompt=_FACT_SYSTEM,
            )
        else:
            markdown_body = f"# {title}\n\n{text}\n"

        # Extract LLM-suggested tags
        page_tags, clean_body = _extract_tags(markdown_body)
        if tags:
            page_tags.extend(tags)

        # Create a synthetic source reference
        source_ref = f"user_input_{_ts_slug()}.md"

        page = WikiPage(
            title=title,
            source_file=source_ref,
            content=clean_body,
            tags=page_tags,
        )

        wiki_path = self._write_wiki_page(page)
        await self._embed_wiki_page(page, wiki_path)

        logger.info("[HAKIBrain] Remembered fact → wiki: %s", wiki_path.name)
        return page

    # ------------------------------------------------------------------
    # Conversation memory: persist every exchange so it survives restarts
    # ------------------------------------------------------------------

    async def log_conversation(self, user_text: str, assistant_text: str) -> None:
        """Persist one user/assistant exchange into the Obsidian vault.

        Two things happen, both best-effort (a failure never breaks a turn):

        1. The exchange is appended to ``conversations/YYYY-MM-DD.md`` so it is
           human-readable inside Obsidian and survives app restarts.
        2. The exchange is embedded into ChromaDB so it becomes semantically
           searchable by :meth:`search` / :meth:`search_and_format` — i.e. the
           LLM can recall it across sessions.

        Unlike :meth:`remember_fact`, this does NOT call the LLM, so it is
        cheap enough to run on every turn.
        """
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text and not assistant_text:
            return

        ts = datetime.now(timezone.utc)
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._append_conversation_log, ts, user_text, assistant_text
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[HAKIBrain] conversation log append failed: %s", exc)

        # Embed the exchange for semantic recall (best-effort, off-thread).
        if self._chroma_collection is not None:
            snippet = f"User: {user_text}\nHAKI: {assistant_text}"
            doc_id = "conv_" + hashlib.sha256(
                f"{ts.isoformat()}|{snippet}".encode()
            ).hexdigest()[:16]
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self._chroma_executor,
                    lambda: self._chroma_collection.upsert(
                        ids=[doc_id],
                        documents=[snippet[:4_000]],
                        metadatas=[
                            {
                                "title": f"Conversation {ts.strftime('%Y-%m-%d %H:%M')}",
                                "source": "conversation",
                                "type": "conversation",
                                "created": ts.isoformat(),
                                "tags": "conversation",
                            }
                        ],
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[HAKIBrain] conversation embed failed: %s", exc)

    def _append_conversation_log(
        self, ts: datetime, user_text: str, assistant_text: str
    ) -> None:
        """Append a single exchange to today's conversation markdown file."""
        log_path = self._conversations / f"{ts.strftime('%Y-%m-%d')}.md"
        stamp = ts.strftime("%H:%M:%S")
        entry = f"\n**[{stamp}] User:** {user_text}\n\n**[{stamp}] HAKI:** {assistant_text}\n"
        if not log_path.exists():
            header = f"# Conversation log — {ts.strftime('%Y-%m-%d')}\n"
            log_path.write_text(header + entry, encoding="utf-8")
        else:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(entry)

    def load_recent_history(self, max_messages: int = 20) -> list[dict]:
        """Load the most recent conversation turns for cross-session memory.

        Reads the latest daily conversation log(s) and returns up to
        ``max_messages`` entries as ``{"role", "content"}`` dicts ready to seed
        the orchestrator's running history.  Synchronous (plain file IO) so it
        can be called once at startup.
        """
        if not self._conversations.exists():
            return []

        logs = sorted(self._conversations.glob("*.md"))
        if not logs:
            return []

        messages: list[dict] = []
        # Walk the most recent files until we have enough messages.
        for log_path in reversed(logs):
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            file_msgs: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                m = re.match(r"\*\*\[[^\]]+\]\s*(User|HAKI):\*\*\s*(.*)", line)
                if not m:
                    continue
                role = "user" if m.group(1) == "User" else "assistant"
                content = m.group(2).strip()
                if content:
                    file_msgs.append({"role": role, "content": content})
            # Prepend older-file messages before newer ones.
            messages = file_msgs + messages
            if len(messages) >= max_messages:
                break

        return messages[-max_messages:]

    # ------------------------------------------------------------------
    # Retrieval: semantic search over wiki/
    # ------------------------------------------------------------------

    async def search(
        self, query: str, k: int = 5
    ) -> list[dict]:
        """
        Semantic search over all wiki pages.

        Parameters
        ----------
        query:
            Natural language query (Hindi, English, or Hinglish).
        k:
            Maximum number of results.

        Returns
        -------
        list[dict]
            Each result contains: {"title", "content", "source", "distance"}
        """
        if self._chroma_collection is None:
            # Fall back to filename-only search if ChromaDB not initialised
            return self._fallback_text_search(query, k)

        # Skip semantic search if the embedding model hasn't finished loading yet.
        # The pre-warm thread will have it ready within ~30s of startup.
        if self._embeddings is not None and not getattr(
            getattr(self._embeddings, '_embed_fn', None), '_model_ready', True
        ):
            logger.debug("[HAKIBrain] Embedding model not ready yet — skipping semantic search")
            return []

        try:
            count = self._chroma_collection.count()
            if count == 0:
                return []
            results = await asyncio.get_event_loop().run_in_executor(
                self._chroma_executor,
                lambda: self._chroma_collection.query(
                    query_texts=[query],
                    n_results=min(k, count),
                    include=["documents", "metadatas", "distances"],
                ),
            )
        except Exception as exc:
            logger.warning("[HAKIBrain] ChromaDB query failed: %s", exc)
            return self._fallback_text_search(query, k)

        hits: list[dict] = []
        if not results or not results.get("ids"):
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "id": doc_id,
                    "title": meta.get("title", doc_id),
                    "content": doc,
                    "source": meta.get("source", ""),
                    "distance": dist,
                    "wiki_path": meta.get("wiki_path", ""),
                }
            )

        return hits

    async def search_and_format(
        self, query: str, k: int = 5
    ) -> str:
        """
        Semantic search and format results as a Markdown summary for the LLM context.
        """
        hits = await self.search(query, k=k)
        if not hits:
            return ""

        parts = ["## Relevant Knowledge\n"]
        for hit in hits:
            parts.append(f"### {hit['title']}\n{hit['content'][:800]}\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        """Background loop: check raw/ every _watch_interval seconds."""
        while True:
            try:
                await asyncio.sleep(self._watch_interval)
                results = await self.ingest_pending()
                if results:
                    logger.info(
                        "[HAKIBrain] Watcher ingested %d file(s)", len(results)
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[HAKIBrain] Watcher error: %s", exc)

    # ------------------------------------------------------------------
    # Internal: hardware guard (Req 7.6)
    # ------------------------------------------------------------------

    def _check_low_memory(self) -> bool:
        """
        Return True if available system memory is below the configured
        low-memory threshold (self._low_memory_mb), signalling that the
        Heavy Pass should be deferred to the next scheduled run.

        Returns False (no guard) if psutil is not installed — pipeline
        configuration must never prevent Heavy Pass execution outright.
        """
        try:
            import psutil  # type: ignore[import]
        except ImportError:
            logger.debug(
                "[HAKIBrain] psutil not installed — skipping low-memory guard"
            )
            return False

        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception as exc:
            logger.warning("[HAKIBrain] Failed to read available memory: %s", exc)
            return False

        is_low = available_mb < self._low_memory_mb
        if is_low:
            logger.warning(
                "[HAKIBrain] Low memory detected: %.1f MB available < %.1f MB threshold",
                available_mb, self._low_memory_mb,
            )
        return is_low

    # ------------------------------------------------------------------
    # Internal: file ingestion
    # ------------------------------------------------------------------

    async def _ingest_file(self, file_path: Path) -> IngestionResult:
        """Process one file from raw/ through the hybrid Fast Pass / Heavy Pass
        pipeline.

        Fast Pass runs first (deterministic, LLM-free). If it finds >=1
        entity, one Memory_Note is written per entity and the source file is
        moved to processed/ only after ALL writes succeed. If Fast Pass finds
        nothing (or errors), the Heavy Pass (LLM-backed) is attempted instead.
        If neither pass succeeds, the source file is left untouched in raw/.
        """
        filename = file_path.name

        # Req 9.5, 9.8: wait for any active live turn to complete before
        # performing Vault-modifying operations.
        if not await self._wait_for_live_turn():
            logger.warning(
                "[HAKIBrain] Skipping ingestion of %s — live-turn deferral "
                "timeout exceeded", filename,
            )
            return IngestionResult(
                source_file=filename,
                success=False,
                error="Aborted: live-turn deferral timeout exceeded (10 min)",
            )

        self._in_progress.add(filename)

        try:
            logger.info("[HAKIBrain] Ingesting: %s", filename)

            # 1. Read content (unchanged read path)
            content, _content_type = await self._read_file(file_path)
            if not content.strip():
                return IngestionResult(
                    source_file=filename,
                    success=False,
                    error="Empty or unreadable file",
                )

            run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            source_vault_rel = f"{_RAW_DIR}/{filename}"

            # 2. Fast Pass
            from core.memory.fast_pass import FastPassStatus

            fast_result = self._fast_pass.extract(content, filename)

            if fast_result.status == FastPassStatus.SUCCESS:
                note_paths = []
                write_failed = False
                for entity in fast_result.entities:
                    note_path = self._note_writer.write_fast_pass_note(
                        entity, filename, source_vault_rel, run_date=run_date
                    )
                    if note_path is None:
                        write_failed = True
                        break
                    note_paths.append(note_path)

                if write_failed or not note_paths:
                    logger.error(
                        "[HAKIBrain] Fast Pass note write failed for %s — "
                        "leaving source in raw/", filename,
                    )
                    return IngestionResult(
                        source_file=filename,
                        success=False,
                        error="Fast Pass note write failed",
                        pass_used="fast",
                    )

                return self._finalize_ingestion(
                    file_path, filename, note_paths[0].path, pass_used="fast"
                )

            # 3. Heavy Pass (Fast Pass returned NO_ENTITIES or ERROR)
            if self._check_low_memory():
                logger.warning(
                    "[HAKIBrain] Low memory detected — deferring Heavy Pass for "
                    "%s to next run; leaving source in raw/", filename,
                )
                return IngestionResult(
                    source_file=filename,
                    success=False,
                    error="Deferred: low memory",
                )

            from core.memory.heavy_pass import HeavyPassExtractor, HeavyPassStatus

            heavy_extractor = HeavyPassExtractor(
                chroma_collection=self._chroma_collection,
                llm_router=self._llm,
                timeout_secs=self._llm_timeout,
            )
            heavy_result = await heavy_extractor.extract(content, filename)

            if heavy_result.status != HeavyPassStatus.SUCCESS:
                logger.error(
                    "[HAKIBrain] Heavy Pass failed for %s (%s) — leaving source "
                    "in raw/", filename, heavy_result.status.value,
                )
                return IngestionResult(
                    source_file=filename,
                    success=False,
                    error=heavy_result.error_msg or heavy_result.status.value,
                )

            note_path = self._note_writer.write_heavy_pass_note(
                heavy_result, filename, source_vault_rel, run_date=run_date
            )
            if note_path is None:
                logger.error(
                    "[HAKIBrain] Heavy Pass note write failed for %s — leaving "
                    "source in raw/", filename,
                )
                return IngestionResult(
                    source_file=filename,
                    success=False,
                    error="Heavy Pass note write failed",
                    pass_used="heavy",
                )

            return self._finalize_ingestion(
                file_path, filename, note_path.path, pass_used="heavy"
            )

        except Exception as exc:
            logger.error("[HAKIBrain] Failed to ingest %s: %s", filename, exc)
            return IngestionResult(
                source_file=filename,
                success=False,
                error=str(exc),
            )
        finally:
            self._in_progress.discard(filename)

    def _finalize_ingestion(
        self, file_path: Path, filename: str, note_path: Path, pass_used: str
    ) -> IngestionResult:
        """Move *file_path* from raw/ to processed/ after notes were written
        successfully, and build the resulting IngestionResult.

        Note writes are NOT rolled back if the move fails (Req 4.6, 10.5):
        the Memory_Note(s) already exist and deleting them would destroy
        information. Instead the move failure is logged as a diagnostic and
        the source file is left in raw/ — this means the file will be
        re-ingested on the next run, which is a deliberate "never lose the
        source" tradeoff that can produce a duplicate Memory_Note in the rare
        case where the move fails right after a successful write.
        """
        try:
            dest = self._get_processed_dest(filename)
            shutil.move(str(file_path), str(dest))
            logger.info("[HAKIBrain] Moved %s → processed/", filename)
        except OSError as exc:
            logger.error(
                "[HAKIBrain] Failed to move %s to processed/ after successful "
                "%s Pass write: %s — leaving source in raw/ (may be "
                "re-ingested on the next run)", filename, pass_used, exc,
            )
            return IngestionResult(
                source_file=filename,
                wiki_page_path=str(note_path),
                success=False,
                error=f"Move to processed/ failed: {exc}",
                pass_used=pass_used,
            )

        return IngestionResult(
            source_file=filename,
            wiki_page_path=str(note_path),
            success=True,
            pass_used=pass_used,
        )

    def _get_processed_dest(self, filename: str) -> Path:
        """Return a collision-safe destination path in processed/ for
        *filename* (Req 4.3).

        If `processed/{filename}` already exists, the destination becomes
        `processed/{stem}_{timestamp}{suffix}` instead, so an existing file
        of the same name is never overwritten.
        """
        dest = self._processed / filename
        if not dest.exists():
            return dest
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        return self._processed / f"{stem}_{_ts_slug()}{suffix}"

    async def _read_file(self, file_path: Path) -> tuple[str, str]:
        """
        Read a file and return (content_text, content_type_label).

        Handles: text files, PDFs (via pdfminer/pypdf), images (basic OCR).
        """
        suffix = file_path.suffix.lower()

        if suffix in _SUPPORTED_PDF:
            return await self._read_pdf(file_path), "PDF document"

        if suffix in _SUPPORTED_IMAGE:
            return await self._read_image_ocr(file_path), "image/screenshot"

        if suffix in _SUPPORTED_TEXT_EXTENSIONS or suffix == "":
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, lambda: file_path.read_text(errors="replace")
            )
            return content, "text"

        # Unknown extension — try reading as text
        loop = asyncio.get_event_loop()
        try:
            content = await loop.run_in_executor(
                None, lambda: file_path.read_text(errors="replace")
            )
            return content, "text (unknown format)"
        except Exception:
            return "", "unreadable"

    async def _read_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_pdf_text, file_path)

    async def _read_image_ocr(self, file_path: Path) -> str:
        """Basic OCR for images using pytesseract if available."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _ocr_image, file_path)

    # ------------------------------------------------------------------
    # Internal: wiki page writing
    # ------------------------------------------------------------------

    def _write_wiki_page(self, page: WikiPage) -> Path:
        """Write wiki page to disk. Returns the path."""
        filename = f"{page.title}.md"
        wiki_path = self._wiki / filename

        # Handle collisions
        if wiki_path.exists():
            wiki_path = self._wiki / f"{page.title}_{_ts_slug()}.md"

        wiki_path.write_text(page.to_markdown(), encoding="utf-8")
        logger.info("[HAKIBrain] Wrote wiki page: %s", wiki_path.name)
        return wiki_path

    # ------------------------------------------------------------------
    # Internal: ChromaDB embedding
    # ------------------------------------------------------------------

    async def _embed_wiki_page(self, page: WikiPage, wiki_path: Path) -> None:
        """Embed the wiki page into ChromaDB."""
        if self._chroma_collection is None or self._embeddings is None:
            return

        try:
            doc_id = hashlib.sha256(str(wiki_path).encode()).hexdigest()[:16]

            # Use the clean body for embedding (no YAML frontmatter noise)
            self._chroma_collection.upsert(
                ids=[doc_id],
                documents=[page.content[:4_000]],  # ChromaDB document limit
                metadatas=[
                    {
                        "title": page.title,
                        "source": f"[[processed/{page.source_file}]]",
                        "wiki_path": str(wiki_path),
                        "created": page.created_at,
                        "tags": ",".join(page.tags),
                    }
                ],
            )
            logger.debug("[HAKIBrain] Embedded wiki page: %s", page.title)
        except Exception as exc:
            logger.warning("[HAKIBrain] ChromaDB embed failed for %s: %s", page.title, exc)

    # ------------------------------------------------------------------
    # Internal: fallback text search (when ChromaDB not available)
    # ------------------------------------------------------------------

    def _fallback_text_search(self, query: str, k: int) -> list[dict]:
        """Simple case-insensitive full-text search over wiki/ files."""
        query_lower = query.lower()
        results: list[tuple[int, dict]] = []

        for wiki_file in self._wiki.glob("*.md"):
            try:
                content = wiki_file.read_text(encoding="utf-8", errors="replace")
                score = content.lower().count(query_lower)
                if score > 0:
                    results.append(
                        (
                            score,
                            {
                                "id": wiki_file.stem,
                                "title": wiki_file.stem,
                                "content": content[:800],
                                "source": "",
                                "distance": 1.0 / (score + 1),
                                "wiki_path": str(wiki_file),
                            },
                        )
                    )
            except Exception:
                continue

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:k]]

    # ------------------------------------------------------------------
    # Public: rebuild ChromaDB index from wiki/
    # ------------------------------------------------------------------

    async def rebuild_index(self) -> int:
        """
        Re-embed all wiki/ pages into ChromaDB.

        Useful after restoring from backup or first-run.
        Returns the number of pages indexed.
        """
        if self._chroma_collection is None or self._embeddings is None:
            logger.warning("[HAKIBrain] Cannot rebuild — no embeddings engine")
            return 0

        count = 0
        for wiki_file in self._wiki.glob("*.md"):
            try:
                content = wiki_file.read_text(encoding="utf-8", errors="replace")
                # Extract metadata from frontmatter
                meta = _parse_frontmatter(content)
                body = _strip_frontmatter(content)

                doc_id = hashlib.sha256(str(wiki_file).encode()).hexdigest()[:16]
                self._chroma_collection.upsert(
                    ids=[doc_id],
                    documents=[body[:4_000]],
                    metadatas=[
                        {
                            "title": meta.get("title", wiki_file.stem),
                            "source": str(meta.get("sources", [""])[0]) if meta.get("sources") else "",
                            "wiki_path": str(wiki_file),
                            "tags": meta.get("tags", ""),
                        }
                    ],
                )
                count += 1
            except Exception as exc:
                logger.warning("[HAKIBrain] Failed to re-embed %s: %s", wiki_file.name, exc)

        logger.info("[HAKIBrain] Rebuilt index: %d pages", count)
        return count

    # ------------------------------------------------------------------
    # Public: stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return counts for each folder."""
        raw_count = len(list(self._raw.glob("*"))) if self._raw.exists() else 0
        proc_count = len(list(self._processed.glob("*"))) if self._processed.exists() else 0
        wiki_count = len(list(self._wiki.glob("*.md"))) if self._wiki.exists() else 0
        chroma_count = self._chroma_collection.count() if self._chroma_collection else 0
        return {
            "raw": raw_count,
            "processed": proc_count,
            "wiki": wiki_count,
            "chroma_vectors": chroma_count,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text[:80].strip("_")
    return text or "note"


def _ts_slug() -> str:
    """Short timestamp slug for collision avoidance."""
    return str(int(time.time()))


def _truncate_for_llm(text: str, max_chars: int = 40_000) -> str:
    """Truncate text to fit in the LLM context window."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... content truncated ...]\n\n" + text[-half:]


def _extract_tags(markdown: str) -> tuple[list[str], str]:
    """
    Extract tags from the last line of markdown if it starts with "TAGS:".
    Returns (tags_list, clean_markdown_without_tags_line).
    """
    lines = markdown.rstrip().split("\n")
    if lines and lines[-1].strip().startswith("TAGS:"):
        tags_line = lines[-1].strip()[5:].strip()
        tags = [t.strip().lstrip("#") for t in tags_line.split(",") if t.strip()]
        return tags, "\n".join(lines[:-1]).strip()
    return [], markdown.strip()


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from Markdown."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        import yaml  # type: ignore[import]
        return yaml.safe_load(content[3:end]) or {}
    except Exception:
        return {}


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from Markdown."""
    if not content.startswith("---"):
        return content
    end = content.find("---", 3)
    if end == -1:
        return content
    return content[end + 3:].lstrip("\n")


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF using pdfminer or pypdf."""
    # Try pdfminer first (better text extraction)
    try:
        from pdfminer.high_level import extract_text  # type: ignore[import]
        return extract_text(str(file_path))
    except ImportError:
        pass

    # Fall back to pypdf
    try:
        from pypdf import PdfReader  # type: ignore[import]
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass

    return f"[PDF file: {file_path.name} — install pdfminer or pypdf to extract text]"


def _ocr_image(file_path: Path) -> str:
    """Basic OCR using pytesseract."""
    try:
        from PIL import Image  # type: ignore[import]
        import pytesseract    # type: ignore[import]
        img = Image.open(str(file_path))
        return pytesseract.image_to_string(img)
    except ImportError:
        return f"[Image file: {file_path.name} — install Pillow and pytesseract for OCR]"
    except Exception as exc:
        return f"[OCR failed for {file_path.name}: {exc}]"
