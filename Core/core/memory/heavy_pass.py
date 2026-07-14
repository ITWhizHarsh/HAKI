"""
HeavyPassExtractor — LLM-backed fallback extraction for the Heavy Pass.

Part of the HAKI Brain Memory Processing Pipeline. The Heavy Pass is invoked
only when the Fast Pass yields nothing. It retrieves the top-3 semantically
similar existing Memory_Notes from ChromaDB, then calls Bonsai-8B (via the
existing `LLMRouter` with `prefer_local=True`) with an evolutionary prompt
that merges old memory with new content and draws an Evolutionary_Link.

See: .kiro/specs/haki-brain-memory-processing-pipeline/design.md
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.model_provider.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# Default Heavy Pass LLM timeout in seconds; overridden by the
# HAKI_PIPELINE_LLM_TIMEOUT_SECS environment variable.
_HEAVY_PASS_TIMEOUT_SECS = 30

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# System prompt describing Bonsai-8B's role during the Heavy Pass. Used for
# both the evolutionary and fresh-synthesis prompts.
_HEAVY_PASS_SYSTEM = (
    "You are Bonsai-8B, the memory synthesis assistant for HAKI's personal "
    "knowledge vault. Your job is to read raw source content and produce a "
    "single, concise Memory_Note written in Markdown that captures the "
    "durable facts worth remembering. Be factual and specific — extract "
    "names, dates, places, decisions, and relationships. Do not editorialize, "
    "add commentary, or wrap your answer in code fences. Output only the "
    "Memory_Note body (and, when instructed, an EVOLVED_FROM line)."
)

# Used when the ChromaDB semantic query returns an existing Memory_Note
# (Req 2.3, 2.4). Instructs the LLM to merge the old note with the new
# source content and draw an Evolutionary_Link via an EVOLVED_FROM line.
_EVOLUTIONARY_PROMPT = """\
An existing Memory_Note titled "{old_note_name}" was found to be semantically \
related to a new Source_File named "{filename}".

--- EXISTING MEMORY_NOTE ({old_note_name}) ---
{old_note_content}
--- END EXISTING MEMORY_NOTE ---

--- NEW SOURCE_FILE CONTENT ({filename}) ---
{new_content}
--- END NEW SOURCE_FILE CONTENT ---

Merge the facts from the existing Memory_Note with the new information from \
the Source_File into a single, updated Memory_Note. Keep facts that still \
hold true, update or replace facts that the new content supersedes or \
contradicts, and add any genuinely new facts. Write the result as concise \
Markdown.

After the Memory_Note body, add exactly one line in this exact format, with \
no other text on that line:
EVOLVED_FROM: {old_note_name}
"""

# Used when the ChromaDB semantic query returns no results (Req 2.6).
# Instructs the LLM to synthesize a fresh Memory_Note with no old note to
# evolve from, so no EVOLVED_FROM line should be produced.
_FRESH_SYNTHESIS_PROMPT = """\
--- NEW SOURCE_FILE CONTENT ({filename}) ---
{new_content}
--- END NEW SOURCE_FILE CONTENT ---

No existing related memory was found for this Source_File. Synthesize a new, \
concise Memory_Note in Markdown that captures the durable facts worth \
remembering from the content above. Do not include an EVOLVED_FROM line — \
this is a fresh memory with no prior note to evolve from.
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class HeavyPassStatus(str, Enum):
    """Outcome of a Heavy Pass extraction attempt."""

    SUCCESS = "success"      # Bonsai-8B produced usable memory content
    TIMEOUT = "timeout"      # LLM did not respond within the configured timeout
    LLM_ERROR = "llm_error"  # LLM returned empty / unparseable content
    ERROR = "error"          # other exception (e.g. ChromaDB failure)


@dataclass
class HeavyPassResult:
    """Result of running HeavyPassExtractor.extract() on a Source_File."""

    status: HeavyPassStatus
    memory_content: str = ""
    old_memory_note_name: Optional[str] = None
    evolutionary_link: Optional[str] = None
    error_msg: Optional[str] = None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class HeavyPassExtractor:
    """
    Orchestrates semantic retrieval + Bonsai-8B synthesis for a single
    Source_File, invoked only when the Fast Pass yields nothing.

    Parameters
    ----------
    chroma_collection : optional
        The `haki_wiki` ChromaDB collection used to perform the top-3
        semantic query against existing Memory_Notes (Req 2.2). May be
        `None` (e.g. in tests, or before the Vault has any embeddings) —
        `_query_old_note()` then returns `None` without querying.
    llm_router : LLMRouter, optional
        The shared `LLMRouter` instance used to call Bonsai-8B during
        `extract()` with `prefer_local=True`.
    timeout_secs : float
        Maximum time to wait for the LLM response during `extract()`,
        default 30 seconds, normally overridden by the
        `HAKI_PIPELINE_LLM_TIMEOUT_SECS` environment variable by the caller.
    """

    def __init__(
        self,
        chroma_collection=None,
        llm_router: Optional["LLMRouter"] = None,
        timeout_secs: float = _HEAVY_PASS_TIMEOUT_SECS,
    ) -> None:
        self._chroma = chroma_collection
        self._llm = llm_router
        self._timeout = timeout_secs

    # ------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------

    def _query_old_note(self, content: str) -> Optional[Tuple[str, str]]:
        """
        Query the ChromaDB collection for the top-3 Memory_Notes most
        semantically similar to *content*, and return the title and
        document content of the top-1 result as the candidate old
        Memory_Note for evolutionary linking (Req 2.2, 2.3).

        Returns
        -------
        A `(title, content)` tuple for the top-1 result, or `None` if:
          - no `chroma_collection` was injected,
          - the collection is empty, or
          - the query raises an exception or returns no results (Req 5.5 —
            an empty result set means the new Memory_Note is created
            without an Evolutionary_Link).
        """
        if self._chroma is None:
            return None

        try:
            count = self._chroma.count()
            if count <= 0:
                return None

            results = self._chroma.query(
                query_texts=[content[:2_000]],
                n_results=min(3, count),
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("[HeavyPassExtractor] ChromaDB query failed: %s", exc)
            return None

        if not results:
            return None

        ids = results.get("ids")
        if not ids or not ids[0]:
            return None

        # Top-1 result is selected as the evolutionary-link candidate.
        metadatas = results.get("metadatas")
        documents = results.get("documents")
        if not metadatas or not metadatas[0]:
            return None
        if not documents or not documents[0]:
            return None

        top_meta = metadatas[0][0]
        top_doc = documents[0][0]
        title = top_meta.get("title")
        if not title or not top_doc:
            return None
        return title, top_doc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(self, content: str, filename: str) -> HeavyPassResult:
        """
        Run the Heavy Pass for a single Source_File.

        1. Query ChromaDB for a candidate old Memory_Note (`_query_old_note`).
        2. Build the evolutionary prompt (old note found) or the fresh
           synthesis prompt (no old note found).
        3. Call `LLMRouter.chat(prefer_local=True)` wrapped in
           `asyncio.wait_for()` with the configured timeout (Req 2.7).
        4. Parse the `EVOLVED_FROM:` line out of the response, if present,
           and strip it from the returned `memory_content` (Req 2.5, 2.8).
        """
        try:
            old_note = self._query_old_note(content)

            if old_note is not None:
                old_note_name, old_note_content = old_note
                prompt = _EVOLUTIONARY_PROMPT.format(
                    old_note_name=old_note_name,
                    old_note_content=old_note_content[:3_000],
                    filename=filename,
                    new_content=content[:4_000],
                )
            else:
                old_note_name = None
                prompt = _FRESH_SYNTHESIS_PROMPT.format(
                    filename=filename,
                    new_content=content[:6_000],
                )

            try:
                response = await asyncio.wait_for(
                    self._llm.chat(
                        user_message=prompt,
                        system_prompt=_HEAVY_PASS_SYSTEM,
                        prefer_local=True,
                    ),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[HeavyPassExtractor] Bonsai-8B timed out after %ss for %s",
                    self._timeout,
                    filename,
                )
                return HeavyPassResult(
                    status=HeavyPassStatus.TIMEOUT,
                    error_msg=f"LLM timed out after {self._timeout}s for {filename}",
                )

            if not response or not response.strip():
                logger.error(
                    "[HeavyPassExtractor] Empty LLM response for %s", filename
                )
                return HeavyPassResult(
                    status=HeavyPassStatus.LLM_ERROR,
                    error_msg=f"Empty LLM response for {filename}",
                )

            match = re.search(r"^EVOLVED_FROM:\s*(.+)$", response, re.MULTILINE)
            memory_content = response
            evolutionary_link: Optional[str] = None
            if match:
                evolutionary_link = match.group(1).strip()
                memory_content = (
                    response[: match.start()] + response[match.end():]
                ).strip()

            if not memory_content.strip():
                logger.error(
                    "[HeavyPassExtractor] Unparseable LLM response for %s", filename
                )
                return HeavyPassResult(
                    status=HeavyPassStatus.LLM_ERROR,
                    error_msg=f"Unparseable LLM response for {filename}",
                )

            return HeavyPassResult(
                status=HeavyPassStatus.SUCCESS,
                memory_content=memory_content,
                old_memory_note_name=evolutionary_link or old_note_name,
                evolutionary_link=evolutionary_link,
            )

        except Exception as exc:
            logger.error(
                "[HeavyPassExtractor] Unexpected error extracting %s: %s",
                filename,
                exc,
            )
            return HeavyPassResult(status=HeavyPassStatus.ERROR, error_msg=str(exc))
