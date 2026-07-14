"""
IntentRouter — intent classification and capability routing.

Classifies each conversational turn into one of the nine HAKI intents
and routes execution to the correct capability handler stub.
Side-effecting intents pass through the DialogueManager gate before
any execution begins; read-only / conversational intents proceed
directly.

Design: Intent Routing.
Requirements: 6.1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from core.dialogue import DialogueManager, SlotFillResult
from core.model_provider import Capability, ModelProviderRegistry, StubModelProvider
from core.orchestrator.orchestrator import Intent, TurnContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hinglish-aware chat persona system prompt
# ---------------------------------------------------------------------------

_HINGLISH_CHAT_SYSTEM_PROMPT = """You are HAKI, a warm, friendly, and helpful personal voice assistant. \
You remember the ongoing conversation and refer back to earlier turns naturally.

LANGUAGE RULES (very important):
- ALWAYS write your responses using ONLY Latin / Roman / English characters. NEVER use Devanagari script (do not write Hindi in the देवनागरी script under any circumstance).
- If the user speaks or writes in Hindi or Hinglish, reply in Hinglish — that is, Hindi expressed in Roman/Latin letters.
- If the user speaks in English, reply in English.
- Match the user's language naturally; mirror their mix of Hindi and English.

STYLE:
- Keep replies concise and conversational, because they will be spoken aloud by a text-to-speech engine.
- Be warm and personable, like a helpful friend.

EXAMPLE (follow this pattern):
User: hello bhaiya kaise ho aaj kya chal raha hai zindagi me ? do you have any work for me ? agar koi bhi kaam ho, to mujhe bata do mai kar dunga
HAKI: Arre namaste! Main bilkul mast hoon, aap sunao kaise ho? Filhaal koi kaam nahi hai par aap chaaho to main aapki kisi cheez mein help kar sakta hoon.

Notice how HAKI replied in Roman-script Hinglish (Latin letters only), never in Devanagari."""


# ---------------------------------------------------------------------------
# IntentResult — the output of classify()
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """
    The result of intent classification for a single conversational turn.

    Attributes
    ----------
    intent : Intent
        The classified intent.
    confidence : float
        Confidence score in [0.0, 1.0].  1.0 for now (stub LLM); real
        backends may return calibrated scores.
    raw_label : str
        The raw label string returned by the LLM before parsing.  Useful
        for debugging and future logging.
    language_hint : str | None
        The language composition detected for this turn, if available.
    """

    intent: Intent
    confidence: float = 1.0
    raw_label: str = ""
    language_hint: str | None = None


# ---------------------------------------------------------------------------
# Type alias for capability handlers
# ---------------------------------------------------------------------------

# A capability handler is an async generator that accepts a TurnContext
# and yields zero or more response tokens / chunks.
CapabilityHandler = Callable[[TurnContext], AsyncGenerator[str, None]]


# ---------------------------------------------------------------------------
# Intent → required slots (for DialogueManager gate on side-effecting intents)
# ---------------------------------------------------------------------------

# Side-effecting intents — must pass through the DialogueManager gate.
_SIDE_EFFECTING_INTENTS: frozenset[Intent] = frozenset(
    [
        Intent.MAC_COMMAND,
        Intent.SCHEDULE,
        Intent.TASK,
        Intent.RUN_AUTOMATION,
    ]
)

# Read-only / conversational intents — proceed without dialogue gating.
_READ_ONLY_INTENTS: frozenset[Intent] = frozenset(
    [
        Intent.CHAT,
        Intent.RECALL,
        Intent.REMEMBER,
        Intent.READ_ALOUD,
        Intent.IMAGE,
        Intent.META,
    ]
)

# Minimum slots required per side-effecting intent.
# An empty list means no pre-execution slot check beyond intent classification.
_REQUIRED_SLOTS: dict[Intent, list[str]] = {
    Intent.MAC_COMMAND: [],
    Intent.SCHEDULE: ["event_title", "event_datetime"],
    Intent.TASK: ["task_title"],
    Intent.RUN_AUTOMATION: ["automation_name"],
}


# ---------------------------------------------------------------------------
# Module-level default ImageStudio instance (session-scoped, lazy-created)
# ---------------------------------------------------------------------------

_default_image_studio: "Any | None" = None
_default_studio_lock: "Any" = None

# ---------------------------------------------------------------------------
# Module-level default AutomationLibrary instance (session-scoped, lazy-created)
# ---------------------------------------------------------------------------

_default_automation_library: "Any | None" = None
_default_automation_library_lock: "Any" = None


def _get_default_image_studio() -> "Any":
    """
    Return a module-level default :class:`~core.image_studio.ImageStudio`
    instance, created lazily on first call.

    This instance persists for the lifetime of the intent_router module
    (i.e. the lifetime of the HAKI Core service), giving all image turns
    within a session access to the same history.  Tests that need a fresh
    history should inject their own instance via ``ctx.extras["image_studio"]``.
    """
    import threading  # noqa: PLC0415
    from core.image_studio import ImageStudio  # noqa: PLC0415

    global _default_image_studio, _default_studio_lock
    if _default_studio_lock is None:
        _default_studio_lock = threading.Lock()

    with _default_studio_lock:
        if _default_image_studio is None:
            _default_image_studio = ImageStudio()
        return _default_image_studio


def _get_default_automation_library() -> "Any":
    """
    Return a module-level default :class:`~core.automation.AutomationLibrary`
    instance, created lazily on first call.

    This instance persists for the lifetime of the intent_router module,
    giving all run_automation turns within a session access to the same
    registered automations.  Tests that need a fresh or pre-populated
    library should inject their own instance via
    ``ctx.extras["automation_library_instance"]``.
    """
    import threading  # noqa: PLC0415
    from core.automation import AutomationLibrary  # noqa: PLC0415

    global _default_automation_library, _default_automation_library_lock
    if _default_automation_library_lock is None:
        _default_automation_library_lock = threading.Lock()

    with _default_automation_library_lock:
        if _default_automation_library is None:
            _default_automation_library = AutomationLibrary()
        return _default_automation_library


# ---------------------------------------------------------------------------
# Module-level default MacController instance (session-scoped, lazy-created)
# ---------------------------------------------------------------------------

_default_mac_controller: "Any | None" = None
_default_mac_controller_lock: "Any" = None
_default_screen_agent: "Any | None" = None
_default_screen_agent_lock: "Any" = None


def _get_default_mac_controller() -> "Any":
    """
    Return a module-level default :class:`~core.automation.mac_controller.MacController`
    instance, created lazily on first call.

    The single instance persists for the lifetime of the intent_router
    module so all mac_command turns within a session share it.
    """
    import threading  # noqa: PLC0415
    from core.automation.mac_controller import MacController  # noqa: PLC0415

    global _default_mac_controller, _default_mac_controller_lock
    if _default_mac_controller_lock is None:
        _default_mac_controller_lock = threading.Lock()

    with _default_mac_controller_lock:
        if _default_mac_controller is None:
            _default_mac_controller = MacController()
        return _default_mac_controller


def _get_default_screen_agent(
    llm_router: "Any | None" = None,
    ipc_writer: "Any | None" = None,
) -> "Any":
    """
    Return a module-level default :class:`~core.automation.screen_agent.ScreenAgent`
    instance, created lazily on first call.

    The agent is always re-created if a fresh llm_router/ipc_writer is supplied
    so IPC callbacks stay current across turns.
    """
    import threading  # noqa: PLC0415
    from core.automation.screen_agent import ScreenAgent  # noqa: PLC0415

    global _default_screen_agent, _default_screen_agent_lock
    if _default_screen_agent_lock is None:
        _default_screen_agent_lock = threading.Lock()

    gemini_key = os.environ.get("HAKI_GEMINI_API_KEY")
    mac_ctrl = _get_default_mac_controller()

    with _default_screen_agent_lock:
        if _default_screen_agent is None or llm_router is not None:
            _default_screen_agent = ScreenAgent(
                llm_router=llm_router,
                gemini_api_key=gemini_key,
                mac_controller=mac_ctrl,
                ipc_writer=ipc_writer,
            )
        return _default_screen_agent


# ---------------------------------------------------------------------------
# Mac-command fast-path detection (keyword/regex, English + Hinglish)
# ---------------------------------------------------------------------------

# Screen reading / analysis requests (English + romanized Hindi).
_SCREEN_ANALYSIS_RE = re.compile(
    r"\b("
    r"what'?s\s+on\s+(my|the)\s+screen|"
    r"read\s+(my|the)\s+screen|"
    r"analy[sz]e\s+(my|the)\s+screen|"
    r"what\s+do\s+you\s+see|"
    r"what\s+can\s+you\s+see|"
    r"describe\s+(my|the)\s+screen|"
    r"screen\s+pe\s+kya\s+hai|"
    r"screen\s+par\s+kya\s+hai|"
    r"screen\s+(padho|pado|dekho|dekh)|"
    r"dekho\s+screen\s+pe|"
    r"screen\s+read\s+kar"
    r")\b",
    re.IGNORECASE,
)

# App-open requests. English verbs precede the app name; Hinglish verbs
# typically follow the app name (e.g. "whatsapp kholo").
_OPEN_APP_RE = re.compile(
    r"("
    r"\b(open|launch|start|run)\b\s+\w+|"          # english: open whatsapp
    r"\b(open|launch|start|chalu|chaalu)\s+karo\b|"  # open karo / chalu karo
    r"\b(kholo|khol\s+do|khol\s+de|chalu\s+karo|chaalu\s+karo|start\s+karo|on\s+karo)\b"
    r")",
    re.IGNORECASE,
)

# Keywords stripped from a transcript to recover the bare app name.
_OPEN_KEYWORDS_RE = re.compile(
    r"\b(please|kya|tum|aap|mera|meri|app|application|"
    r"open|launch|start|run|chalu|chaalu|karo|kholo|khol|do|de|on)\b",
    re.IGNORECASE,
)


def _is_screen_analysis_request(transcript: str) -> bool:
    """Return True if the transcript asks HAKI to read/analyze the screen."""
    return bool(_SCREEN_ANALYSIS_RE.search(transcript or ""))


def _is_open_app_request(transcript: str) -> bool:
    """Return True if the transcript asks HAKI to open/launch an app."""
    return bool(_OPEN_APP_RE.search(transcript or ""))


def _extract_app_name(transcript: str) -> str:
    """
    Recover the bare app name from an open-app transcript by stripping the
    open/launch/kholo/chalu-karo keywords and leftover filler words.

    Anything after a conjunction ("and", "aur", "then", "phir") is dropped so
    "open music and play" yields "music", not "music and play".
    """
    text = transcript or ""
    # Cut at the first conjunction so a trailing second command does not leak
    # into the app name.
    text = re.split(
        r"\s+(?:and|aur|then|phir|uske\s+baad|after\s+that)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = _OPEN_KEYWORDS_RE.sub(" ", text)
    # Collapse whitespace and trim trailing punctuation.
    text = re.sub(r"\s+", " ", text).strip(" .,!?\t\n")
    return text


# ---------------------------------------------------------------------------
# Rich Mac-action parsing (call / message / volume / mute / close / type)
# ---------------------------------------------------------------------------

# Words that mark a messaging app so we can pick the right channel.
_WHATSAPP_RE = re.compile(r"\bwhats?\s?app\b", re.IGNORECASE)
_IMESSAGE_RE = re.compile(r"\b(imessage|messages|text|sms|message)\b", re.IGNORECASE)

# Close / quit an app.
_CLOSE_KEYWORDS_RE = re.compile(
    r"\b(close|quit|kill|exit|band|bandh)\b", re.IGNORECASE
)
_CLOSE_STRIP_RE = re.compile(
    r"\b(please|kya|tum|aap|app|application|close|quit|kill|exit|band|bandh|karo|"
    r"kar|do|de|the|window)\b",
    re.IGNORECASE,
)

# Trailing filler that often follows a contact name in Hinglish.
_NAME_STRIP_RE = re.compile(
    r"\b(please|kar|karo|kardo|kar\s+do|do|de|de\s+do|lagao|laga|lagado|"
    r"ko|se|now|abhi|jaldi)\b",
    re.IGNORECASE,
)


def _clean_name(text: str) -> str:
    """Trim filler/command words and punctuation from an extracted name."""
    text = _NAME_STRIP_RE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .,!?\t\n")
    return text


def _extract_recipient_and_message(text: str) -> tuple[str, str]:
    """Best-effort split of a messaging command into (recipient, message).

    Handles common English and Hinglish phrasings, e.g.:
      * "message mummy saying I'll be late"
      * "send papa a text that dinner is ready"
      * "mummy ko message bhejo ki main aa raha hoon"
      * "papa ko whatsapp karo hello"
      * "bhai ko bol do good night"
    Returns ("", "") when nothing usable could be parsed.
    """
    t = (text or "").strip()

    # Form A (English): "... RECIPIENT (saying|that|:|-) MESSAGE"
    m = re.search(
        r"(?:message|text|whats?app|sms|send|msg|bolo|bol\s+do|tell)\b\s*"
        r"(?:to\s+|a\s+|an\s+)?(?P<rec>.+?)\s+(?:saying|that|:|-|ki)\s+(?P<msg>.+)$",
        t,
        re.IGNORECASE,
    )
    if m:
        rec = _clean_name(m.group("rec"))
        msg = m.group("msg").strip(" .,!?\t\n")
        if rec and msg:
            return rec, msg

    # Form B (Hinglish): "RECIPIENT ko (whatsapp|message|text) (karo|bhejo|...) MESSAGE"
    m = re.search(
        r"(?P<rec>.+?)\s+ko\s+(?:whats?app|message|msg|text|sms)?\s*"
        r"(?:karo|kardo|kar\s+do|bhejo|bhej\s+do|bhej|send)\s+(?P<msg>.+)$",
        t,
        re.IGNORECASE,
    )
    if m:
        rec = _clean_name(m.group("rec"))
        msg = m.group("msg").strip(" .,!?\t\n")
        if rec and msg:
            return rec, msg

    # Form C (Hinglish): "RECIPIENT ko bolo/bol do MESSAGE"
    m = re.search(
        r"(?P<rec>.+?)\s+ko\s+(?:bolo|bol\s+do|keh\s+do|kehna)\s+(?P<msg>.+)$",
        t,
        re.IGNORECASE,
    )
    if m:
        rec = _clean_name(m.group("rec"))
        msg = m.group("msg").strip(" .,!?\t\n")
        if rec and msg:
            return rec, msg

    # Form D (English, no connector): "message RECIPIENT MESSAGE WORDS..."
    # e.g. "message mummy good evening", "text papa on my way".
    # The recipient is the first token after the verb; the rest is the body.
    # Last-resort, voice-friendly fallback.
    m = re.search(
        r"\b(?:message|msg|text|whats?app|sms)\s+(?:to\s+)?(?P<rec>[A-Za-z]+)\s+(?P<msg>.+)$",
        t,
        re.IGNORECASE,
    )
    if m:
        rec = _clean_name(m.group("rec"))
        msg = m.group("msg").strip(" .,!?\t\n")
        if rec and msg:
            return rec, msg

    return "", ""


def _extract_call_target(text: str) -> str:
    """Extract the person/number to call from a call command."""
    t = (text or "").strip()
    # Hinglish: "mummy ko call karo" → name precedes "ko call".
    m = re.search(r"(?P<rec>.+?)\s+ko\s+(?:call|phone|dial|fone)\b", t, re.IGNORECASE)
    if m:
        rec = _clean_name(m.group("rec"))
        if rec:
            return rec
    # English: "call mummy" / "phone papa" / "dial 9876543210".
    m = re.search(r"\b(?:call|phone|dial|fone)\b\s+(?:to\s+)?(?P<rec>.+)$", t, re.IGNORECASE)
    if m:
        return _clean_name(m.group("rec"))
    return ""


def _parse_mac_command(transcript: str) -> dict | None:
    """Parse a transcript into a structured Mac action.

    Returns a dict like ``{"action": "call", "target": "mummy"}`` or ``None``
    when the transcript is not a recognised rich Mac action (callers then fall
    back to open-app / screen-analysis / help).
    """
    t = (transcript or "").strip()
    low = t.lower()
    if not low:
        return None

    # --- Volume: explicit numeric level ("volume 50", "set volume to 30%") ---
    m = re.search(r"\bvolume\b[^0-9]*?(\d{1,3})", low)
    if m and not re.search(r"\b(call|phone|dial|message|text|whats?app)\b", low):
        return {"action": "volume_set", "level": int(m.group(1))}

    # --- Brightness (built-in display) ---
    # Note: STT often mangles "increase the brightness" into "I will add the
    # brightness" — so any mention of brightness routes here, defaulting to UP.
    if re.search(r"\b(brightness|bright|roshni|chamak)\b", low) or re.search(
        r"\bscreen\b.*\b(light|roshni|dim)\b", low
    ):
        if re.search(r"\b(max|maximum|full|poori|puri|sabse\s+zyada|highest)\b", low):
            return {"action": "brightness_max"}
        if re.search(r"\b(min|minimum|lowest|sabse\s+kam|bilkul\s+kam)\b", low):
            return {"action": "brightness_min"}
        if re.search(
            r"\b(down|kam|ghata|ghatao|decrease|low|lower|reduce|dim|halka)\b", low
        ):
            return {"action": "brightness_down"}
        # default → up (covers increase / badha / zyada / high / "add")
        return {"action": "brightness_up"}

    # --- Media playback (Apple Music / Spotify) ---
    if re.search(r"\b(pause|pauz)\b", low) and not re.search(r"\bvideo\b", low):
        return {"action": "media", "media": "pause"}
    if re.search(r"\b(resume|unpause|phir\s+se\s+chalao)\b", low):
        return {"action": "media", "media": "play"}
    if re.search(r"\b(next|skip|agla|agle|aage\s+badha)\b", low) and re.search(
        r"\b(song|track|gaana|gana|music)\b", low
    ):
        return {"action": "media", "media": "next"}
    if re.search(r"\b(previous|prev|pichla|pichhla|last|peeche)\b", low) and re.search(
        r"\b(song|track|gaana|gana|music)\b", low
    ):
        return {"action": "media", "media": "previous"}
    if re.search(
        r"\b(play|chala|chalao|chalu)\b.*\b(song|music|gaana|gana|track|playlist)\b",
        low,
    ) or re.search(
        r"\b(song|music|gaana|gana|playlist)\b.*\b(chala|chalao|chalu|play)\b", low
    ):
        return {"action": "media", "media": "play"}

    # --- Mute / unmute ---
    if re.search(r"\bunmute\b|awaaz\s+(chalu|on)|sound\s+on", low):
        return {"action": "unmute"}
    if re.search(r"\bmute\b|silent\s+kar|awaaz\s+band|sound\s+band|chup\s+kar", low):
        return {"action": "mute"}

    # --- Volume up / down ---
    if re.search(
        r"(volume|awaaz|sound|aawaz).*(up|badha|badhao|increase|tej|zyada|high|raise)"
        r"|(badha|badhao|increase|raise).*(volume|awaaz|sound|aawaz)",
        low,
    ):
        return {"action": "volume_up"}
    if re.search(
        r"(volume|awaaz|sound|aawaz).*(down|kam|ghata|ghatao|decrease|low|halka|lower)"
        r"|(kam\s+kar|ghatao|decrease|lower).*(volume|awaaz|sound|aawaz)",
        low,
    ):
        return {"action": "volume_down"}

    # --- Call ---
    if re.search(r"\b(call|phone|dial|fone)\b", low) and not re.search(
        r"\b(message|text|whats?app|sms)\b", low
    ):
        target = _extract_call_target(t)
        if target:
            return {"action": "call", "target": target}
        return {"action": "call", "target": ""}

    # --- Messaging (WhatsApp / iMessage) ---
    # Only treat as a messaging command when there's a real send intent.
    # The bare word "whatsapp" / "message" must NOT hijack an open request
    # like "open whatsapp" (which previously kept asking "kisko message bheju?").
    has_msg_keyword = bool(
        re.search(r"\b(message|msg|text|sms|bhejo|bhej\s+do)\b", low)
        or re.search(r"\bko\b.*\b(bolo|bol\s+do|keh\s+do)\b", low)
    )
    is_open_request = bool(re.search(r"\b(open|launch|start|khol|kholo)\b", low))
    if has_msg_keyword:
        recipient, message = _extract_recipient_and_message(t)
        has_send_verb = bool(
            re.search(r"\b(bhejo|bhej\s+do|send|bolo|bol\s+do|keh\s+do)\b", low)
        )
        # Claim messaging only if we extracted something, or there is an explicit
        # send verb. An "open ..." request with nothing to send falls through to
        # the open-app handler instead.
        if (recipient or message or has_send_verb) and not (
            is_open_request and not recipient and not message
        ):
            app = "whatsapp" if _WHATSAPP_RE.search(low) else "imessage"
            return {
                "action": "message",
                "app": app,
                "recipient": recipient,
                "message": message,
            }

    # --- Close / quit an app ---
    if _CLOSE_KEYWORDS_RE.search(low):
        name = _CLOSE_STRIP_RE.sub(" ", t)
        name = re.sub(r"\s+", " ", name).strip(" .,!?\t\n")
        if name:
            return {"action": "close", "app": name}

    # --- Type text into the frontmost app ---
    m = re.search(r"^\s*(?:type|likho|likh\s+do|write)\b[:,]?\s+(?P<txt>.+)$", t, re.IGNORECASE)
    if m:
        txt = m.group("txt").strip()
        send = bool(re.search(r"\b(and\s+send|aur\s+bhejo|then\s+send)\b", txt, re.IGNORECASE))
        txt = re.sub(r"\s*(?:and\s+send|aur\s+bhejo|then\s+send)\s*$", "", txt, flags=re.IGNORECASE).strip()
        return {"action": "type", "text": txt, "send": send}

    return None


def _is_mac_action_request(transcript: str) -> bool:
    """True if the transcript is any rich Mac action (call/message/volume/...)."""
    return _parse_mac_command(transcript) is not None


# Verbs that mark a clause as an actionable command (multi-command splitter).
_COMMAND_VERB_RE = re.compile(
    r"\b(open|launch|start|run|close|quit|kill|exit|play|pause|resume|next|skip|"
    r"previous|volume|mute|unmute|brightness|message|msg|text|call|phone|dial|type|"
    r"khol|kholo|band|bandh|chalu|chala|chalao|bhejo|bhej|likho|screen)\b",
    re.IGNORECASE,
)

_CONJUNCTION_SPLIT_RE = re.compile(
    r"\s+(?:and|aur|then|phir|uske\s+baad|after\s+that)\s+", re.IGNORECASE
)


def _split_into_commands(transcript: str) -> list[str]:
    """Split a multi-step utterance into independent command clauses.

    Conservative on purpose: message / type bodies (which may legitimately
    contain "and") are never split, and a split is only honoured when at least
    two of the resulting clauses look like real commands.  Otherwise the
    original transcript is returned unchanged as a single clause.
    """
    t = (transcript or "").strip()
    if not t:
        return [t]

    # Protect message / type bodies from being torn apart by "and"/"aur".
    parsed = _parse_mac_command(t)
    if parsed and parsed.get("action") == "message" and (
        parsed.get("message") or parsed.get("recipient")
    ):
        return [t]
    if parsed and parsed.get("action") == "type" and parsed.get("text"):
        return [t]

    parts = [
        p.strip(" .,!?\t\n")
        for p in _CONJUNCTION_SPLIT_RE.split(t)
        if p.strip(" .,!?\t\n")
    ]
    if len(parts) < 2:
        return [t]

    command_like = [p for p in parts if _COMMAND_VERB_RE.search(p)]
    if len(command_like) >= 2:
        return parts
    return [t]


def _parse_bare_media(clause: str) -> str | None:
    """Map a short standalone clause to a media command (for split clauses).

    Used so a clause like "play" or "pause" left over after splitting
    ("open music and play") still drives the media player, without making the
    bare verb hijack normal chat in the classifier fast-path.
    """
    c = (clause or "").strip().lower().strip(" .,!?")
    if re.fullmatch(r"(pause|pauz|ruko|rok\s+do)", c):
        return "pause"
    if re.fullmatch(r"(play|chalao|chala\s+do|resume|gaana\s+chalao|music\s+chalao)", c):
        return "play"
    if re.fullmatch(r"(next|skip|agla|agla\s+gaana|next\s+song|next\s+track)", c):
        return "next"
    if re.fullmatch(r"(previous|prev|pichla|pichla\s+gaana|previous\s+song)", c):
        return "previous"
    return None


async def _execute_single_mac(
    transcript: str,
    controller: "Any",
    gemini_api_key: "str | None",
    llm_router: "Any",
    ipc_writer: "Any",
) -> str:
    """Execute ONE Mac command clause and return a short spoken-friendly reply.

    Dispatch order: screen-analysis → deterministic action (call / message /
    volume / brightness / media / mute / close / type) → bare media verb →
    open-app → agentic ScreenAgent loop for anything that needs to see the UI.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return ""

    # 1) Screen reading / analysis (vision via Gemini).
    if _is_screen_analysis_request(transcript):
        return await controller.analyze_screen(
            question=transcript, gemini_api_key=gemini_api_key
        )

    # 2) Rich deterministic actions.
    parsed = _parse_mac_command(transcript)
    if parsed is not None:
        action = parsed.get("action")

        if action == "call":
            target = parsed.get("target") or ""
            if not target:
                return "Kisko call karu? Naam ya number batao."
            return await controller.make_call(target)

        if action == "message":
            recipient = parsed.get("recipient") or ""
            message = parsed.get("message") or ""
            if not recipient or not message:
                return (
                    "Kisko aur kya message bhejna hai? "
                    "Jaise: 'mummy ko message bhejo main aa raha hoon'."
                )
            if parsed.get("app") == "whatsapp":
                return await controller.send_whatsapp(recipient, message)
            return await controller.send_imessage(recipient, message)

        if action == "volume_set":
            return await controller.set_volume(int(parsed.get("level", 50)))
        if action == "volume_up":
            return await controller.adjust_volume("up")
        if action == "volume_down":
            return await controller.adjust_volume("down")
        if action == "mute":
            return await controller.set_mute(True)
        if action == "unmute":
            return await controller.set_mute(False)

        if action == "brightness_up":
            return await controller.adjust_brightness("up")
        if action == "brightness_down":
            return await controller.adjust_brightness("down")
        if action == "brightness_max":
            return await controller.set_brightness_extreme(True)
        if action == "brightness_min":
            return await controller.set_brightness_extreme(False)

        if action == "media":
            return await controller.media_control(parsed.get("media", "playpause"))

        if action == "close":
            return await controller.close_app(parsed.get("app", ""))

        if action == "type":
            return await controller.type_text(
                parsed.get("text", ""), press_return=bool(parsed.get("send"))
            )

    # 3) Bare media verb (e.g. a leftover "play" / "pause" clause).
    bare_media = _parse_bare_media(transcript)
    if bare_media is not None:
        return await controller.media_control(bare_media)

    # 4) Simple open / launch an app.
    if _is_open_app_request(transcript):
        app_name = _extract_app_name(transcript)
        if not app_name:
            return "Kaunsa app kholu? App ka naam batao."
        return await controller.open_app(app_name)

    # 5) AGENTIC LOOP — anything that needs to see the screen and click through.
    agent = _get_default_screen_agent(llm_router=llm_router, ipc_writer=ipc_writer)
    if ipc_writer is not None:
        agent._ipc_writer = ipc_writer
    result = await agent.run(transcript)
    return result.message


# ---------------------------------------------------------------------------
# Handler stubs (async generators)
# ---------------------------------------------------------------------------


async def _chat_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Wire LLM for conversational chat with memory context (Req 6, 7).
    Passes context to the LLM for a freeform response shaped by personality.
    """
    extras: dict = getattr(ctx, "extras", {}) or {}
    llm_router = extras.get("llm_router")
    haki_brain = extras.get("haki_brain")
    
    # If no LLM is available, fall back to stub behavior
    if llm_router is None:
        yield f"[chat stub] '{ctx.transcript}'"
        return
    
    # Build context with memory if available
    context_parts: list[str] = []
    
    # Add memory context from HAKIBrain if available
    if haki_brain is not None:
        try:
            # Strict timeout — brain context is optional, never block the turn
            brain_context = await asyncio.wait_for(
                haki_brain.search_and_format(ctx.transcript, k=3),
                timeout=2.0,
            )
            if brain_context:
                context_parts.append(brain_context)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("_chat_handler: HAKIBrain query skipped: %r", exc)
    
    # Build the prompt — Hinglish-aware persona + (optional) brain context.
    system_prompt = _HINGLISH_CHAT_SYSTEM_PROMPT
    
    if context_parts:
        system_prompt += "\n\n" + "\n".join(context_parts)
    
    # Running conversation history injected by the Orchestrator (prior turns).
    history = ctx.extras.get("conversation_history") or []

    user_message = ctx.transcript
    
    try:
        response = await llm_router.chat(
            user_message=user_message,
            system_prompt=system_prompt,
            history=history,
        )
        
        yield response if response else "I'm sorry, I couldn't generate a response."
        
    except Exception as exc:
        logger.error("_chat_handler: LLM invocation failed: %r", exc)
        yield f"I encountered an error while processing your request: {exc}"


async def _recall_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Memory recall (Req 7.3, 7.7).

    Searches the HAKI Brain (Obsidian wiki + stored conversations) and, when an
    LLM is available, has it answer the user's question grounded in the hits.
    Falls back to reading the top matches aloud when no LLM is wired.
    """
    extras: dict = getattr(ctx, "extras", {}) or {}
    haki_brain = extras.get("haki_brain")
    llm_router = extras.get("llm_router")
    query = (ctx.transcript or "").strip()

    if haki_brain is None:
        yield "Abhi meri memory available nahi hai."
        return

    try:
        hits = await asyncio.wait_for(haki_brain.search(query, k=5), timeout=3.0)
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        logger.warning("_recall_handler: brain search failed: %r", exc)
        yield "Memory search karte waqt thodi dikkat aayi."
        return

    if not hits:
        yield "Mujhe iske baare mein kuch yaad nahi hai abhi."
        return

    # Build a compact context block from the top hits.
    context = "\n\n".join(
        f"- {h.get('title', 'note')}: {str(h.get('content', ''))[:400]}"
        for h in hits[:3]
    )

    if llm_router is not None:
        try:
            answer = await llm_router.chat(
                f"Question: {query}\n\nRelevant memory:\n{context}",
                system_prompt=(
                    "You are HAKI. Answer the user's question using ONLY the "
                    "relevant memory provided. Reply concisely in Roman-script "
                    "Hinglish (Latin letters only, never Devanagari). If the "
                    "memory does not contain the answer, say so honestly."
                ),
            )
            yield answer if answer else context
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("_recall_handler: LLM grounding failed: %r", exc)

    # No LLM — read back the top match.
    top = hits[0]
    yield f"Yaad aaya — {top.get('title', 'note')}: {str(top.get('content', ''))[:500]}"


async def _remember_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Store new information into the HAKI Brain (Req 7.1).

    Persists the fact to the Obsidian wiki via ``HAKIBrain.remember_fact`` and
    only confirms to the user after the durable write completes.
    """
    extras: dict = getattr(ctx, "extras", {}) or {}
    haki_brain = extras.get("haki_brain")
    text = (ctx.transcript or "").strip()

    # Strip a leading "remember (that)" / "yaad rakho/rakhna" command phrase so
    # we store the actual fact, not the instruction.
    fact = re.sub(
        r"^\s*(please\s+)?(remember\s+(that\s+)?|yaad\s+(rakho|rakhna|rakh\s+lo)\s*(ki\s+)?|"
        r"note\s+(kar\s+lo|that\s+)?)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip() or text

    if haki_brain is None:
        yield "Abhi memory store available nahi hai, isliye yaad nahi rakh paya."
        return

    try:
        page = await haki_brain.remember_fact(fact)
        yield f"Yaad kar liya! '{page.title}' note bana diya apne knowledge base mein."
    except Exception as exc:  # noqa: BLE001
        logger.warning("_remember_handler: remember_fact failed: %r", exc)
        yield "Sorry, yaad rakhne mein dikkat aa gayi. Dobara try karein?"


async def _read_aloud_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Screen_Reader.capture_focused() + Voice_Engine for read-aloud (Req 1).
    Captures frontmost window text and streams it to TTS.
    """
    yield "[read_aloud stub] Reading screen content aloud."


async def _mac_command_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Real macOS control handler (Req 21).

    Decides the action from the transcript and drives a module-level
    :class:`~core.automation.mac_controller.MacController` singleton:

      * screen-analysis request → capture + analyze the screen via Gemini
        (needs ``HAKI_GEMINI_API_KEY`` in the environment), then describe it;
      * app-open request → strip the open/launch/kholo/chalu-karo keywords to
        recover the app name and launch it via ``open -a``;
      * otherwise → a brief help message listing what HAKI can do.

    Responses are short, spoken-friendly Roman-script Hinglish. A
    MacController instance may be injected via
    ``ctx.extras["mac_controller"]`` for tests.

    SECURITY: opening apps and capturing/analyzing the screen are powerful
    local capabilities. They require macOS Screen Recording and
    Accessibility/Automation permissions; captures may contain sensitive
    on-screen data and, for analysis, are uploaded to the Gemini API.
    """
    extras: dict = getattr(ctx, "extras", {}) or {}
    transcript = (ctx.transcript or "").strip()

    controller = extras.get("mac_controller") or _get_default_mac_controller()
    gemini_api_key = os.environ.get("HAKI_GEMINI_API_KEY")
    llm_router = extras.get("llm_router")
    ipc_writer = extras.get("ipc_writer")

    try:
        # Split multi-step utterances ("open whatsapp and message mummy good
        # evening", "open music and play") into independent clauses and run
        # each in order. Message/type bodies are protected from splitting.
        clauses = _split_into_commands(transcript)

        if len(clauses) > 1:
            parts: list[str] = []
            for clause in clauses:
                reply = await _execute_single_mac(
                    clause, controller, gemini_api_key, llm_router, ipc_writer
                )
                if reply:
                    parts.append(reply)
            yield " ".join(parts) if parts else "Ho gaya."
            return

        # Single command.
        goes_agentic = (
            not _is_screen_analysis_request(transcript)
            and not _is_mac_action_request(transcript)
            and not _is_open_app_request(transcript)
        )
        if goes_agentic:
            # Agentic loop can take a few seconds — give a quick spoken cue.
            yield "Theek hai, screen dekh ke karta hoon... "

        yield await _execute_single_mac(
            transcript, controller, gemini_api_key, llm_router, ipc_writer
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("_mac_command_handler: unexpected error: %r", exc)
        yield f"Sorry, command run karte waqt error aaya: {exc}"


async def _run_automation_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Wire Automation_Library.run() to execute a named automation (Reqs 17.2, 17.4, 17.5, 17.6, 17.7).

    Resolution order for the automation name:
    1. Check TurnContext.extras["automation_name"] — injected by the
       DialogueManager after resolving the ``automation_name`` required slot.
    2. Fall back to TurnContext.extras["automation_library_instance"] if
       injected (used in integration tests).
    3. Fall back to a module-level default :class:`AutomationLibrary` so
       the handler works out-of-the-box without extra wiring.

    The automation is looked up by *exact* name (Req 17.4).  If no exact
    match is found the library suggests the nearest stored name — we yield
    that suggestion as the response without executing anything (Req 17.4).

    Progress events (step started, completed, failed) are collected and a
    final human-readable summary is yielded to the TTS pipeline (Req 17.2).
    """
    from core.automation import AutomationLibrary  # noqa: PLC0415
    from core.execution.execution_engine import StepEventType  # noqa: PLC0415

    extras: dict = getattr(ctx, "extras", {}) or {}

    # Resolve the AutomationLibrary instance to use.
    library: AutomationLibrary = extras.get(
        "automation_library_instance"
    ) or _get_default_automation_library()

    # Resolve the automation name from the slot resolved by DialogueManager
    # (stored as "automation_name" in extras) or fall back to the transcript.
    automation_name: str = extras.get("automation_name") or ctx.transcript.strip()

    import asyncio  # noqa: PLC0415

    # Resolve optional IPC writer for streaming AUTOMATION_PROGRESS events to UI.
    ipc_writer = extras.get("ipc_writer")

    async def _send_progress(step_label: str, status: str, message: str = "") -> None:
        """Send an AUTOMATION_PROGRESS IPC event to the Swift UI."""
        if ipc_writer is not None:
            try:
                await ipc_writer({
                    "type": "AUTOMATION_PROGRESS",
                    "payload": {
                        "automation_name": automation_name,
                        "step": step_label,
                        "status": status,
                        "message": message,
                    },
                })
            except Exception as _exc:
                logger.debug("_run_automation_handler: IPC progress send failed: %r", _exc)

    lines: list[str] = []
    try:
        async for event in library.run(automation_name):
            if event.event_type == StepEventType.FAILED and event.step_id is None:
                # No-match event: the message contains the suggestion.
                no_match_msg = event.message or f"No automation named '{automation_name}' found."
                await _send_progress("(none)", "not_found", no_match_msg)
                yield no_match_msg
                return
            elif event.event_type == StepEventType.STARTED:
                step_label = event.step.intent if event.step else (event.step_id or "step")
                lines.append(f"Running step: {step_label}")
                await _send_progress(step_label, "started")
            elif event.event_type == StepEventType.COMPLETED:
                step_label = event.step.intent if event.step else (event.step_id or "step")
                lines.append(f"Completed: {step_label}")
                await _send_progress(step_label, "completed")
            elif event.event_type == StepEventType.FAILED and event.step_id is not None:
                step_label = event.step.intent if event.step else (event.step_id or "step")
                fail_msg = event.message or ""
                lines.append(
                    f"Step failed: {step_label}"
                    + (f" — {fail_msg}" if fail_msg else "")
                )
                await _send_progress(step_label, "failed", fail_msg)
            elif event.event_type == StepEventType.PLAN_COMPLETE:
                from core.execution.execution_engine import ExecutionReport  # noqa: PLC0415
                report = event.data
                if isinstance(report, ExecutionReport) and report.completion_event:
                    ce = report.completion_event
                    summary_parts: list[str] = []
                    if ce.executed_step_ids:
                        summary_parts.append(
                            f"Automation '{automation_name}' complete. "
                            f"Ran {len(ce.executed_step_ids)} step(s)."
                        )
                    else:
                        summary_parts.append(
                            f"Automation '{automation_name}' finished with no steps completed."
                        )
                    if ce.failed_steps:
                        failed_names = ", ".join(sid for sid, _ in ce.failed_steps)
                        summary_parts.append(f"Failed: {failed_names}.")
                    if ce.not_performed_step_ids:
                        summary_parts.append(
                            f"Not performed: {', '.join(ce.not_performed_step_ids)}."
                        )
                    if report.cancelled:
                        summary_parts.append("Automation was cancelled.")
                    summary = " ".join(summary_parts)
                    lines.append(summary)
                    await _send_progress("(complete)", "plan_complete", summary)
                else:
                    done_msg = event.message or f"Automation '{automation_name}' complete."
                    lines.append(done_msg)
                    await _send_progress("(complete)", "plan_complete", done_msg)
    except Exception as exc:
        logger.error("_run_automation_handler: unexpected error: %r", exc)
        await _send_progress("(error)", "failed", str(exc))
        yield f"An error occurred while running automation '{automation_name}': {exc}"
        return

    yield "\n".join(lines) if lines else f"Automation '{automation_name}' finished."


async def _image_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Wire Image_Studio.generate() / Image_Studio.edit() for image tasks (Req 15).
    Generates or edits an image from the voice description and presents the result.

    Edit detection heuristic: if the transcript contains a word like "edit",
    "modify", "change", "adjust", "make", "turn" alongside a reference to an
    existing session image, the intent is treated as an edit; otherwise it is
    treated as a new generation request.

    The ImageStudio instance is resolved from the TurnContext extras dict so
    that the orchestrator can inject it in tests or IPC wiring.  When no
    instance is present a default one (stub provider, default save dir) is
    created for this call — this ensures the image capability works
    out-of-the-box without extra configuration.
    """
    import asyncio  # noqa: PLC0415

    from core.image_studio import ImageStudio  # noqa: PLC0415

    # Resolve the ImageStudio instance:
    # 1. Check TurnContext.extras (injected by the orchestrator for tests / IPC)
    # 2. Use a session-scoped default instance on the handler module
    extras: dict = getattr(ctx, "extras", {}) or {}
    studio: ImageStudio = extras.get("image_studio") or _get_default_image_studio()

    transcript = ctx.transcript

    # Detect edit vs generate based on verb presence (Req 15.2, 15.3)
    _EDIT_VERBS = re.compile(
        r'\b(edit|modify|change|adjust|alter|update|fix|improve|make|turn|convert|'
        r'add|remove|replace|colour|color|lighten|darken|crop|resize|rotate)\b',
        re.I,
    )
    _EDIT_REFS = re.compile(
        r'\b(that|this|it|the image|the picture|the last|the latest|'
        r'earlier image|previous image|above|that one|the one)\b',
        re.I,
    )

    is_edit = (
        _EDIT_VERBS.search(transcript) is not None
        and (
            _EDIT_REFS.search(transcript) is not None
            or studio.last_image() is not None
        )
    )

    # Run the synchronous ImageStudio call in a thread so we don't block
    # the async event loop (image generation can be CPU/IO-bound).
    if is_edit:
        result = await asyncio.to_thread(studio.edit, transcript)
    else:
        result = await asyncio.to_thread(studio.generate, transcript)

    yield result.message


async def _schedule_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Wire Scheduler.propose_event() for calendar event creation (Req 11).

    Constructs an ActionableItem from the resolved slots (extracted from the
    DialogueManager-resolved extras), proposes a CalendarProposal, and
    dispatches a PROPOSAL IPC event to the Swift UI so it can render a
    proposal card for user confirmation / rejection / edit.

    The Scheduler instance and IPC writer are resolved from
    ``ctx.extras``:
      - ``ctx.extras["scheduler"]``    — a ``core.scheduler.Scheduler`` instance
      - ``ctx.extras["ipc_writer"]``   — an ``async (dict) -> None`` callable
                                         that sends a JSON ServerMessage
      - ``ctx.extras["event_title"]``  — resolved slot (event title)
      - ``ctx.extras["event_datetime"]`` — resolved slot (ISO-8601 datetime string)

    If no scheduler is injected, a simple text response is emitted instead.
    """
    from core.scheduler import Scheduler  # noqa: PLC0415
    from core.scheduler.models import Task as SchedulerTask, TaskSource, Severity  # noqa: PLC0415

    extras: dict = getattr(ctx, "extras", {}) or {}
    scheduler: Scheduler = extras.get("scheduler")
    ipc_writer = extras.get("ipc_writer")

    # Extract resolved slots (set by DialogueManager / extras).
    event_title: str = extras.get("event_title") or ""
    event_datetime: str = extras.get("event_datetime") or ""

    # If no title/datetime could be resolved, fall back to the transcript.
    if not event_title:
        event_title = ctx.transcript.strip()
    if not event_datetime:
        event_datetime = ""

    if scheduler is None:
        # No real scheduler injected — emit informative stub text.
        yield (
            f"I'll schedule that for you. Event: '{event_title}'"
            + (f" at {event_datetime}" if event_datetime else "")
            + ". (Scheduler not yet wired — no calendar event created.)"
        )
        return

    # Build an ActionableItem from the resolved slots.
    import re  # noqa: PLC0415
    date_str: str | None = None
    time_str: str | None = None
    if event_datetime:
        # Try to parse ISO-8601: "2024-06-01T14:30" or "2024-06-01 14:30"
        _dt_match = re.match(
            r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})",
            event_datetime,
        )
        if _dt_match:
            date_str = _dt_match.group(1)
            time_str = _dt_match.group(2)
        else:
            # Try date-only.
            _d_match = re.match(r"(\d{4}-\d{2}-\d{2})", event_datetime)
            if _d_match:
                date_str = _d_match.group(1)

    # We define a minimal ActionableItem-like dict here since the Swift model
    # is not available in Python Core — the IPC message carries the full data.
    actionable_dict = {
        "id": f"actionable-{ctx.transcript[:8]}",
        "source_account": "voice",
        "source_message_id": f"turn-{id(ctx)}",
        "type": "event",
        "date": date_str,
        "time": time_str,
        "location": None,
        "description": event_title,
        "needs_clarification": not (date_str and time_str),
    }

    # Build the CalendarProposal via Scheduler.propose_event.
    # Since Scheduler.propose_event expects an ActionableItem (Swift model),
    # and the Python Scheduler works with its own data model, we create a Task
    # here and simultaneously send the proposal IPC message for the UI.

    import uuid  # noqa: PLC0415
    proposal_id = str(uuid.uuid4())
    proposal_payload = {
        "id": proposal_id,
        "title": event_title,
        "date": date_str,
        "time": time_str,
        "location": None,
        "description": event_title,
        "needs_clarification": not (date_str and time_str),
        "status": "proposed",
    }

    # Send the PROPOSAL message to the Swift UI (Req 11.1 — ≤5 s after identification).
    if ipc_writer is not None:
        try:
            await ipc_writer({
                "type": "PROPOSAL",
                "payload": proposal_payload,
            })
            logger.debug("_schedule_handler: sent PROPOSAL IPC message for '%s'", event_title)
        except Exception as exc:
            logger.warning("_schedule_handler: failed to send PROPOSAL IPC message: %r", exc)

    # Also schedule a Task via the Python Scheduler so reminders are set up.
    try:
        import datetime as _dt  # noqa: PLC0415
        due_dt: _dt.datetime | None = None
        if date_str:
            try:
                if time_str:
                    due_dt = _dt.datetime.fromisoformat(f"{date_str}T{time_str}:00")
                else:
                    due_dt = _dt.datetime.fromisoformat(f"{date_str}T09:00:00")
                due_dt = due_dt.replace(tzinfo=_dt.timezone.utc)
            except ValueError:
                due_dt = None

        task = scheduler.create_task(
            title=event_title,
            due_date=due_dt,
            severity=None,  # will default + notify (Req 12.11)
            source=TaskSource.COMMAND,
        )
        reminders = scheduler.compute_reminders(task)
        logger.debug(
            "_schedule_handler: created task %s with %d reminder(s)",
            task.id, len(reminders),
        )
        yield (
            f"I've proposed a calendar event: '{event_title}'"
            + (f" on {date_str}" if date_str else "")
            + (f" at {time_str}" if time_str else "")
            + ". Please confirm, reject, or edit in the HAKI panel."
        )
    except Exception as exc:
        logger.warning("_schedule_handler: Scheduler.create_task failed: %r", exc)
        yield (
            f"I tried to schedule '{event_title}' but encountered an error: {exc}. "
            "Please try again."
        )


async def _task_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Wire Task_Tracker.add() for task creation (Req 13).

    Persists the task and assigns severity; no partial write on failure.
    Sends a TASK_ADDED IPC control event so the Swift UI can refresh.

    Resolves from ``ctx.extras``:
      - ``ctx.extras["scheduler"]``    — ``core.scheduler.Scheduler`` instance
      - ``ctx.extras["task_tracker"]`` — ``core.scheduler.TaskTracker`` instance
      - ``ctx.extras["ipc_writer"]``   — async IPC send callable
      - ``ctx.extras["task_title"]``   — resolved slot (task title)
    """
    from core.scheduler import Scheduler, TaskTracker  # noqa: PLC0415
    from core.scheduler.models import TaskSource, Severity  # noqa: PLC0415

    extras: dict = getattr(ctx, "extras", {}) or {}
    scheduler: Scheduler = extras.get("scheduler")
    task_tracker: TaskTracker = extras.get("task_tracker")
    ipc_writer = extras.get("ipc_writer")

    task_title: str = extras.get("task_title") or ctx.transcript.strip()

    if scheduler is None or task_tracker is None:
        yield (
            f"I'll add that as a task: '{task_title}'. "
            "(Task_Tracker not yet wired — no persistent task created.)"
        )
        return

    try:
        task = scheduler.create_task(
            title=task_title,
            severity=None,  # will default + notify (Req 12.11)
            source=TaskSource.COMMAND,
        )
        task_tracker.add(task)
        logger.debug("_task_handler: added task %s ('%s')", task.id, task_title)

        # Notify the Swift UI via IPC.
        if ipc_writer is not None:
            try:
                await ipc_writer({
                    "type": "TASK_ADDED",
                    "payload": task.to_dict(),
                })
            except Exception as exc:
                logger.warning("_task_handler: failed to send TASK_ADDED IPC: %r", exc)

        # Schedule the due-date watcher if we are inside an async context.
        try:
            await task_tracker.schedule_due_watcher_async(task)
        except Exception:
            pass  # Watcher is best-effort.

        yield (
            f"Task added: '{task.title}' (severity: {task.severity.value}). "
            "I'll remind you at the appropriate time."
        )
    except Exception as exc:
        logger.warning("_task_handler: TaskTracker.add failed: %r", exc)
        yield (
            f"I couldn't add the task '{task_title}' due to an error: {exc}. "
            "Please try again."
        )


async def _meta_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Clock / Settings / Privacy_Manager for meta requests (Req 2, 9, 14, 20).
    Handles queries about time, settings, permissions, and privacy toggles.
    """
    yield f"[meta stub] Handling meta request: '{ctx.transcript}'"


async def _unknown_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Fallback handler for unclassified intents.
    Routes back to chat as a safe default.
    """
    yield "[unknown stub] Could not classify intent; defaulting to chat."


# ---------------------------------------------------------------------------
# Handler for missing-slot gate (inline async generator)
# ---------------------------------------------------------------------------


async def _missing_slots_handler(
    missing: list[str],
) -> Callable[[TurnContext], AsyncGenerator[str, None]]:
    """Return a handler that informs the user about missing required slots."""

    async def _handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
        missing_str = ", ".join(missing)
        yield (
            f"I need a bit more information before I can do that. "
            f"Missing: {missing_str}. Could you provide those details?"
        )

    return _handler


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------

# Mapping from intent to its handler stub.
_HANDLER_MAP: dict[Intent, CapabilityHandler] = {
    Intent.CHAT: _chat_handler,
    Intent.RECALL: _recall_handler,
    Intent.REMEMBER: _remember_handler,
    Intent.READ_ALOUD: _read_aloud_handler,
    Intent.MAC_COMMAND: _mac_command_handler,
    Intent.RUN_AUTOMATION: _run_automation_handler,
    Intent.IMAGE: _image_handler,
    Intent.SCHEDULE: _schedule_handler,
    Intent.TASK: _task_handler,
    Intent.META: _meta_handler,
    Intent.UNKNOWN: _unknown_handler,
}

# System prompt template for intent classification.
_CLASSIFY_SYSTEM_PROMPT = """You are the intent classifier for HAKI, a personal AI assistant.
Classify the user's request into exactly one of these intents:
  chat           — general conversation, questions, or statements
  recall         — asking HAKI to recall or look up something it knows
  remember       — asking HAKI to remember or store a piece of information
  read_aloud     — asking HAKI to read on-screen content aloud
  mac_command    — ad-hoc control of the Mac (open app, send message, etc.)
  run_automation — running a named/saved automation
  image          — generating or editing an image
  schedule       — creating a calendar event or reminder
  task           — creating or managing a task
  meta           — time, settings, privacy, or permission queries

Reply with a single intent keyword and nothing else."""


class IntentRouter:
    """
    Classifies a conversational turn into an intent and routes it to the
    appropriate capability handler stub.

    classify()
    ----------
    Uses the LLM (via Model_Provider) with a structured classification prompt
    to map the transcript to one of the nine HAKI intents.  Returns an
    :class:`IntentResult` with the intent and metadata.

    route()
    -------
    Accepts an :class:`IntentResult` and a :class:`TurnContext`, runs the
    DialogueManager gate for side-effecting intents, and returns an async
    generator that yields response chunks for the turn.  If required slots
    are missing, the generator yields an informative clarification message
    instead of executing the capability.

    Side-effecting intents (mac_command, schedule, task, run_automation)
    pass through the DialogueManager gate: ``dialogue_manager.assess()``
    is called with the required slots; if slots are insufficient the
    capability handler is NOT invoked.

    Read-only / conversational intents (chat, recall, remember, read_aloud,
    image, meta) proceed without dialogue gating.

    Requirements: 6.1.
    """

    def __init__(
        self,
        dialogue_manager: DialogueManager | None = None,
        llm_provider: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dialogue_manager:
            Injected :class:`~core.dialogue.DialogueManager` instance used to
            gate side-effecting intents.  If None a default instance is
            created (no memory context).
        llm_provider:
            A :class:`~core.model_provider.ModelProvider` for LLM calls.
            Must support ``.invoke(prompt, ...)`` returning a dict with an
            ``"output"`` key.  If None a :class:`~core.model_provider.StubModelProvider`
            is used (returns ``"chat"`` intent for any input).
        """
        self._dialogue_manager: DialogueManager = dialogue_manager or DialogueManager()

        # Set up a default LLM provider stub if none is supplied.
        if llm_provider is None:
            registry = ModelProviderRegistry()
            self._llm = StubModelProvider(Capability.LLM, registry)
        else:
            self._llm = llm_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(
        self,
        transcript: str,
        language_result: dict | None = None,
    ) -> IntentResult:
        """
        Classify the transcript into one of the nine HAKI intents.

        Calls the LLM provider (via Model_Provider abstraction) with a
        structured classification prompt.  Falls back to ``Intent.CHAT``
        if the LLM response cannot be mapped to a known intent.

        The call to ``self._llm.invoke()`` is synchronous but wrapped in
        ``asyncio.to_thread`` so it does not block the event loop when a
        real (potentially slow) model backend is in use.

        Parameters
        ----------
        transcript:
            The STT transcript of the user's utterance.
        language_result:
            Optional dict from the Language_Engine (e.g.
            ``{"composition": "hinglish", "tokens": [...]}``).  Used to
            enrich the classification prompt with a language hint.

        Returns
        -------
        IntentResult
            The classified intent along with confidence and raw LLM output.
        """
        # Build the language-aware classification prompt.
        lang_hint = ""
        language_hint: str | None = None
        if language_result and isinstance(language_result, dict):
            composition = language_result.get("composition")
            if composition:
                lang_hint = f" [language: {composition}]"
                language_hint = str(composition)

        # ------------------------------------------------------------------
        # FAST-PATH: keyword/regex Mac-command detection.
        #
        # Runs BEFORE the (possibly stubbed) LLM so "open WhatsApp",
        # "whatsapp kholo", "spotify chalu karo", "what's on my screen",
        # "screen pe kya hai", etc. are always routed to MAC_COMMAND
        # regardless of the classifier backend. Falls through to the LLM
        # when nothing matches.
        # ------------------------------------------------------------------
        if (
            _is_screen_analysis_request(transcript)
            or _is_open_app_request(transcript)
            or _is_mac_action_request(transcript)
        ):
            logger.debug(
                "IntentRouter.classify: fast-path matched MAC_COMMAND for %r",
                transcript[:40],
            )
            return IntentResult(
                intent=Intent.MAC_COMMAND,
                confidence=1.0,
                raw_label="mac_command (fast-path)",
                language_hint=language_hint,
            )

        prompt = (
            f"{_CLASSIFY_SYSTEM_PROMPT}\n\n"
            f"User{lang_hint}: {transcript}"
        )

        try:
            # Offload to thread so a blocking real LLM backend does not
            # stall the async event loop.
            import asyncio
            result = await asyncio.to_thread(self._llm.invoke, prompt)

            # Extract the raw label string from the provider response.
            if isinstance(result, dict):
                raw = str(
                    result.get("output", result.get("intent", result.get("input", "")))
                ).strip().lower()
            else:
                raw = str(result).strip().lower()

            intent = self._parse_intent(raw)
            logger.debug(
                "IntentRouter.classify: transcript=%r → intent=%s (raw=%r)",
                transcript[:40], intent.value, raw[:40],
            )
            return IntentResult(
                intent=intent,
                confidence=1.0,
                raw_label=raw,
                language_hint=language_hint,
            )

        except Exception as exc:
            logger.warning(
                "IntentRouter.classify failed (%r) — defaulting to CHAT", exc
            )
            return IntentResult(
                intent=Intent.CHAT,
                confidence=0.0,
                raw_label="",
                language_hint=language_hint,
            )

    async def route(
        self,
        intent_result: IntentResult,
        turn_context: TurnContext,
    ) -> AsyncGenerator[str, None]:
        """
        Route the turn to the owning capability subsystem and return an async
        generator that yields response chunks.

        For side-effecting intents (mac_command, schedule, task,
        run_automation) the DialogueManager gate is checked first.  If
        required slots are missing the returned generator yields a
        clarification prompt rather than executing the capability (Req 23.1).

        For read-only / conversational intents the capability handler is
        returned directly without any slot check.

        Parameters
        ----------
        intent_result:
            The :class:`IntentResult` returned by :py:meth:`classify`.
        turn_context:
            The current :class:`~core.orchestrator.orchestrator.TurnContext`.

        Yields
        ------
        str
            Response chunks / tokens from the capability handler.
        """
        intent = intent_result.intent
        handler = _HANDLER_MAP.get(intent, _unknown_handler)

        if intent in _SIDE_EFFECTING_INTENTS:
            needed_slots = _REQUIRED_SLOTS.get(intent, [])
            slot_result: SlotFillResult = self._dialogue_manager.assess(
                turn_context.transcript,
                needed_slots,
            )

            if not slot_result.sufficient:
                # Defer to the dialogue gate — try memory first (Req 23.2)
                missing = slot_result.missing
                fill_result = self._dialogue_manager.fill_from_memory(missing)

                if fill_result.still_missing:
                    # Still missing after memory resolution — ask user
                    still_missing_str = ", ".join(fill_result.still_missing)
                    logger.debug(
                        "IntentRouter.route: intent=%s missing slots=%r (after memory fill)",
                        intent.value, fill_result.still_missing,
                    )
                    yield (
                        f"I need a bit more information before I can do that. "
                        f"Missing: {still_missing_str}. Could you provide those details?"
                    )
                    return
                else:
                    # All slots resolved from memory — mark them and fall through
                    for slot_name, value in fill_result.resolved.items():
                        self._dialogue_manager.mark_resolved(slot_name, value)
                    logger.debug(
                        "IntentRouter.route: intent=%s all missing slots resolved from memory",
                        intent.value,
                    )
                    # Fall through to execute the capability handler

        # Execute the capability handler.
        async for chunk in handler(turn_context):
            yield chunk

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_intent(raw: str) -> Intent:
        """
        Map a raw LLM output string to an :class:`~core.orchestrator.orchestrator.Intent`.

        Performs a fuzzy prefix match so that extra punctuation or whitespace
        from the LLM does not cause a hard failure.
        """
        # Strip surrounding quotes, punctuation, and whitespace.
        cleaned = raw.strip("\"'.,!? \t\n").lower()

        _ALIAS_MAP: dict[str, Intent] = {
            "chat": Intent.CHAT,
            "recall": Intent.RECALL,
            "remember": Intent.REMEMBER,
            "read_aloud": Intent.READ_ALOUD,
            "readaloud": Intent.READ_ALOUD,
            "read aloud": Intent.READ_ALOUD,
            "mac_command": Intent.MAC_COMMAND,
            "maccommand": Intent.MAC_COMMAND,
            "mac command": Intent.MAC_COMMAND,
            "run_automation": Intent.RUN_AUTOMATION,
            "runautomation": Intent.RUN_AUTOMATION,
            "run automation": Intent.RUN_AUTOMATION,
            "image": Intent.IMAGE,
            "schedule": Intent.SCHEDULE,
            "task": Intent.TASK,
            "meta": Intent.META,
            "unknown": Intent.UNKNOWN,
        }

        if cleaned in _ALIAS_MAP:
            return _ALIAS_MAP[cleaned]

        # Prefix match fallback — pick the first alias whose key starts with
        # the cleaned string or that the cleaned string starts with.
        for key, intent_val in _ALIAS_MAP.items():
            if cleaned.startswith(key) or key.startswith(cleaned):
                return intent_val

        return Intent.CHAT
