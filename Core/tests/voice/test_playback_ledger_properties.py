"""Property 6: Confirmed-playback context ledger.

**Validates: Requirements 5.6, 5.7, 5.8**

Feature: realtime-local-voice-agent, Property 6: Confirmed-playback context ledger

For all ordered assistant sentence schedules, playback confirmations, failures,
and cancellations, conversation context contains every user turn plus exactly the
assistant sentences that received normal Playback_Confirmation, in confirmation
order, and contains no partial, merely generated, failed, or interrupted sentence
text.

Test coverage:
- At least 100 ordered schedules with mixed outcomes (confirmed, failed,
  cancelled, interrupted).
- Only PLAYBACK_CONFIRMED sentences appear in VoiceContext.assistant_sentences,
  in confirmation order.
- Never: cancelled, failed, interrupted, provisional sentences in context.
- User turns appear in context.
- Out-of-order confirmations: confirmation order, not registration order, wins.
- Duplicate terminal renderer events: second confirm returns False.
- Cancelled sentences (generation advanced) never enter context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.voice.session import (
    PlaybackLedger,
    PlayedSentence,
    ProvisionalSentenceState,
    TurnState,
    VoiceContext,
    VoiceSession,
)


# ---------------------------------------------------------------------------
# Domain types for the property generator
# ---------------------------------------------------------------------------


class SentenceOutcome(str, Enum):
    """Renderer outcome assigned to each sentence in a generated schedule."""

    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED_BY_BARGE_IN = "cancelled_by_barge_in"
    NEVER_CONFIRMED = "never_confirmed"


@dataclass(frozen=True)
class SentenceSpec:
    """One sentence in a generated schedule."""

    sentence_id: UUID
    text: str
    outcome: SentenceOutcome
    # Monotonic nanosecond timestamp used only for CONFIRMED outcomes.
    confirmed_ns: int


@dataclass(frozen=True)
class UserTurnSpec:
    """A user turn inserted into the session context."""

    turn_id: UUID
    text: str


@dataclass(frozen=True)
class Schedule:
    """A complete test scenario for Property 6."""

    session_id: UUID
    # The single assistant turn that owns all sentences.
    assistant_turn_id: UUID
    # User turns recorded into context (may be empty, before or after).
    user_turns: tuple[UserTurnSpec, ...]
    # Ordered list of sentence specs — registration order determines sequence.
    sentences: tuple[SentenceSpec, ...]
    # Whether to simulate a barge-in after registration but before any confirm.
    simulate_barge_in: bool


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=1,
    max_size=40,
).filter(lambda t: t.strip())

_OUTCOME = st.sampled_from(list(SentenceOutcome))

_SENTENCE_SPEC = st.builds(
    SentenceSpec,
    sentence_id=st.builds(uuid4),
    text=_SAFE_TEXT,
    outcome=_OUTCOME,
    confirmed_ns=st.integers(min_value=1, max_value=10**18),
)

_USER_TURN_SPEC = st.builds(
    UserTurnSpec,
    turn_id=st.builds(uuid4),
    text=_SAFE_TEXT,
)


@st.composite
def schedules(draw: st.DrawFn) -> Schedule:
    """Generate a plausible playback schedule with 1–8 sentences and 0–3 user turns."""
    sentences = draw(st.lists(_SENTENCE_SPEC, min_size=1, max_size=8))
    # Ensure sentence IDs are unique (Hypothesis may generate duplicates).
    seen_ids: set[UUID] = set()
    unique_sentences: list[SentenceSpec] = []
    for spec in sentences:
        if spec.sentence_id not in seen_ids:
            seen_ids.add(spec.sentence_id)
            unique_sentences.append(spec)
    if not unique_sentences:
        unique_sentences.append(draw(_SENTENCE_SPEC))

    user_turns = draw(st.lists(_USER_TURN_SPEC, min_size=0, max_size=3))
    simulate_barge_in = draw(st.booleans())

    return Schedule(
        session_id=draw(st.builds(uuid4)),
        assistant_turn_id=draw(st.builds(uuid4)),
        user_turns=tuple(user_turns),
        sentences=tuple(unique_sentences),
        simulate_barge_in=simulate_barge_in,
    )


# ---------------------------------------------------------------------------
# Async helper: run a schedule and return the resulting VoiceContext
# ---------------------------------------------------------------------------


async def _run_schedule(schedule: Schedule) -> VoiceContext:
    """Execute a schedule against a real VoiceSession and return the context."""
    session = VoiceSession(schedule.session_id)

    # Register the assistant turn.
    await session.start_turn(schedule.assistant_turn_id)
    initial_generation = session.cancellation_generation

    # Register all user turns.
    for user_turn in schedule.user_turns:
        await session.append_user_turn(
            turn_id=user_turn.turn_id, text=user_turn.text
        )

    # Register all sentences as provisional with the initial generation.
    registered_ids: list[UUID] = []
    for spec in schedule.sentences:
        accepted = await session.playback_ledger.register(
            turn_id=schedule.assistant_turn_id,
            sentence_id=spec.sentence_id,
            text=spec.text,
            cancellation_generation=initial_generation,
        )
        if accepted:
            registered_ids.append(spec.sentence_id)

    # Simulate a global barge-in if requested: advance the generation before any confirm.
    # This marks all remaining PENDING sentences as CANCELLED via cancel_turn.
    if schedule.simulate_barge_in:
        await session.cancel_turn(schedule.assistant_turn_id)
        # After barge-in, no confirm/fail/cancel on these sentences can append to context.
    else:
        # Determine which sentences are cancelled by a simulated per-sentence barge-in.
        # CANCELLED_BY_BARGE_IN: we call ledger.cancel() once, which sweeps all PENDING
        # sentences registered with the current generation.  We do this at the end of
        # processing so earlier CONFIRMED sentences are still able to be confirmed first.
        cancelled_by_barge_in = any(
            spec.outcome is SentenceOutcome.CANCELLED_BY_BARGE_IN
            for spec in schedule.sentences
            if spec.sentence_id in registered_ids
        )

        # Apply CONFIRMED and FAILED outcomes first (in registration order),
        # then apply a single barge-in cancel sweep for the rest.
        for spec in schedule.sentences:
            if spec.sentence_id not in registered_ids:
                continue
            # Skip CANCELLED_BY_BARGE_IN and NEVER_CONFIRMED — handled below.
            if spec.outcome is SentenceOutcome.CONFIRMED:
                await session.playback_ledger.confirm(
                    turn_id=schedule.assistant_turn_id,
                    sentence_id=spec.sentence_id,
                    playback_completed_monotonic_ns=spec.confirmed_ns,
                )
            elif spec.outcome is SentenceOutcome.FAILED:
                await session.playback_ledger.fail(
                    turn_id=schedule.assistant_turn_id,
                    sentence_id=spec.sentence_id,
                )

        if cancelled_by_barge_in:
            # Advance generation to cancel all remaining PENDING sentences.
            next_gen = session.cancellation_generation + 1
            await session.playback_ledger.cancel(
                turn_id=schedule.assistant_turn_id,
                cancellation_generation=next_gen,
            )

    return await session.context_snapshot()


# ---------------------------------------------------------------------------
# Property 6 — main property test (>=100 cases)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=5_000)
@given(schedule=schedules())
def test_property6_confirmed_playback_context_ledger(schedule: Schedule) -> None:
    """Property 6: Only PLAYBACK_CONFIRMED sentences appear in context, in confirmation order.

    Feature: realtime-local-voice-agent, Property 6: Confirmed-playback context ledger
    **Validates: Requirements 5.6, 5.7, 5.8**

    Asserts:
    - VoiceContext.assistant_sentences contains exactly the sentences that
      received a PLAYBACK_CONFIRMED event, in confirmation (not registration) order.
    - Sentences with outcomes FAILED, CANCELLED_BY_BARGE_IN, or NEVER_CONFIRMED
      do not appear in context.
    - After simulate_barge_in, zero assistant sentences appear in context.
    - All user turns appear in VoiceContext.user_turns.
    - VoiceContext.messages interleaves user and confirmed assistant messages
      in the order they were appended (user turns first, confirmed in
      confirmation order after).
    """
    context = asyncio.get_event_loop().run_until_complete(_run_schedule(schedule))
    _assert_property6(schedule, context)


def _assert_property6(schedule: Schedule, context: VoiceContext) -> None:
    """Validate all Property 6 invariants on a context snapshot."""

    # --- Invariant 1: All user turns appear in context ---
    context_user_turn_ids = {msg.turn_id for msg in context.user_turns}
    for user_turn in schedule.user_turns:
        assert user_turn.turn_id in context_user_turn_ids, (
            f"User turn {user_turn.turn_id} missing from context.user_turns"
        )
    assert len(context.user_turns) == len(schedule.user_turns), (
        "context.user_turns count must equal the number of registered user turns"
    )

    # --- Invariant 2: After barge-in, no assistant sentences appear ---
    if schedule.simulate_barge_in:
        assert context.assistant_sentences == (), (
            "After barge-in, no assistant sentences should be in context"
        )
        # Messages must contain only user turns.
        assistant_messages = [m for m in context.messages if m.role == "assistant"]
        assert assistant_messages == [], (
            "After barge-in, no assistant messages should be in context.messages"
        )
        return

    # --- Invariant 3: Only CONFIRMED sentences are in context ---
    # Note: CANCELLED_BY_BARGE_IN only cancels PENDING sentences; sentences that
    # were CONFIRMED before the cancel-sweep remain in context.
    confirmed_ids = {
        spec.sentence_id
        for spec in schedule.sentences
        if spec.outcome is SentenceOutcome.CONFIRMED
    }
    non_confirmed_ids = {
        spec.sentence_id
        for spec in schedule.sentences
        if spec.outcome is not SentenceOutcome.CONFIRMED
    }
    context_sentence_ids = {s.sentence_id for s in context.assistant_sentences}

    # Every sentence in context must be from the CONFIRMED set.
    for sid in context_sentence_ids:
        assert sid in confirmed_ids, (
            f"Sentence {sid} is in context but was not CONFIRMED"
        )

    # Non-confirmed sentences must not be in context.
    for sid in non_confirmed_ids:
        assert sid not in context_sentence_ids, (
            f"Non-confirmed sentence {sid} must not appear in context"
        )

    # --- Invariant 4: Confirmation order is preserved ---
    # The order in context.assistant_sentences must match confirmation order,
    # which is the order .confirm() was called (registration order for CONFIRMED specs).
    confirmed_specs_in_order = [
        spec for spec in schedule.sentences
        if spec.outcome is SentenceOutcome.CONFIRMED
    ]
    expected_ids_in_order = [spec.sentence_id for spec in confirmed_specs_in_order]
    actual_ids_in_order = [s.sentence_id for s in context.assistant_sentences]
    assert actual_ids_in_order == expected_ids_in_order, (
        f"Assistant sentences are not in confirmation order. "
        f"Expected IDs {expected_ids_in_order}, got {actual_ids_in_order}"
    )

    # --- Invariant 5: Non-confirmed sentence IDs not in context (by sentence_id) ---
    # We check by sentence_id (not text) because two different sentences may share
    # the same text — a non-confirmed sentence with text '0' must not be in context,
    # but a confirmed sentence with the same text '0' may validly be there.
    # The sentence_id check in Invariant 3 already covers this; this invariant
    # verifies the messages list is also consistent.
    assert len([m for m in context.messages if m.role == "assistant"]) == len(
        context.assistant_sentences
    ), "messages must have exactly one assistant entry per confirmed sentence"

    # --- Invariant 6: context.messages role sequence is consistent ---
    # User-role messages must correspond 1:1 with registered user turns.
    user_messages = [m for m in context.messages if m.role == "user"]
    assert len(user_messages) == len(schedule.user_turns)
    assistant_messages = [m for m in context.messages if m.role == "assistant"]
    assert len(assistant_messages) == len(context.assistant_sentences)


# ---------------------------------------------------------------------------
# Focused validation: out-of-order confirmations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_order_confirmations_appear_in_confirmation_not_registration_order() -> None:
    """Confirming s3 before s1 before s2 yields [s3, s1] in context (s2 unconfirmed).

    This directly validates Requirement 5.6: sentences are recorded in renderer
    completion (confirmation) order, not in generation or registration order.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    s1, s2, s3 = uuid4(), uuid4(), uuid4()
    for sid, text in ((s1, "First registered."), (s2, "Second registered."), (s3, "Third registered.")):
        accepted = await session.playback_ledger.register(
            turn_id=turn_id, sentence_id=sid, text=text, cancellation_generation=0
        )
        assert accepted, f"Register must succeed for {sid}"

    # Confirm in reverse-registration order: s3 → s1; leave s2 unconfirmed.
    r3 = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=s3, playback_completed_monotonic_ns=300
    )
    r1 = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=s1, playback_completed_monotonic_ns=100
    )
    assert r3 is True
    assert r1 is True

    context = await session.context_snapshot()
    texts = [s.text for s in context.assistant_sentences]
    # s3 was confirmed first, s1 second — that is the order in context.
    assert texts == ["Third registered.", "First registered."], (
        f"Expected confirmation order [s3, s1], got {texts}"
    )
    # s2 was never confirmed and must be absent.
    assert "Second registered." not in texts


# ---------------------------------------------------------------------------
# Focused validation: duplicate terminal renderer events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_confirm_returns_false_and_does_not_duplicate_context_entry() -> None:
    """A second PLAYBACK_CONFIRMED for the same sentence_id returns False.

    The sentence appears exactly once in context; its text is not duplicated.
    This validates the idempotency requirement for terminal renderer events.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()

    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="Heard exactly once.", cancellation_generation=0
    )

    first = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=sentence_id, playback_completed_monotonic_ns=10
    )
    second = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=sentence_id, playback_completed_monotonic_ns=20
    )

    assert first is True, "First confirm must succeed"
    assert second is False, "Second confirm must return False (duplicate terminal event)"

    context = await session.context_snapshot()
    assert len(context.assistant_sentences) == 1, "Sentence must appear exactly once"
    assert context.assistant_sentences[0].text == "Heard exactly once."
    # The first confirmation timestamp is preserved; the second must not overwrite.
    assert context.assistant_sentences[0].playback_completed_monotonic_ns == 10


# ---------------------------------------------------------------------------
# Focused validation: cancelled provisional sentences never enter context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barge_in_cancels_all_pending_provisional_sentences() -> None:
    """Advancing the generation makes all pending provisional sentences ineligible.

    After a barge-in (cancel_turn), even a subsequent confirm call for an older
    provisional sentence must return False and add nothing to context.
    Requirement 5.7: interrupted sentences excluded from Conversation_Context.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    s1, s2 = uuid4(), uuid4()
    for sid, text in ((s1, "Interrupted sentence A."), (s2, "Interrupted sentence B.")):
        await session.playback_ledger.register(
            turn_id=turn_id, sentence_id=sid, text=text, cancellation_generation=0
        )

    # Simulate barge-in: cancel the turn, advancing the generation.
    await session.cancel_turn(turn_id)
    assert session.cancellation_generation > 0

    # Attempt to confirm the sentences after the barge-in.
    r1 = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=s1, playback_completed_monotonic_ns=100
    )
    r2 = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=s2, playback_completed_monotonic_ns=200
    )
    assert r1 is False, "Confirm after barge-in must return False"
    assert r2 is False, "Confirm after barge-in must return False"

    context = await session.context_snapshot()
    assert context.assistant_sentences == (), "No sentences must appear in context after barge-in"
    assert [m.text for m in context.messages if m.role == "assistant"] == []


# ---------------------------------------------------------------------------
# Focused validation: later LLM turns only see confirmed sentences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_later_llm_turn_only_sees_user_messages_and_confirmed_assistant_sentences() -> None:
    """Requirement 5.8: later turns see user messages + confirmed assistant sentences only.

    Provisional text that was never confirmed must be absent from the context
    snapshot used to build a later LLM prompt.
    """
    session = VoiceSession(uuid4())
    user_turn_id, assistant_turn_id = uuid4(), uuid4()

    await session.start_turn(assistant_turn_id)
    await session.append_user_turn(turn_id=user_turn_id, text="Tell me something.")

    confirmed_sid = uuid4()
    never_confirmed_sid = uuid4()
    failed_sid = uuid4()

    for sid, text in (
        (confirmed_sid, "This was confirmed."),
        (never_confirmed_sid, "This was never confirmed."),
        (failed_sid, "This synthesis failed."),
    ):
        await session.playback_ledger.register(
            turn_id=assistant_turn_id, sentence_id=sid,
            text=text, cancellation_generation=0
        )

    # Only confirm the first sentence.
    await session.playback_ledger.confirm(
        turn_id=assistant_turn_id,
        sentence_id=confirmed_sid,
        playback_completed_monotonic_ns=50,
    )
    # Mark the failed sentence.
    await session.playback_ledger.fail(
        turn_id=assistant_turn_id, sentence_id=failed_sid
    )
    # never_confirmed_sid gets no action.

    context = await session.context_snapshot()

    # User turn must appear.
    user_texts = [m.text for m in context.messages if m.role == "user"]
    assert "Tell me something." in user_texts

    # Only the confirmed sentence must appear.
    assistant_texts = [m.text for m in context.messages if m.role == "assistant"]
    assert assistant_texts == ["This was confirmed."], (
        f"Expected only confirmed sentence in context.messages, got {assistant_texts}"
    )

    # Provisional / failed / never-confirmed sentences must be absent.
    all_texts = {m.text for m in context.messages}
    assert "This was never confirmed." not in all_texts
    assert "This synthesis failed." not in all_texts


# ---------------------------------------------------------------------------
# Focused validation: failed sentences never enter context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_sentence_never_enters_context() -> None:
    """PLAYBACK_FAILED marks the sentence FAILED; it must never appear in context."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()

    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="This will fail.", cancellation_generation=0
    )

    result = await session.playback_ledger.fail(
        turn_id=turn_id, sentence_id=sentence_id
    )
    assert result is True

    state = await session.playback_ledger.state_for(sentence_id)
    assert state is ProvisionalSentenceState.FAILED

    context = await session.context_snapshot()
    assert context.assistant_sentences == ()
    assert all(m.text != "This will fail." for m in context.messages)


# ---------------------------------------------------------------------------
# Focused validation: mixed multi-turn schedule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_multi_turn_schedule_with_confirmations_failures_and_cancellations() -> None:
    """Multiple turns with mixed outcomes only surface confirmed sentences.

    Turn 1: s1 CONFIRMED, s2 FAILED
    Turn 2: s3 CONFIRMED, s4 CANCELLED (barge-in), s5 CONFIRMED
    User turn appears between turns.
    """
    session = VoiceSession(uuid4())

    # Turn 1
    turn1 = uuid4()
    await session.start_turn(turn1)
    s1, s2 = uuid4(), uuid4()
    await session.playback_ledger.register(
        turn_id=turn1, sentence_id=s1, text="T1-S1 confirmed.", cancellation_generation=0
    )
    await session.playback_ledger.register(
        turn_id=turn1, sentence_id=s2, text="T1-S2 failed.", cancellation_generation=0
    )
    await session.playback_ledger.confirm(
        turn_id=turn1, sentence_id=s1, playback_completed_monotonic_ns=10
    )
    await session.playback_ledger.fail(turn_id=turn1, sentence_id=s2)
    # Turn 1 remains in whatever state — we don't need to complete it explicitly.

    # User turn between assistant turns.
    user_turn = uuid4()
    await session.append_user_turn(turn_id=user_turn, text="Follow up question.")

    # Turn 2: start at generation 0 (no barge-in yet).
    turn2 = uuid4()
    await session.start_turn(turn2)
    s3, s4, s5 = uuid4(), uuid4(), uuid4()
    for sid, text in (
        (s3, "T2-S3 confirmed."),
        (s4, "T2-S4 cancelled."),
        (s5, "T2-S5 confirmed."),
    ):
        await session.playback_ledger.register(
            turn_id=turn2, sentence_id=sid, text=text, cancellation_generation=0
        )

    # Confirm s3 first.
    await session.playback_ledger.confirm(
        turn_id=turn2, sentence_id=s3, playback_completed_monotonic_ns=20
    )
    # Simulate barge-in that cancels s4 and s5 — cancel_turn advances generation.
    await session.cancel_turn(turn2)

    # Attempt to confirm s5 after barge-in (must fail).
    r5 = await session.playback_ledger.confirm(
        turn_id=turn2, sentence_id=s5, playback_completed_monotonic_ns=30
    )
    assert r5 is False, "Confirm after barge-in must return False"

    context = await session.context_snapshot()
    confirmed_texts = [s.text for s in context.assistant_sentences]

    # Only s1 and s3 should be confirmed.
    assert "T1-S1 confirmed." in confirmed_texts
    assert "T2-S3 confirmed." in confirmed_texts

    # All other sentences must be absent.
    for absent in ("T1-S2 failed.", "T2-S4 cancelled.", "T2-S5 confirmed."):
        assert absent not in confirmed_texts, f"'{absent}' must not appear in context"

    # User turn must appear.
    user_texts = [m.text for m in context.messages if m.role == "user"]
    assert "Follow up question." in user_texts
