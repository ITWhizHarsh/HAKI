"""Focused unit tests for session-owned voice turn state and frame ordering.

Validates: Requirements 4.1, 4.6, 4.8, 5.8
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from core.voice.frames import TypedVoiceFrame, VoiceFrameMetadata, VoiceFrameType
from core.voice.session import (
    FrameOrderingError,
    InvalidTurnTransition,
    LateFrameRejected,
    TurnQueueName,
    TurnState,
    VoiceQueueLimits,
    VoiceSession,
)


def _frame(session: VoiceSession, turn_id, sequence: int, *, generation: int | None = None):
    return TypedVoiceFrame(
        frame_type=VoiceFrameType.TRANSCRIPTION,
        metadata=VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=sequence,
            cancellation_generation=(
                session.cancellation_generation if generation is None else generation
            ),
        ),
        payload=f"frame-{sequence}",
    )


@pytest.mark.asyncio
async def test_turn_state_machine_allows_only_linear_progression_and_terminal_states() -> None:
    """A turn can progress linearly, fail/cancel from active work, and never reopen."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    with pytest.raises(InvalidTurnTransition, match="capturing to reasoning"):
        await session.turns.transition(turn_id, TurnState.REASONING)

    for state in (
        TurnState.PARTIAL,
        TurnState.FINAL_PENDING_SILENCE,
        TurnState.REASONING,
        TurnState.SYNTHESIZING,
        TurnState.PLAYING,
        TurnState.COMPLETED,
    ):
        assert await session.turns.transition(turn_id, state) is state

    with pytest.raises(InvalidTurnTransition, match="completed to capturing"):
        await session.turns.transition(turn_id, TurnState.CAPTURING)

    direct_final_turn = uuid4()
    await session.start_turn(direct_final_turn)
    assert (
        await session.turns.transition(direct_final_turn, TurnState.FINAL_PENDING_SILENCE)
        is TurnState.FINAL_PENDING_SILENCE
    )
    assert await session.turns.transition(direct_final_turn, TurnState.CANCELLED) is TurnState.CANCELLED


@pytest.mark.asyncio
async def test_interleaved_turns_have_independent_locks_and_sequences() -> None:
    """Frames from separate turns may interleave without sharing order state or queues."""
    session = VoiceSession(uuid4())
    first_turn, second_turn = uuid4(), uuid4()
    first, second = await asyncio.gather(session.start_turn(first_turn), session.start_turn(second_turn))

    await asyncio.gather(
        session.accept_frame(_frame(session, first_turn, 100), queue=TurnQueueName.CONTROL),
        session.accept_frame(_frame(session, second_turn, 7), queue=TurnQueueName.CONTROL),
        session.accept_frame(_frame(session, first_turn, 101), queue=TurnQueueName.CONTROL),
        session.accept_frame(_frame(session, second_turn, 8), queue=TurnQueueName.CONTROL),
    )

    assert [first.queues.control.get_nowait().metadata.sequence for _ in range(2)] == [100, 101]
    assert [second.queues.control.get_nowait().metadata.sequence for _ in range(2)] == [7, 8]


@pytest.mark.asyncio
async def test_terminal_turn_and_stale_generation_reject_late_frames() -> None:
    """Cancellation makes queued producers stale and terminal turns reject every later frame."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    await session.accept_frame(_frame(session, turn_id, 0), queue=TurnQueueName.CONTROL)
    previous_generation = session.cancellation_generation

    assert await session.cancel_turn(turn_id) == previous_generation + 1
    with pytest.raises(LateFrameRejected, match="terminal turn rejects late frames"):
        await session.accept_frame(
            _frame(session, turn_id, 1, generation=previous_generation),
            queue=TurnQueueName.CONTROL,
        )


@pytest.mark.asyncio
async def test_per_turn_ordering_and_bounded_partial_coalescing_are_preserved() -> None:
    """A turn requires the next sequence while partial UI work stays bounded/latest-wins."""
    session = VoiceSession(uuid4(), queue_limits=VoiceQueueLimits(partial=1))
    turn_id = uuid4()
    record = await session.start_turn(turn_id)

    await session.accept_frame(_frame(session, turn_id, 20), queue=TurnQueueName.PARTIAL)
    with pytest.raises(FrameOrderingError, match="next sequence"):
        await session.accept_frame(_frame(session, turn_id, 22), queue=TurnQueueName.PARTIAL)
    await session.accept_frame(_frame(session, turn_id, 21), queue=TurnQueueName.PARTIAL)

    assert record.queues.partial.qsize() == 1
    assert record.queues.partial.get_nowait().metadata.sequence == 21


@pytest.mark.asyncio
async def test_context_is_session_owned_and_exposes_only_confirmed_assistant_sentences() -> None:
    """Voice context contains accepted user text plus renderer-confirmed assistant text only."""
    session = VoiceSession(uuid4())
    user_turn_id, assistant_turn_id = uuid4(), uuid4()
    await session.append_user_turn(turn_id=user_turn_id, text="Kal meeting reschedule kar do")
    await session.confirm_played_sentence(
        turn_id=assistant_turn_id,
        sentence_id=uuid4(),
        text="I will reschedule it.",
        playback_completed_monotonic_ns=123,
    )

    context = await session.context_snapshot()
    assert [message.role for message in context.messages] == ["user", "assistant"]
    assert [message.text for message in context.messages] == [
        "Kal meeting reschedule kar do",
        "I will reschedule it.",
    ]
    assert [sentence.text for sentence in context.assistant_sentences] == ["I will reschedule it."]
