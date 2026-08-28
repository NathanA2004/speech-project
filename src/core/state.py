"""Thread-safe, event-driven finite state machine for the local speech agent.

States follow the pipeline in PROMPT.md, using the Prompt 3 names
``RECORDING_SPEECH`` / ``PROCESSING_INTENT`` as canonical values (with
architecture aliases ``RECORDING_UTTERANCE`` / ``PROCESSING_LOCAL_INFERENCE``).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SystemState(str, Enum):
    """Lifecycle of the offline listen → wake → record → infer loop."""

    IDLE_LISTENING = "IDLE_LISTENING"
    KEYWORD_DETECTED = "KEYWORD_DETECTED"
    RECORDING_SPEECH = "RECORDING_SPEECH"
    PROCESSING_INTENT = "PROCESSING_INTENT"
    EXECUTING_ACTION = "EXECUTING_ACTION"

    # Architecture aliases from PROMPT.md (same members, not extra states).
    RECORDING_UTTERANCE = "RECORDING_SPEECH"
    PROCESSING_LOCAL_INFERENCE = "PROCESSING_INTENT"


# Allowed directed edges. Any state may also reset to IDLE_LISTENING.
TRANSITIONS: dict[SystemState, frozenset[SystemState]] = {
    SystemState.IDLE_LISTENING: frozenset({SystemState.KEYWORD_DETECTED}),
    SystemState.KEYWORD_DETECTED: frozenset(
        {SystemState.RECORDING_SPEECH, SystemState.IDLE_LISTENING}
    ),
    SystemState.RECORDING_SPEECH: frozenset(
        {SystemState.PROCESSING_INTENT, SystemState.IDLE_LISTENING}
    ),
    SystemState.PROCESSING_INTENT: frozenset(
        {SystemState.EXECUTING_ACTION, SystemState.IDLE_LISTENING}
    ),
    SystemState.EXECUTING_ACTION: frozenset({SystemState.IDLE_LISTENING}),
}


class InvalidStateTransition(ValueError):
    """Raised when a requested FSM edge is not in ``TRANSITIONS``."""

    def __init__(self, previous: SystemState, target: SystemState) -> None:
        self.previous = previous
        self.target = target
        super().__init__(f"invalid state transition: {previous.value} -> {target.value}")


@dataclass(frozen=True)
class StateChangeEvent:
    """Payload delivered to enter / exit / transition hooks."""

    previous: SystemState
    current: SystemState
    trigger: str
    timestamp: float
    payload: Mapping[str, Any] = field(default_factory=dict)


TransitionHook = Callable[[StateChangeEvent], None]


class StateMachine:
    """Event-driven FSM with per-state and global hooks.

    Callbacks run *after* the lock is released so a hook may transition again
    (re-entrant via ``RLock``). Invalid edges raise ``InvalidStateTransition``.
    """

    def __init__(
        self,
        initial: SystemState = SystemState.IDLE_LISTENING,
        *,
        history_size: int = 64,
    ) -> None:
        self._lock = threading.RLock()
        self._state = SystemState(initial)
        self._history: deque[StateChangeEvent] = deque(maxlen=max(1, int(history_size)))
        self._on_enter: dict[SystemState, list[TransitionHook]] = {
            s: [] for s in SystemState
        }
        self._on_exit: dict[SystemState, list[TransitionHook]] = {
            s: [] for s in SystemState
        }
        self._on_transition: list[TransitionHook] = []

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    @property
    def history(self) -> tuple[StateChangeEvent, ...]:
        with self._lock:
            return tuple(self._history)

    def can_transition(self, target: SystemState) -> bool:
        target = SystemState(target)
        with self._lock:
            if target is self._state:
                return True
            return target in TRANSITIONS[self._state]

    def on_enter(self, state: SystemState, hook: TransitionHook) -> None:
        with self._lock:
            self._on_enter[SystemState(state)].append(hook)

    def on_exit(self, state: SystemState, hook: TransitionHook) -> None:
        with self._lock:
            self._on_exit[SystemState(state)].append(hook)

    def on_transition(self, hook: TransitionHook) -> None:
        """Subscribe to every successful state change."""
        with self._lock:
            self._on_transition.append(hook)

    def transition(
        self,
        target: SystemState,
        *,
        trigger: str = "",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[StateChangeEvent]:
        """Move to ``target``. No-op (returns ``None``) if already there."""
        target = SystemState(target)
        with self._lock:
            previous = self._state
            if target is previous:
                return None
            if target not in TRANSITIONS[previous]:
                raise InvalidStateTransition(previous, target)
            event = StateChangeEvent(
                previous=previous,
                current=target,
                trigger=str(trigger),
                timestamp=time.perf_counter(),
                payload=dict(payload or {}),
            )
            self._state = target
            self._history.append(event)
            exit_hooks = list(self._on_exit[previous])
            trans_hooks = list(self._on_transition)
            enter_hooks = list(self._on_enter[target])

        for hook in exit_hooks:
            hook(event)
        for hook in trans_hooks:
            hook(event)
        for hook in enter_hooks:
            hook(event)
        return event

    def reset(
        self,
        *,
        trigger: str = "reset",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[StateChangeEvent]:
        """Return to ``IDLE_LISTENING`` from any state (always a legal edge)."""
        with self._lock:
            if self._state is SystemState.IDLE_LISTENING:
                return None
        return self.transition(
            SystemState.IDLE_LISTENING, trigger=trigger, payload=payload
        )

    def __repr__(self) -> str:
        return f"StateMachine(state={self.state.value})"
