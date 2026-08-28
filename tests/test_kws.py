"""Unit tests for the FSM and KWS engine using synthetic tensors (no mic)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audio.features import extract_mfcc
from config import (
    FRAME_SAMPLES,
    KWS_HOP_SAMPLES,
    KWS_MAX_INFERENCE_MS,
    KWS_WAKE_THRESHOLD,
    KWS_WINDOW_SAMPLES,
    MFCC_FRAMES_PER_KWS_WINDOW,
    MFCC_N_MFCC,
    SAMPLE_RATE,
)
from core.state import (
    InvalidStateTransition,
    StateChangeEvent,
    StateMachine,
    SystemState,
)
from models.kws_engine import KWSEngine
from onnx_stub import write_kws_model_stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mfcc_filled(value: float) -> np.ndarray:
    return np.full((1, MFCC_N_MFCC, MFCC_FRAMES_PER_KWS_WINDOW), value, dtype=np.float32)


def _sine(n: int = KWS_WINDOW_SAMPLES, freq: float = 440.0, amplitude: float = 0.4) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = amplitude * np.sin(2.0 * np.pi * freq * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


class ScriptedSession:
    """Minimal ONNX-like session that returns a scripted score per call."""

    def __init__(self, scores: list[float], input_shape=(1, 40, 8)) -> None:
        self.scores = list(scores)
        self.calls: list[np.ndarray] = []
        self._i = 0
        self._input_shape = input_shape

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=list(self._input_shape))]

    def get_outputs(self):
        return [SimpleNamespace(name="output", shape=[1, 1])]

    def run(self, _output_names, feeds):
        self.calls.append(np.array(feeds["input"], copy=True))
        idx = min(self._i, len(self.scores) - 1)
        self._i += 1
        score = self.scores[idx]
        return [np.array([[score]], dtype=np.float32)]


@pytest.fixture
def kws_model_path(tmp_path):
    return write_kws_model_stub(tmp_path / "kws_model.onnx")


@pytest.fixture
def fsm():
    return StateMachine()


# ---------------------------------------------------------------------------
# Finite state machine
# ---------------------------------------------------------------------------
def test_fsm_starts_idle_listening():
    fsm = StateMachine()
    assert fsm.state is SystemState.IDLE_LISTENING


def test_fsm_happy_path_and_prompt_aliases():
    fsm = StateMachine()
    assert SystemState.RECORDING_UTTERANCE is SystemState.RECORDING_SPEECH
    assert SystemState.PROCESSING_LOCAL_INFERENCE is SystemState.PROCESSING_INTENT

    fsm.transition(SystemState.KEYWORD_DETECTED, trigger="kws_wake")
    fsm.transition(SystemState.RECORDING_SPEECH, trigger="start_record")
    fsm.transition(SystemState.PROCESSING_INTENT, trigger="vad_end")
    fsm.transition(SystemState.IDLE_LISTENING, trigger="done")
    assert fsm.state is SystemState.IDLE_LISTENING
    assert [e.current for e in fsm.history] == [
        SystemState.KEYWORD_DETECTED,
        SystemState.RECORDING_SPEECH,
        SystemState.PROCESSING_INTENT,
        SystemState.IDLE_LISTENING,
    ]


def test_fsm_invalid_transition_raises():
    fsm = StateMachine()
    with pytest.raises(InvalidStateTransition, match="IDLE_LISTENING -> RECORDING_SPEECH"):
        fsm.transition(SystemState.RECORDING_SPEECH)
    assert fsm.state is SystemState.IDLE_LISTENING


def test_fsm_same_state_is_noop():
    fsm = StateMachine()
    assert fsm.transition(SystemState.IDLE_LISTENING) is None
    assert fsm.history == ()


def test_fsm_reset_from_any_state():
    fsm = StateMachine()
    fsm.transition(SystemState.KEYWORD_DETECTED)
    fsm.transition(SystemState.RECORDING_SPEECH)
    event = fsm.reset(trigger="abort")
    assert event is not None
    assert event.current is SystemState.IDLE_LISTENING
    assert event.trigger == "abort"
    assert fsm.reset() is None


def test_fsm_event_hooks_fire_in_order():
    fsm = StateMachine()
    order: list[str] = []
    payloads: list[StateChangeEvent] = []

    fsm.on_exit(SystemState.IDLE_LISTENING, lambda e: order.append("exit_idle"))
    fsm.on_enter(SystemState.KEYWORD_DETECTED, lambda e: order.append("enter_kw"))
    fsm.on_transition(lambda e: order.append("trans"))
    fsm.on_transition(payloads.append)

    event = fsm.transition(
        SystemState.KEYWORD_DETECTED, trigger="kws_wake", payload={"score": 0.91}
    )
    assert order == ["exit_idle", "trans", "enter_kw"]
    assert event is not None
    assert event.previous is SystemState.IDLE_LISTENING
    assert event.current is SystemState.KEYWORD_DETECTED
    assert event.trigger == "kws_wake"
    assert event.payload["score"] == 0.91
    assert payloads[0] is event


def test_can_transition():
    fsm = StateMachine()
    assert fsm.can_transition(SystemState.KEYWORD_DETECTED)
    assert fsm.can_transition(SystemState.IDLE_LISTENING)
    assert not fsm.can_transition(SystemState.PROCESSING_INTENT)


# ---------------------------------------------------------------------------
# KWS scoring (synthetic tensors + ONNX stub)
# ---------------------------------------------------------------------------
def test_missing_kws_model_raises(tmp_path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError, match="kws_model"):
        KWSEngine(model_path=missing)


def test_stub_onnx_io(kws_model_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(kws_model_path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["output"]


def test_synthetic_tensor_scores_match_mean(kws_model_path):
    engine = KWSEngine(model_path=kws_model_path)
    low = engine.process_features(_mfcc_filled(0.0))
    mid = engine.process_features(_mfcc_filled(0.42))
    high = engine.process_features(_mfcc_filled(0.95))
    assert low == pytest.approx(0.0, abs=1e-5)
    assert mid == pytest.approx(0.42, abs=1e-5)
    assert high == pytest.approx(0.95, abs=1e-5)
    for score in (low, mid, high):
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


def test_scores_clipped_to_unit_interval():
    session = ScriptedSession([ -3.0, 0.5, 8.0 ])
    engine = KWSEngine(session=session)
    assert engine.process_features(_mfcc_filled(0.0)) == pytest.approx(_sigmoid_ref(-3.0))
    assert engine.process_features(_mfcc_filled(0.0)) == pytest.approx(0.5)
    assert engine.process_features(_mfcc_filled(0.0)) == pytest.approx(_sigmoid_ref(8.0))
    assert 0.0 <= engine.last_score <= 1.0


def _sigmoid_ref(x: float) -> float:
    import math

    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def test_threshold_from_config_is_085():
    assert KWS_WAKE_THRESHOLD == 0.85


def test_score_below_threshold_stays_idle(fsm):
    session = ScriptedSession([0.84])
    engine = KWSEngine(session=session, fsm=fsm, threshold=KWS_WAKE_THRESHOLD)
    score = engine.process_features(_mfcc_filled(0.1))
    assert score == pytest.approx(0.84)
    assert score < KWS_WAKE_THRESHOLD
    assert fsm.state is SystemState.IDLE_LISTENING
    assert engine.detected is False


def test_score_at_threshold_triggers_keyword_detected(fsm):
    session = ScriptedSession([KWS_WAKE_THRESHOLD])
    engine = KWSEngine(session=session, fsm=fsm)
    events: list[StateChangeEvent] = []
    fsm.on_enter(SystemState.KEYWORD_DETECTED, events.append)

    score = engine.process_features(_mfcc_filled(1.0))
    assert score == pytest.approx(KWS_WAKE_THRESHOLD)
    assert engine.detected is True
    assert fsm.state is SystemState.KEYWORD_DETECTED
    assert events[0].trigger == "kws_wake"
    assert events[0].payload["score"] == pytest.approx(KWS_WAKE_THRESHOLD)


def test_high_stub_tensor_crosses_threshold(kws_model_path, fsm):
    engine = KWSEngine(model_path=kws_model_path, fsm=fsm)
    engine.process_features(_mfcc_filled(0.10))
    assert fsm.state is SystemState.IDLE_LISTENING
    engine.process_features(_mfcc_filled(0.90))
    assert engine.last_score == pytest.approx(0.90, abs=1e-5)
    assert engine.running_score == pytest.approx(0.90, abs=1e-5)
    assert fsm.state is SystemState.KEYWORD_DETECTED


def test_does_not_retrigger_once_already_detected(fsm):
    session = ScriptedSession([0.99, 0.99])
    engine = KWSEngine(session=session, fsm=fsm)
    engine.process_features(_mfcc_filled(1.0))
    engine.process_features(_mfcc_filled(1.0))
    assert fsm.state is SystemState.KEYWORD_DETECTED
    assert len(fsm.history) == 1


def test_process_window_uses_mfcc_shape(kws_model_path):
    captured: list[tuple] = []
    engine = KWSEngine(model_path=kws_model_path)
    inner = engine._session.run

    def wrapped(output_names, feeds):
        captured.append(feeds["input"].shape)
        return inner(output_names, feeds)

    engine._session.run = wrapped  # type: ignore[method-assign]
    engine.process_window(_sine(KWS_WINDOW_SAMPLES))
    assert captured == [(1, MFCC_N_MFCC, MFCC_FRAMES_PER_KWS_WINDOW)]
    assert captured[0] == extract_mfcc(_sine(KWS_WINDOW_SAMPLES)).shape


def test_sliding_window_runs_after_100ms_of_pcm():
    session = ScriptedSession([0.2, 0.3, 0.4])
    engine = KWSEngine(session=session, hop_samples=KWS_HOP_SAMPLES)
    frame = _sine(FRAME_SAMPLES)
    score = 0.0
    # 3 frames = 90 ms < 100 ms window; no inference yet.
    for _ in range(3):
        score = engine.process_pcm(frame)
    assert session.calls == []
    assert score == 0.0
    # 4th frame → 120 ms, first 100 ms window fires.
    score = engine.process_pcm(frame)
    assert len(session.calls) == 1
    assert session.calls[0].shape == (1, 40, 8)
    assert score == pytest.approx(0.2)


def test_inference_latency_under_budget(kws_model_path):
    engine = KWSEngine(model_path=kws_model_path)
    feats = _mfcc_filled(0.3)
    for _ in range(8):
        engine.process_features(feats)
    assert engine.last_inference_ms >= 0.0
    assert engine.last_inference_ms < KWS_MAX_INFERENCE_MS


def test_multiclass_softmax_uses_wake_index():
    class VecSession:
        def get_inputs(self):
            return [SimpleNamespace(name="logits", shape=[1, 3])]

        def get_outputs(self):
            return [SimpleNamespace(name="logits", shape=[1, 3])]

        def run(self, _names, _feeds):
            return [np.array([[1.0, 4.0, 0.5]], dtype=np.float32)]

    engine = KWSEngine(session=VecSession(), wake_index=1)
    score = engine.process_features(_mfcc_filled(0.0))
    # softmax([1, 4, 0.5]) at index 1 is the peak, well above 0.85
    assert 0.0 <= score <= 1.0
    assert score > 0.85


def test_engine_reset_clears_buffer_and_scores(fsm):
    session = ScriptedSession([0.99])
    engine = KWSEngine(session=session, fsm=fsm)
    engine.process_features(_mfcc_filled(1.0))
    engine.reset(reset_fsm=True)
    assert engine.last_score == 0.0
    assert engine.running_score == 0.0
    assert fsm.state is SystemState.IDLE_LISTENING
