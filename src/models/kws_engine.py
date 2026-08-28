"""Offline Tier-1 keyword spotting via local ONNX Runtime.

Sliding-window inference on 100 ms MFCC tensors from ``audio.features``.
When the wake score crosses ``KWS_WAKE_THRESHOLD``, the attached FSM moves
``IDLE_LISTENING`` → ``KEYWORD_DETECTED``. No network, no cloud APIs.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from audio.features import extract_mfcc, pcm_to_float32
from config import (
    KWS_HOP_SAMPLES,
    KWS_MAX_INFERENCE_MS,
    KWS_MODEL_PATH,
    KWS_SCORE_EMA_ALPHA,
    KWS_WAKE_INDEX,
    KWS_WAKE_THRESHOLD,
    KWS_WINDOW_SAMPLES,
    MFCC_FRAMES_PER_KWS_WINDOW,
    MFCC_N_MFCC,
)
from core.state import StateMachine, SystemState

_FEATURE_INPUT_NAMES = frozenset({"input", "mfcc", "features", "feats", "audio", "data"})


def _positive_dims(shape: object, fallback: tuple[int, ...]) -> tuple[int, ...]:
    dims: list[int] = []
    raw = list(shape) if shape is not None else []
    for i, dim in enumerate(raw):
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif i < len(fallback):
            dims.append(fallback[i])
        else:
            dims.append(1)
    return tuple(dims) if dims else fallback


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-x))


class KWSEngine:
    """100 ms sliding-window wake-word detector (CPU ONNX Runtime).

    Parameters
    ----------
    model_path
        Local ``kws_model.onnx``. Ignored when ``session`` is injected.
    fsm
        Optional state machine. A score at or above ``threshold`` while
        idle triggers ``KEYWORD_DETECTED``.
    session
        Injected ``InferenceSession`` (tests / synthetic tensors).
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        fsm: Optional[StateMachine] = None,
        session: Optional[ort.InferenceSession] = None,
        threshold: float = KWS_WAKE_THRESHOLD,
        window_samples: int = KWS_WINDOW_SAMPLES,
        hop_samples: int = KWS_HOP_SAMPLES,
        wake_index: int = KWS_WAKE_INDEX,
        ema_alpha: float = KWS_SCORE_EMA_ALPHA,
        max_inference_ms: float = KWS_MAX_INFERENCE_MS,
    ) -> None:
        if hop_samples <= 0:
            raise ValueError("hop_samples must be positive")
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        self.model_path = Path(model_path) if model_path is not None else KWS_MODEL_PATH
        self.fsm = fsm
        self.threshold = float(threshold)
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.wake_index = int(wake_index)
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.max_inference_ms = float(max_inference_ms)

        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"KWS ONNX not found at {self.model_path}. "
                    "Place kws_model.onnx on local disk under models/kws/."
                )
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        self._session = session

        inputs = list(session.get_inputs())
        if not inputs:
            raise ValueError("KWS ONNX model has no inputs")
        feature_in = inputs[0]
        for item in inputs:
            if item.name.lower() in _FEATURE_INPUT_NAMES:
                feature_in = item
                break
        extra = [i.name for i in inputs if i.name != feature_in.name]
        if extra:
            raise ValueError(
                "KWS ONNX must take a single feature tensor; "
                f"unexpected extra inputs {extra}"
            )
        self._input_name = feature_in.name
        self._input_shape = _positive_dims(
            getattr(feature_in, "shape", None),
            (1, MFCC_N_MFCC, MFCC_FRAMES_PER_KWS_WINDOW),
        )
        self._output_names = [o.name for o in session.get_outputs()]
        if not self._output_names:
            raise ValueError("KWS ONNX model has no outputs")

        self._pcm = np.zeros(0, dtype=np.float32)
        self._last_score = 0.0
        self._running_score = 0.0
        self.last_inference_ms = 0.0

    @property
    def last_score(self) -> float:
        """Most recent window score in ``[0.0, 1.0]``."""
        return self._last_score

    @property
    def running_score(self) -> float:
        """EMA of window scores in ``[0.0, 1.0]``."""
        return self._running_score

    @property
    def detected(self) -> bool:
        return self._last_score >= self.threshold

    def reset(self, *, reset_fsm: bool = False) -> None:
        """Clear the PCM hop buffer and scores. Optionally reset the FSM."""
        self._pcm = np.zeros(0, dtype=np.float32)
        self._last_score = 0.0
        self._running_score = 0.0
        self.last_inference_ms = 0.0
        if reset_fsm and self.fsm is not None:
            self.fsm.reset(trigger="kws_reset")

    def process_pcm(self, pcm: np.ndarray) -> float:
        """Push capture samples; run inference whenever a 100 ms window is ready.

        Typical call cadence is one 30 ms frame (``FRAME_SAMPLES``). Returns the
        latest detection score (unchanged if the window is not yet full).
        """
        chunk = pcm_to_float32(pcm)
        if chunk.size:
            self._pcm = np.concatenate([self._pcm, chunk])
        score = self._last_score
        while self._pcm.size >= self.window_samples:
            window = self._pcm[: self.window_samples]
            score = self.process_window(window)
            self._pcm = self._pcm[self.hop_samples :]
        return score

    def process_window(self, pcm: np.ndarray) -> float:
        """Run MFCC + ONNX on a single 100 ms PCM buffer."""
        feats = extract_mfcc(pcm)
        return self.process_features(feats)

    def process_features(self, features: np.ndarray) -> float:
        """Run ONNX on a synthetic or extracted MFCC tensor.

        ``features`` should match ``extract_mfcc``: ``(1, n_mfcc, n_frames)``.
        """
        feeds = {self._input_name: self._adapt_features(features)}
        t0 = time.perf_counter()
        outs = self._session.run(self._output_names, feeds)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        named = {name: value for name, value in zip(self._output_names, outs)}
        raw = named.get("output", outs[0])
        for key in ("scores", "score", "logits", "keyword", "output"):
            if key in named:
                raw = named[key]
                break

        score = self._to_score(raw)
        alpha = self.ema_alpha
        self._last_score = score
        self._running_score = alpha * score + (1.0 - alpha) * self._running_score
        self._maybe_trigger(score)
        return score

    def _adapt_features(self, features: np.ndarray) -> np.ndarray:
        feats = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
        if feats.ndim == 2:
            feats = feats[np.newaxis, ...]
        target = self._input_shape
        if feats.ndim == 3 and len(target) == 4:
            feats = feats[:, np.newaxis, ...]
        elif (
            feats.ndim == 3
            and len(target) >= 3
            and target[1] == feats.shape[2]
            and target[2] == feats.shape[1]
        ):
            feats = np.transpose(feats, (0, 2, 1))
        return np.ascontiguousarray(feats, dtype=np.float32)

    def _to_score(self, raw: object) -> float:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            x = float(arr[0])
            if x < 0.0 or x > 1.0:
                x = _sigmoid(x)
            return float(min(1.0, max(0.0, x)))

        if np.any(arr < 0.0) or float(np.sum(arr)) > 1.01:
            shifted = arr - np.max(arr)
            exp = np.exp(shifted)
            arr = exp / np.maximum(np.sum(exp), 1e-12)
        idx = self.wake_index
        if idx < 0 or idx >= arr.size:
            idx = int(np.argmax(arr))
        x = float(arr[idx])
        return float(min(1.0, max(0.0, x)))

    def _maybe_trigger(self, score: float) -> None:
        if self.fsm is None or score < self.threshold:
            return
        if self.fsm.state is SystemState.IDLE_LISTENING:
            self.fsm.transition(
                SystemState.KEYWORD_DETECTED,
                trigger="kws_wake",
                payload={"score": score},
            )
