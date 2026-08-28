"""Unit tests that do not require a microphone."""

from collections import deque

import numpy as np

from audio.stream import frame_rms
from config import (
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    KWS_MODEL_PATH,
    MODELS_DIR,
    RING_BUFFER_FRAMES,
    SAMPLE_RATE,
    SLM_MODEL_PATH,
    STT_DOWNLOAD_ROOT,
    VAD_MODEL_PATH,
)


def test_frame_samples_are_30ms_at_16k():
    assert SAMPLE_RATE == 16_000
    assert FRAME_DURATION_MS == 30
    assert FRAME_SAMPLES == 480


def test_ring_buffer_holds_three_seconds():
    assert RING_BUFFER_FRAMES == 100
    ring: deque[np.ndarray] = deque(maxlen=RING_BUFFER_FRAMES)
    for i in range(150):
        ring.append(np.full(FRAME_SAMPLES, i, dtype=np.int16))
    assert len(ring) == 100
    concatenated = np.concatenate(list(ring))
    assert concatenated.size == SAMPLE_RATE * 3


def test_frame_rms_silence_and_peak():
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    peak = np.full(FRAME_SAMPLES, 32767, dtype=np.int16)
    assert frame_rms(silence) == 0.0
    assert frame_rms(peak) > 0.9


def test_model_paths_are_local_models_dir():
    assert MODELS_DIR.name == "models"
    assert KWS_MODEL_PATH.as_posix().endswith("models/kws/kws_model.onnx")
    assert VAD_MODEL_PATH.as_posix().endswith("models/kws/silero_vad.onnx")
    assert SLM_MODEL_PATH.as_posix().endswith("models/slm/model.gguf")
    assert STT_DOWNLOAD_ROOT.as_posix().endswith("models/stt")
    for path in (KWS_MODEL_PATH, VAD_MODEL_PATH, SLM_MODEL_PATH, STT_DOWNLOAD_ROOT):
        assert "models" in path.parts
        assert path.is_absolute() or path.parts[0] == "models"
        # Resolved paths live under the repo's ./models directory.
        assert MODELS_DIR in path.parents or path == STT_DOWNLOAD_ROOT
