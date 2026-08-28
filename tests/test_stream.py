"""Unit tests that do not require a microphone."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audio.stream import AudioStreamManager, frame_rms, to_int16_mono
from config import (
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    KWS_MODEL_PATH,
    KWS_MODEL_RELATIVE,
    MODELS_DIR,
    MODELS_DIR_RELATIVE,
    PROJECT_ROOT,
    RING_BUFFER_FRAMES,
    SAMPLE_RATE,
    SLM_MODEL_PATH,
    SLM_MODEL_RELATIVE,
    STT_DIR_RELATIVE,
    STT_DOWNLOAD_ROOT,
    VAD_MODEL_PATH,
    VAD_MODEL_RELATIVE,
)


def test_frame_samples_are_30ms_at_16k():
    assert SAMPLE_RATE == 16_000
    assert FRAME_DURATION_MS == 30
    assert FRAME_SAMPLES == 480


def test_manager_ring_buffer_holds_three_seconds():
    mgr = AudioStreamManager()
    status = SimpleNamespace(input_overflow=False)
    frame = np.zeros((FRAME_SAMPLES, 1), dtype=np.int16)
    for i in range(150):
        frame[:] = i
        mgr._on_audio(frame, FRAME_SAMPLES, None, status)
    snap = mgr.snapshot_ring_buffer()
    assert snap.size == SAMPLE_RATE * 3
    assert snap.dtype == np.int16
    assert mgr.stats().frames_captured == 150


def test_manager_counts_overflows_and_queue_drops():
    mgr = AudioStreamManager(queue_max=1)
    ok = SimpleNamespace(input_overflow=False)
    overflow = SimpleNamespace(input_overflow=True)
    frame = np.ones((FRAME_SAMPLES, 1), dtype=np.int16)
    mgr._on_audio(frame, FRAME_SAMPLES, None, overflow)
    mgr._on_audio(frame, FRAME_SAMPLES, None, ok)
    stats = mgr.stats()
    assert stats.overflow_count == 1
    assert stats.queue_drop_count >= 1
    assert stats.dropped_frames == stats.overflow_count + stats.queue_drop_count


def test_frame_rms_silence_and_peak():
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    peak = np.full(FRAME_SAMPLES, 32767, dtype=np.int16)
    assert frame_rms(silence) == 0.0
    assert frame_rms(peak) > 0.9


def test_float_pcm_converts_to_int16_not_truncated_silence():
    floats = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)
    pcm = to_int16_mono(floats, FRAME_SAMPLES)
    assert pcm.dtype == np.int16
    assert pcm.size == FRAME_SAMPLES
    assert int(np.mean(np.abs(pcm))) > 1000


def test_model_paths_are_local_models_dir():
    assert MODELS_DIR_RELATIVE.as_posix() == "models"
    assert KWS_MODEL_RELATIVE.as_posix() == "models/kws/kws_model.onnx"
    assert VAD_MODEL_RELATIVE.as_posix() == "models/kws/silero_vad.onnx"
    assert SLM_MODEL_RELATIVE.as_posix() == "models/slm/model.gguf"
    assert STT_DIR_RELATIVE.as_posix() == "models/stt"

    assert KWS_MODEL_PATH == PROJECT_ROOT / KWS_MODEL_RELATIVE
    assert VAD_MODEL_PATH == PROJECT_ROOT / VAD_MODEL_RELATIVE
    assert SLM_MODEL_PATH == PROJECT_ROOT / SLM_MODEL_RELATIVE
    assert STT_DOWNLOAD_ROOT == PROJECT_ROOT / STT_DIR_RELATIVE
    assert MODELS_DIR == PROJECT_ROOT / "models"
    for path in (KWS_MODEL_PATH, VAD_MODEL_PATH, SLM_MODEL_PATH, STT_DOWNLOAD_ROOT):
        assert path.is_relative_to(MODELS_DIR)


def test_requirements_are_local_only():
    text = Path(__file__).resolve().parent.parent.joinpath("requirements.txt").read_text(
        encoding="utf-8"
    ).lower()
    banned = (
        "openai",
        "anthropic",
        "google-generativeai",
        "google.generativeai",
        "cohere",
        "groq",
    )
    for name in banned:
        assert name not in text
