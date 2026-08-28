"""Unit tests for local VAD + MFCC using synthetic PCM (no microphone)."""

from __future__ import annotations

import numpy as np
import pytest

from audio.features import (
    LocalVADDetector,
    MFCCExtractor,
    extract_mfcc,
    mfcc_frame_count,
    pcm_to_float32,
)
from config import (
    FRAME_SAMPLES,
    KWS_WINDOW_SAMPLES,
    MFCC_FRAMES_PER_KWS_WINDOW,
    MFCC_HOP_LENGTH,
    MFCC_N_MFCC,
    MFCC_WIN_LENGTH,
    SAMPLE_RATE,
    VAD_LSTM_HIDDEN,
    VAD_LSTM_LAYERS,
    VAD_WINDOW_SAMPLES,
)
from onnx_stub import write_silero_vad_stub


def _silence(n: int = FRAME_SAMPLES) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


def _sine(n: int = FRAME_SAMPLES, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = amplitude * np.sin(2.0 * np.pi * freq * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _white_noise(n: int = FRAME_SAMPLES, scale: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * scale
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


@pytest.fixture
def vad_model_path(tmp_path):
    return write_silero_vad_stub(tmp_path / "silero_vad.onnx")


@pytest.fixture
def vad(vad_model_path):
    return LocalVADDetector(model_path=vad_model_path)


def test_stub_onnx_matches_silero_v4_io(vad_model_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(vad_model_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    assert names == ["input", "h", "c", "sr"]
    assert outs == ["output", "hn", "cn"]


def test_missing_vad_model_raises(tmp_path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError, match="silero_vad"):
        LocalVADDetector(model_path=missing)


def test_vad_state_tensors_are_h_and_c(vad):
    assert vad.h.shape == (VAD_LSTM_LAYERS, 1, VAD_LSTM_HIDDEN)
    assert vad.c.shape == (VAD_LSTM_LAYERS, 1, VAD_LSTM_HIDDEN)
    assert vad.h.dtype == np.float32
    assert vad.c.dtype == np.float32
    assert np.all(vad.h == 0) and np.all(vad.c == 0)


def test_vad_probability_low_for_silence(vad):
    scores = [vad.process_frame(_silence()) for _ in range(8)]
    assert all(0.0 <= p <= 1.0 for p in scores)
    assert all(isinstance(p, float) for p in scores)
    assert float(np.mean(scores)) < 0.1


def test_vad_probability_high_for_sine_and_noise(vad):
    sine_scores = [vad.process_frame(_sine(freq=180.0 + 20 * i)) for i in range(8)]
    vad.reset()
    noise_scores = [vad.process_frame(_white_noise(seed=i)) for i in range(8)]
    assert float(np.mean(sine_scores)) > 0.5
    assert float(np.mean(noise_scores)) > 0.5
    assert all(0.0 <= p <= 1.0 for p in sine_scores + noise_scores)


def test_vad_silence_lower_than_active_audio(vad):
    silence = float(np.mean([vad.process_frame(_silence()) for _ in range(5)]))
    vad.reset()
    active = float(np.mean([vad.process_frame(_sine()) for _ in range(5)]))
    assert silence < 0.1
    assert active > 0.5
    assert active > silence + 0.4


def test_vad_accepts_30ms_int16_frames(vad):
    frame = _sine(FRAME_SAMPLES)
    assert frame.size == 480
    assert FRAME_SAMPLES == 480
    assert VAD_WINDOW_SAMPLES == 512
    p = vad.process_frame(frame)
    assert 0.0 <= p <= 1.0


def test_vad_threads_h_c_across_frames(vad_model_path):
    """h and c from one step are the inputs to the next (Identity stub)."""
    calls: list[dict] = []
    real = LocalVADDetector(model_path=vad_model_path)
    inner = real._session.run

    def wrapped(output_names, feeds):
        calls.append({k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in feeds.items()})
        return inner(output_names, feeds)

    real._session.run = wrapped  # type: ignore[method-assign]
    real.process_frame(_sine())
    real.process_frame(_sine(freq=330.0))
    assert len(calls) == 2
    assert calls[0]["h"].shape == (2, 1, 64)
    assert calls[0]["c"].shape == (2, 1, 64)
    np.testing.assert_array_equal(calls[1]["h"], calls[0]["h"])
    np.testing.assert_array_equal(calls[1]["c"], calls[0]["c"])
    np.testing.assert_array_equal(real.h, calls[1]["h"])


def test_vad_reset_clears_state(vad):
    vad.process_frame(_sine())
    vad.reset()
    assert np.all(vad.h == 0) and np.all(vad.c == 0)


def test_vad_is_speech_flags(vad):
    assert vad.is_speech(_silence()) is False
    vad.reset()
    assert vad.is_speech(_sine()) is True


def test_pcm_to_float32_int16_and_float():
    silence = pcm_to_float32(_silence())
    peak = pcm_to_float32(np.full(FRAME_SAMPLES, 32767, dtype=np.int16))
    floats = pcm_to_float32(np.full(FRAME_SAMPLES, 0.25, dtype=np.float32))
    assert silence.dtype == np.float32
    assert float(np.max(np.abs(silence))) == 0.0
    assert float(np.max(peak)) > 0.99
    assert floats.dtype == np.float32
    np.testing.assert_allclose(floats.mean(), 0.25, atol=1e-6)


def test_mfcc_kws_window_shape_and_dtype():
    pcm = _sine(KWS_WINDOW_SAMPLES, freq=440.0)
    feats = extract_mfcc(pcm)
    assert feats.dtype == np.float32
    assert feats.shape == (1, MFCC_N_MFCC, MFCC_FRAMES_PER_KWS_WINDOW)
    assert feats.shape == (1, 40, 8)
    assert np.isfinite(feats).all()


def test_mfcc_frame_count_matches_extractor():
    assert mfcc_frame_count(KWS_WINDOW_SAMPLES) == MFCC_FRAMES_PER_KWS_WINDOW
    assert mfcc_frame_count(FRAME_SAMPLES) == 1 + (FRAME_SAMPLES - MFCC_WIN_LENGTH) // MFCC_HOP_LENGTH
    short = extract_mfcc(_sine(FRAME_SAMPLES))
    assert short.shape == (1, MFCC_N_MFCC, mfcc_frame_count(FRAME_SAMPLES))


def test_mfcc_on_silence_noise_and_sine():
    silence = extract_mfcc(_silence(KWS_WINDOW_SAMPLES), normalize=False)
    noise = extract_mfcc(_white_noise(KWS_WINDOW_SAMPLES), normalize=False)
    sine = extract_mfcc(_sine(KWS_WINDOW_SAMPLES, freq=300.0), normalize=False)
    for feats in (silence, noise, sine):
        assert feats.dtype == np.float32
        assert feats.shape == (1, MFCC_N_MFCC, MFCC_FRAMES_PER_KWS_WINDOW)
        assert np.isfinite(feats).all()
    # Energy (c0) should be lower for silence than for active signals.
    assert float(silence[0, 0].mean()) < float(sine[0, 0].mean())
    assert float(silence[0, 0].mean()) < float(noise[0, 0].mean())


def test_mfcc_extractor_kws_shape_property():
    ext = MFCCExtractor()
    assert ext.output_shape_for_kws_window == (1, 40, 8)
    window = _white_noise(KWS_WINDOW_SAMPLES, seed=3)
    assert ext.extract(window).shape == ext.output_shape_for_kws_window


def test_mfcc_normalized_has_zero_mean_over_time():
    feats = extract_mfcc(_sine(KWS_WINDOW_SAMPLES, freq=500.0), normalize=True)
    mean = feats.mean(axis=-1)
    np.testing.assert_allclose(mean, 0.0, atol=1e-3)


def test_mfcc_pads_short_buffers():
    tiny = _sine(100)
    feats = extract_mfcc(tiny)
    assert feats.shape == (1, MFCC_N_MFCC, 1)
    assert feats.dtype == np.float32


def test_vad_rejects_non_16k():
    with pytest.raises(ValueError, match="16 kHz"):
        LocalVADDetector(model_path="unused.onnx", sample_rate=8000)
