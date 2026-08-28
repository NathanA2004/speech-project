"""Local Silero VAD (ONNX) and MFCC feature extraction.

Everything here is offline: ONNX weights are loaded from disk via
``onnxruntime``, and MFCCs are computed with NumPy (no network, no cloud APIs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from config import (
    FRAME_SAMPLES,
    KWS_WINDOW_SAMPLES,
    MFCC_HOP_LENGTH,
    MFCC_N_FFT,
    MFCC_N_MELS,
    MFCC_N_MFCC,
    MFCC_PREEMPHASIS,
    MFCC_WIN_LENGTH,
    SAMPLE_RATE,
    VAD_LSTM_HIDDEN,
    VAD_LSTM_LAYERS,
    VAD_MODEL_PATH,
    VAD_SPEECH_THRESHOLD,
    VAD_WINDOW_SAMPLES,
)


def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Mono PCM as float32 in [-1, 1]. Accepts int16 or float, any extra dims."""
    x = np.asarray(pcm)
    if x.ndim > 1:
        x = np.reshape(x, -1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return np.clip(np.asarray(x, dtype=np.float32) / 32768.0, -1.0, 1.0)


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
    if not dims:
        return fallback
    return tuple(dims)


# ---------------------------------------------------------------------------
# Silero VAD (local ONNX Runtime)
# ---------------------------------------------------------------------------
class LocalVADDetector:
    """Frame-level voice activity detector using a local ``silero_vad.onnx``.

    Accepts the pipeline's 30 ms frames (480 samples at 16 kHz) and returns a
    speech probability in ``[0.0, 1.0]``. LSTM state tensors ``h`` and ``c``
    are kept across calls (Silero VAD v4). Newer exports that use a combined
    ``state`` tensor are also supported.

    Parameters
    ----------
    model_path
        Local ONNX file. Defaults to ``models/kws/silero_vad.onnx``.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        vad_window_samples: int = VAD_WINDOW_SAMPLES,
        threshold: float = VAD_SPEECH_THRESHOLD,
        *,
        session: Optional[ort.InferenceSession] = None,
    ) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Silero VAD in this pipeline is 16 kHz only, got {sample_rate}")
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.vad_window_samples = int(vad_window_samples)
        self.threshold = float(threshold)
        self.model_path = Path(model_path) if model_path is not None else VAD_MODEL_PATH

        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Silero VAD ONNX not found at {self.model_path}. "
                    "Place silero_vad.onnx on local disk under models/kws/."
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

        self._input_by_name = {i.name: i for i in session.get_inputs()}
        self._output_names = [o.name for o in session.get_outputs()]
        self._uses_hc = "h" in self._input_by_name and "c" in self._input_by_name
        self._uses_state = "state" in self._input_by_name
        if not self._uses_hc and not self._uses_state:
            raise ValueError(
                "ONNX VAD model must expose LSTM state as inputs 'h'+'c' or 'state'; "
                f"got {list(self._input_by_name)}"
            )

        audio_in = self._input_by_name.get("input")
        audio_shape = _positive_dims(
            getattr(audio_in, "shape", None), (1, self.vad_window_samples)
        )
        self._audio_samples = int(audio_shape[-1]) if len(audio_shape) >= 2 else self.vad_window_samples
        # v5 wrapper prepends 64 samples of context (512 + 64 = 576).
        self._context_samples = max(0, self._audio_samples - self.vad_window_samples)

        h_shape = _positive_dims(
            getattr(self._input_by_name.get("h"), "shape", None),
            (VAD_LSTM_LAYERS, 1, VAD_LSTM_HIDDEN),
        )
        c_shape = _positive_dims(
            getattr(self._input_by_name.get("c"), "shape", None),
            (VAD_LSTM_LAYERS, 1, VAD_LSTM_HIDDEN),
        )
        state_shape = _positive_dims(
            getattr(self._input_by_name.get("state"), "shape", None),
            (VAD_LSTM_LAYERS, 1, VAD_LSTM_HIDDEN * 2),
        )
        self._h_shape = h_shape
        self._c_shape = c_shape
        self._state_shape = state_shape

        self._h = np.zeros(self._h_shape, dtype=np.float32)
        self._c = np.zeros(self._c_shape, dtype=np.float32)
        self._state = np.zeros(self._state_shape, dtype=np.float32)
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)
        self.reset()

    @property
    def h(self) -> np.ndarray:
        """LSTM hidden state (v4)."""
        return self._h

    @property
    def c(self) -> np.ndarray:
        """LSTM cell state (v4)."""
        return self._c

    def reset(self, batch_size: int = 1) -> None:
        """Clear ``h`` / ``c`` (and v5 ``state``) so a new utterance starts clean."""
        h_shape = list(self._h_shape)
        c_shape = list(self._c_shape)
        state_shape = list(self._state_shape)
        if len(h_shape) >= 2:
            h_shape[1] = batch_size
        if len(c_shape) >= 2:
            c_shape[1] = batch_size
        if len(state_shape) >= 2:
            state_shape[1] = batch_size
        self._h = np.zeros(tuple(h_shape), dtype=np.float32)
        self._c = np.zeros(tuple(c_shape), dtype=np.float32)
        self._state = np.zeros(tuple(state_shape), dtype=np.float32)
        self._context = np.zeros((batch_size, self._context_samples), dtype=np.float32)

    def process_frame(self, pcm: np.ndarray) -> float:
        """Run VAD on one 30 ms PCM frame. Returns speech probability in [0, 1]."""
        wav = self._prepare_frame(pcm)
        window = np.zeros(self.vad_window_samples, dtype=np.float32)
        n = min(wav.size, self.vad_window_samples)
        window[:n] = wav[:n]
        audio = window.reshape(1, -1)
        if self._context_samples > 0:
            audio = np.concatenate([self._context, audio], axis=1)
            self._context = audio[:, -self._context_samples :]

        feeds: dict[str, np.ndarray] = {}
        if "input" in self._input_by_name:
            feeds["input"] = audio.astype(np.float32, copy=False)
        if self._uses_hc:
            feeds["h"] = self._h
            feeds["c"] = self._c
        if self._uses_state:
            feeds["state"] = self._state
        if "sr" in self._input_by_name:
            feeds["sr"] = np.array(self.sample_rate, dtype=np.int64)

        outs = self._session.run(None, feeds)
        named = {name: value for name, value in zip(self._output_names, outs)}

        prob_arr = named.get("output", outs[0])
        prob = float(np.clip(np.asarray(prob_arr).reshape(-1)[0], 0.0, 1.0))

        if "hn" in named:
            self._h = np.array(named["hn"], dtype=np.float32, copy=True)
        elif "h" in named:
            self._h = np.array(named["h"], dtype=np.float32, copy=True)
        elif self._uses_hc and len(outs) >= 2:
            self._h = np.array(outs[1], dtype=np.float32, copy=True)

        if "cn" in named:
            self._c = np.array(named["cn"], dtype=np.float32, copy=True)
        elif "c" in named:
            self._c = np.array(named["c"], dtype=np.float32, copy=True)
        elif self._uses_hc and len(outs) >= 3:
            self._c = np.array(outs[2], dtype=np.float32, copy=True)

        for key in ("stateN", "state"):
            if key in named and key != "output":
                self._state = np.array(named[key], dtype=np.float32, copy=True)
                break
        else:
            if self._uses_state and not self._uses_hc and len(outs) >= 2:
                self._state = np.array(outs[1], dtype=np.float32, copy=True)

        return prob

    def is_speech(self, pcm: np.ndarray, threshold: Optional[float] = None) -> bool:
        """True when ``process_frame`` is at or above the speech threshold."""
        cut = self.threshold if threshold is None else float(threshold)
        return self.process_frame(pcm) >= cut

    def _prepare_frame(self, pcm: np.ndarray) -> np.ndarray:
        wav = pcm_to_float32(pcm)
        if wav.size == self.frame_samples:
            return wav
        out = np.zeros(self.frame_samples, dtype=np.float32)
        n = min(wav.size, self.frame_samples)
        if n:
            out[:n] = wav[:n]
        return out


# ---------------------------------------------------------------------------
# MFCC (NumPy) for sliding-window KWS
# ---------------------------------------------------------------------------
def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mfcc_frame_count(
    n_samples: int,
    win_length: int = MFCC_WIN_LENGTH,
    hop_length: int = MFCC_HOP_LENGTH,
) -> int:
    """Number of STFT frames for a buffer, matching ``extract_mfcc`` padding."""
    n = max(int(n_samples), int(win_length))
    return 1 + (n - int(win_length)) // int(hop_length)


def mel_filterbank(
    n_mels: int = MFCC_N_MELS,
    n_fft: int = MFCC_N_FFT,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Triangular HTK-style Mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    mels = np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    hz = np.asarray(mel_to_hz(mels), dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(np.int32)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        left_span = center - left
        right_span = right - center
        for j in range(left, center):
            if 0 <= j < n_freqs and left_span > 0:
                fb[i, j] = (j - left) / left_span
        for j in range(center, right):
            if 0 <= j < n_freqs and right_span > 0:
                fb[i, j] = (right - j) / right_span
    return fb


def _dct_ii_ortho_basis(n_mels: int, n_mfcc: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)
    basis = np.cos(np.pi * np.outer(k, 2.0 * n + 1.0) / (2.0 * n_mels))
    basis *= np.sqrt(2.0 / n_mels)
    basis[0] *= np.sqrt(0.5)
    return basis.astype(np.float32)


class MFCCExtractor:
    """Cached Mel filterbank + DCT for 16 kHz sliding-window KWS features."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mfcc: int = MFCC_N_MFCC,
        n_mels: int = MFCC_N_MELS,
        n_fft: int = MFCC_N_FFT,
        win_length: int = MFCC_WIN_LENGTH,
        hop_length: int = MFCC_HOP_LENGTH,
        preemphasis: float = MFCC_PREEMPHASIS,
        normalize: bool = True,
    ) -> None:
        if n_mfcc > n_mels:
            raise ValueError("n_mfcc cannot exceed n_mels")
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis = preemphasis
        self.normalize = normalize
        self._window = np.hamming(win_length).astype(np.float32)
        self._fb = mel_filterbank(n_mels, n_fft, sample_rate)
        self._dct = _dct_ii_ortho_basis(n_mels, n_mfcc)

    @property
    def output_shape_for_kws_window(self) -> tuple[int, int, int]:
        """``(1, n_mfcc, n_frames)`` for a 100 ms KWS window."""
        return (1, self.n_mfcc, mfcc_frame_count(KWS_WINDOW_SAMPLES, self.win_length, self.hop_length))

    def extract(self, pcm_buffer: np.ndarray) -> np.ndarray:
        """Return float32 MFCCs with shape ``(1, n_mfcc, n_frames)``."""
        x = pcm_to_float32(pcm_buffer)
        if x.size < self.win_length:
            padded = np.zeros(self.win_length, dtype=np.float32)
            padded[: x.size] = x
            x = padded
        if self.preemphasis:
            x = np.concatenate(
                (x[:1], x[1:] - np.float32(self.preemphasis) * x[:-1])
            ).astype(np.float32, copy=False)

        n_frames = mfcc_frame_count(x.size, self.win_length, self.hop_length)
        frames = np.lib.stride_tricks.sliding_window_view(x, self.win_length)[:: self.hop_length][
            :n_frames
        ]
        frames = np.ascontiguousarray(frames, dtype=np.float32) * self._window
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32)
        mel = power @ self._fb.T
        log_mel = np.log(np.maximum(mel, 1e-10))
        mfcc = log_mel @ self._dct.T  # (n_frames, n_mfcc)
        feats = np.ascontiguousarray(mfcc.T[np.newaxis, ...], dtype=np.float32)
        if self.normalize and feats.shape[-1] > 1:
            mean = feats.mean(axis=-1, keepdims=True)
            std = feats.std(axis=-1, keepdims=True)
            feats = (feats - mean) / np.maximum(std, 1e-6)
        return feats


_DEFAULT_MFCC: Optional[MFCCExtractor] = None


def extract_mfcc(
    pcm_buffer: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = MFCC_N_MFCC,
    n_mels: int = MFCC_N_MELS,
    n_fft: int = MFCC_N_FFT,
    win_length: int = MFCC_WIN_LENGTH,
    hop_length: int = MFCC_HOP_LENGTH,
    preemphasis: float = MFCC_PREEMPHASIS,
    normalize: bool = True,
) -> np.ndarray:
    """Convert 16 kHz PCM into normalized MFCC features for KWS.

    Returns
    -------
    np.ndarray
        ``float32`` tensor of shape ``(1, n_mfcc, n_frames)``. A 100 ms
        (1600-sample) window yields ``(1, 40, 8)`` with the default 25/10 ms
        STFT hop.
    """
    defaults = (
        sample_rate == SAMPLE_RATE
        and n_mfcc == MFCC_N_MFCC
        and n_mels == MFCC_N_MELS
        and n_fft == MFCC_N_FFT
        and win_length == MFCC_WIN_LENGTH
        and hop_length == MFCC_HOP_LENGTH
        and preemphasis == MFCC_PREEMPHASIS
        and normalize is True
    )
    if defaults:
        global _DEFAULT_MFCC
        if _DEFAULT_MFCC is None:
            _DEFAULT_MFCC = MFCCExtractor()
        return _DEFAULT_MFCC.extract(pcm_buffer)
    return MFCCExtractor(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        preemphasis=preemphasis,
        normalize=normalize,
    ).extract(pcm_buffer)
