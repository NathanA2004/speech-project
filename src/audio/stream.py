"""Real-time PCM capture via sounddevice / PortAudio.

Captures 30 ms frames (16 kHz, 16-bit, mono) into a thread-safe 3-second
ring buffer. The PortAudio callback never allocates beyond a copy + enqueue
so input overflows are counted instead of silently dropped.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

from config import (
    CHANNELS,
    DTYPE,
    FRAME_SAMPLES,
    INPUT_LATENCY_S,
    QUEUE_MAX_FRAMES,
    RING_BUFFER_FRAMES,
    SAMPLE_RATE,
)


def frame_rms(pcm: np.ndarray) -> float:
    """RMS of an int16 frame, normalized to 0.0–1.0."""
    if pcm.size == 0:
        return 0.0
    x = pcm.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass(frozen=True)
class StreamStats:
    frames_captured: int
    overflow_count: int
    queue_drop_count: int
    elapsed_s: float
    expected_frames: int
    rms: float

    @property
    def dropped_frames(self) -> int:
        return self.overflow_count + self.queue_drop_count

    @property
    def timing_gap(self) -> int:
        """How many frames short of wall-clock expectation (0 if on time)."""
        return max(0, self.expected_frames - self.frames_captured)


class AudioStreamManager:
    """Low-latency microphone capture with a rolling pre-trigger ring buffer.

    Parameters
    ----------
    sample_rate, channels, frame_samples, dtype
        Must match the rest of the local pipeline (16 kHz / mono / 30 ms / int16).
    device
        sounddevice input index or name. ``None`` uses the system default.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        frame_samples: int = FRAME_SAMPLES,
        dtype: str = DTYPE,
        device: Optional[int | str] = None,
        latency: float = INPUT_LATENCY_S,
        ring_frames: int = RING_BUFFER_FRAMES,
        queue_max: int = QUEUE_MAX_FRAMES,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_samples = frame_samples
        self.dtype = dtype
        self.device = device
        self.latency = latency

        self._ring: deque[np.ndarray] = deque(maxlen=ring_frames)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_max)
        self._lock = threading.Lock()

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._t0 = 0.0
        self._frames_captured = 0
        self._overflow_count = 0
        self._queue_drop_count = 0
        self._last_rms = 0.0

    def __enter__(self) -> "AudioStreamManager":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._frames_captured = 0
        self._overflow_count = 0
        self._queue_drop_count = 0
        self._last_rms = 0.0
        self._ring.clear()
        self._drain_queue()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.frame_samples,
            latency=self.latency,
            callback=self._on_audio,
            device=self.device,
        )
        self._t0 = time.perf_counter()
        self._running = True
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        # Keep this callback short: copy, enqueue, update counters.
        if status.input_overflow:
            with self._lock:
                self._overflow_count += 1

        if frames != self.frame_samples:
            # Partial block — still copy what we got, pad to a full frame.
            pcm = np.zeros(self.frame_samples, dtype=np.int16)
            n = min(frames, self.frame_samples)
            mono = indata[:n, 0] if indata.ndim > 1 else indata[:n]
            pcm[:n] = np.asarray(mono, dtype=np.int16).reshape(-1)
        else:
            mono = indata[:, 0] if indata.ndim > 1 else indata.reshape(-1)
            pcm = np.copy(np.asarray(mono, dtype=np.int16))

        rms = frame_rms(pcm)
        self._ring.append(pcm)
        with self._lock:
            self._frames_captured += 1
            self._last_rms = rms

        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            with self._lock:
                self._queue_drop_count += 1

    def snapshot_ring_buffer(self) -> np.ndarray:
        """Concatenated 3 s of recent PCM (pre-trigger context)."""
        frames = list(self._ring)
        if not frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(frames)

    def stats(self) -> StreamStats:
        elapsed = time.perf_counter() - self._t0 if self._t0 else 0.0
        expected = int(elapsed * self.sample_rate / self.frame_samples) if elapsed > 0 else 0
        with self._lock:
            return StreamStats(
                frames_captured=self._frames_captured,
                overflow_count=self._overflow_count,
                queue_drop_count=self._queue_drop_count,
                elapsed_s=elapsed,
                expected_frames=expected,
                rms=self._last_rms,
            )

    def iter_frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Blocking iterator of 30 ms int16 frames. Stops when the stream ends."""
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                if not self._running:
                    return
                continue
            if item is None:
                return
            yield item

    async def frames(self, timeout: float = 1.0) -> AsyncIterator[np.ndarray]:
        """Async iterator of 30 ms int16 frames (does not block the event loop)."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(
                None, lambda: self._get_frame(timeout)
            )
            if item is None:
                if not self._running:
                    return
                continue
            yield item

    def _get_frame(self, timeout: float) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
