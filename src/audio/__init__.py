"""Local continuous audio capture and feature extraction."""

from .stream import AudioStreamManager, StreamStats, frame_rms, to_int16_mono

__all__ = ["AudioStreamManager", "StreamStats", "frame_rms", "to_int16_mono"]
