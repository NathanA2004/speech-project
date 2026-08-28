"""Local continuous audio capture and feature extraction."""

from .stream import AudioStreamManager, frame_rms

__all__ = ["AudioStreamManager", "frame_rms"]
