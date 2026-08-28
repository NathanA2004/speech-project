"""Local KWS, STT, and SLM inference engines (disk-only model paths)."""

from .intent_engine import (
    Intent,
    IntentEngine,
    IntentParseError,
    LocalIntentParser,
    LocalTranscriber,
)

__all__ = [
    "KWSEngine",
    "Intent",
    "IntentEngine",
    "IntentParseError",
    "LocalIntentParser",
    "LocalTranscriber",
]


def __getattr__(name: str):
    if name == "KWSEngine":
        from .kws_engine import KWSEngine

        return KWSEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
