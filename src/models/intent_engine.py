"""Offline Tier-2 STT (faster-whisper) and Tier-3 intent parsing (llama.cpp).

All model files are loaded from local disk under ``models/stt`` and
``models/slm``. No cloud APIs, no runtime downloads.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from config import (
    SLM_DIR,
    SLM_MODEL_PATH,
    SLM_N_CTX,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_DOWNLOAD_ROOT,
    STT_MODEL_NAME,
)

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional until a real model is loaded
    WhisperModel = None  # type: ignore[misc, assignment]

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:  # pragma: no cover - optional until a real model is loaded
    Llama = None  # type: ignore[misc, assignment]
    LlamaGrammar = None  # type: ignore[misc, assignment]


ALLOWED_INTENTS = frozenset(
    {"TOGGLE_DEVICE", "QUERY_STATUS", "ADJUST_SETTING", "UNKNOWN"}
)
INTENT_ENUM = ("TOGGLE_DEVICE", "QUERY_STATUS", "ADJUST_SETTING", "UNKNOWN")

# JSON Schema enforced at sample time (llama.cpp grammar) and after decode.
INTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENT_ENUM)},
        "target": {"type": "string"},
        "value": {"type": ["string", "number"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["intent", "target", "value", "confidence"],
    "additionalProperties": False,
}

# GBNF: assistant may emit only this object (no markdown, no chatter).
INTENT_GBNF = r"""
root ::= object
object ::= "{" ws intent-kv "," ws target-kv "," ws value-kv "," ws confidence-kv ws "}"
intent-kv ::= "\"intent\"" ws ":" ws intent
intent ::= "\"TOGGLE_DEVICE\"" | "\"QUERY_STATUS\"" | "\"ADJUST_SETTING\"" | "\"UNKNOWN\""
target-kv ::= "\"target\"" ws ":" ws string
value-kv ::= "\"value\"" ws ":" ws (string | number)
confidence-kv ::= "\"confidence\"" ws ":" ws confidence
confidence ::= "0" "." [0-9]+ | "1" ("." "0"+)? | "1" | "0"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\n\r]*
""".strip()

SYSTEM_PROMPT = (
    "You are a local intent parser. Convert the utterance into a single JSON "
    "object and nothing else. No markdown, no commentary. Schema: "
    '{"intent":"TOGGLE_DEVICE|QUERY_STATUS|ADJUST_SETTING|UNKNOWN",'
    '"target":"<string>","value":"<string or number>","confidence":<0.0-1.0>}.'
)

_WS_RE = re.compile(r"\s+")


class IntentParseError(ValueError):
    """Raised when SLM output is not valid, strictly-typed intent JSON."""


@dataclass(frozen=True)
class Intent:
    """Structured command produced by the local SLM."""

    intent: str
    target: str
    value: Union[str, int, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


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


def clean_transcript(text: str) -> str:
    """Normalize Whisper segments into a single utterance string."""
    cleaned = _WS_RE.sub(" ", (text or "").replace("\u00a0", " ")).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
        cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def parse_intent_json(raw: str) -> Intent:
    """Validate a complete JSON object against the intent schema.

    The entire stripped string must be one JSON object. Conversational
    wrappers, markdown fences, and extra keys are rejected.
    """
    text = (raw or "").strip()
    if not text:
        raise IntentParseError("empty intent output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"intent output is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntentParseError("intent JSON must be an object")
    return intent_from_mapping(data)


def intent_from_mapping(data: dict[str, Any]) -> Intent:
    extra = set(data) - {"intent", "target", "value", "confidence"}
    if extra:
        raise IntentParseError(f"unexpected intent fields: {sorted(extra)}")
    missing = [key for key in ("intent", "target", "value", "confidence") if key not in data]
    if missing:
        raise IntentParseError(f"missing intent fields: {missing}")

    intent = data["intent"]
    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise IntentParseError(
            f"intent must be one of {sorted(ALLOWED_INTENTS)}, got {intent!r}"
        )

    target = data["target"]
    if not isinstance(target, str):
        raise IntentParseError("target must be a string")

    value = data["value"]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise IntentParseError("value must be a string or number")
    if isinstance(value, float) and not np.isfinite(value):
        raise IntentParseError("value must be a finite number")

    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise IntentParseError("confidence must be a number")
    confidence_f = float(confidence)
    if not np.isfinite(confidence_f) or confidence_f < 0.0 or confidence_f > 1.0:
        raise IntentParseError("confidence must be between 0.0 and 1.0")

    return Intent(
        intent=intent,
        target=target,
        value=value,
        confidence=confidence_f,
    )


def _is_whisper_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "model.bin").is_file() or (path / "model.npz").is_file()


def find_local_whisper(download_root: Path, model_name: str) -> Optional[Path]:
    """Locate a CTranslate2 Whisper directory under ``download_root``."""
    root = Path(download_root)
    if _is_whisper_dir(root):
        return root
    named = root / model_name
    if _is_whisper_dir(named):
        return named
    alt = root / f"faster-whisper-{model_name}"
    if _is_whisper_dir(alt):
        return alt
    if not root.is_dir():
        return None
    for pattern in (
        f"models--*--faster-whisper-{model_name}",
        f"models--*--{model_name}",
        f"*{model_name}*",
    ):
        for candidate in root.glob(pattern):
            if _is_whisper_dir(candidate):
                return candidate
            for snap in candidate.glob("snapshots/*"):
                if _is_whisper_dir(snap):
                    return snap
    for bin_path in root.rglob("model.bin"):
        if _is_whisper_dir(bin_path.parent):
            return bin_path.parent
    return None


def find_local_gguf(model_path: Path, search_dir: Path) -> Optional[Path]:
    """Return ``model_path`` if present, else a single ``*.gguf`` in ``search_dir``."""
    path = Path(model_path)
    if path.is_file():
        return path
    directory = Path(search_dir)
    if not directory.is_dir():
        return None
    ggufs = sorted(p for p in directory.glob("*.gguf") if p.is_file())
    if len(ggufs) == 1:
        return ggufs[0]
    preferred = [
        p
        for p in ggufs
        if any(token in p.name.lower() for token in ("llama", "phi", "qwen", "gemma"))
    ]
    if len(preferred) == 1:
        return preferred[0]
    return None


def _segment_text(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("text", ""))
    return str(getattr(segment, "text", segment) or "")


def build_intent_grammar() -> Any:
    """Compile JSON-schema GBNF for llama.cpp constrained decoding."""
    if LlamaGrammar is None:
        return INTENT_GBNF
    schema = json.dumps(INTENT_JSON_SCHEMA)
    if hasattr(LlamaGrammar, "from_json_schema"):
        try:
            return LlamaGrammar.from_json_schema(schema, verbose=False)
        except TypeError:
            try:
                return LlamaGrammar.from_json_schema(schema)
            except Exception:
                pass
        except Exception:
            pass
    try:
        return LlamaGrammar.from_string(INTENT_GBNF, verbose=False)
    except TypeError:
        return LlamaGrammar.from_string(INTENT_GBNF)
    except Exception:
        return INTENT_GBNF


class LocalTranscriber:
    """In-memory faster-whisper STT using a local CTranslate2 model.

    Parameters
    ----------
    model_name
        Whisper size (``tiny.en`` / ``base.en``) or a directory name under
        ``download_root``.
    model
        Injected Whisper-like object (tests). Must implement ``transcribe``.
    """

    def __init__(
        self,
        model_name: str = STT_MODEL_NAME,
        *,
        device: str = STT_DEVICE,
        compute_type: str = STT_COMPUTE_TYPE,
        download_root: Optional[str | Path] = None,
        model: Optional[Any] = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = Path(download_root) if download_root is not None else STT_DOWNLOAD_ROOT
        self.local_files_only = bool(local_files_only)
        self.model_path: Optional[Path] = None

        if model is not None:
            self._model = model
            return
        local = find_local_whisper(self.download_root, model_name)
        if local is None:
            raise FileNotFoundError(
                f"Whisper STT model {model_name!r} not found under {self.download_root}. "
                "Place a CTranslate2 model (model.bin) on local disk under models/stt/."
            )
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is required to load a local STT model. "
                "Install it from requirements.txt."
            )
        self.model_path = local
        # Explicit directory path: never hit the network / Hugging Face hub.
        self._model = WhisperModel(
            str(local),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )

    def transcribe(
        self,
        pcm: np.ndarray,
        *,
        language: str = "en",
        sample_rate: Optional[int] = None,
    ) -> str:
        """Transcribe a PCM buffer already in RAM (int16 or float32, no disk I/O)."""
        del sample_rate  # Whisper CTranslate2 models are 16 kHz; capture matches.
        audio = pcm_to_float32(pcm)
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
        parts = [_segment_text(seg) for seg in segments]
        return clean_transcript(" ".join(parts))


class LocalIntentParser:
    """Offline GGUF intent parser with GBNF / JSON-schema constrained decoding."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        n_ctx: int = SLM_N_CTX,
        llm: Optional[Any] = None,
        grammar: Optional[Any] = None,
        search_dir: Optional[str | Path] = None,
        verbose: bool = False,
    ) -> None:
        self.n_ctx = int(n_ctx)
        self.search_dir = Path(search_dir) if search_dir is not None else SLM_DIR
        requested = Path(model_path) if model_path is not None else SLM_MODEL_PATH
        self.model_path = requested
        self._grammar = grammar if grammar is not None else INTENT_GBNF

        if llm is not None:
            self._llm = llm
            return
        resolved = find_local_gguf(requested, self.search_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"SLM GGUF not found at {requested}. "
                "Place a quantized .gguf on local disk under models/slm/."
            )
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required to load a local GGUF model. "
                "Install it from requirements.txt."
            )
        self.model_path = resolved
        self._grammar = build_intent_grammar() if grammar is None else grammar
        self._llm = Llama(
            model_path=str(resolved),
            n_ctx=self.n_ctx,
            verbose=bool(verbose),
        )

    def parse(self, utterance: str) -> Intent:
        """Map a transcript to a validated ``Intent`` (JSON only, no chatter)."""
        text = clean_transcript(utterance)
        if not text:
            return Intent(intent="UNKNOWN", target="", value="", confidence=0.0)
        raw = self._generate(text)
        return parse_intent_json(raw)

    def _generate(self, utterance: str) -> str:
        kwargs: dict[str, Any] = {
            "max_tokens": 256,
            "temperature": 0.0,
            "grammar": self._grammar,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ]
        if hasattr(self._llm, "create_chat_completion"):
            result = self._llm.create_chat_completion(messages=messages, **kwargs)
            return _completion_text(result)
        if hasattr(self._llm, "create_completion"):
            prompt = f"{SYSTEM_PROMPT}\n\nUtterance: {utterance}\nJSON:"
            result = self._llm.create_completion(prompt=prompt, **kwargs)
            return _completion_text(result)
        result = self._llm(utterance, **kwargs)
        return _completion_text(result)


class IntentEngine:
    """STT → constrained SLM pipeline used during local inference."""

    def __init__(
        self,
        *,
        transcriber: Optional[LocalTranscriber] = None,
        parser: Optional[LocalIntentParser] = None,
        transcribe_model: Optional[Any] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.transcriber = transcriber or LocalTranscriber(model=transcribe_model)
        self.parser = parser or LocalIntentParser(llm=llm)

    def process_pcm(self, pcm: np.ndarray) -> Intent:
        """Transcribe an in-memory PCM buffer, then parse a structured intent."""
        transcript = self.transcriber.transcribe(pcm)
        return self.parser.parse(transcript)


def _completion_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)
    choices = result.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message") or {}
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
        if choice.get("text") is not None:
            return str(choice["text"])
    return ""
