"""Unit tests for offline STT formatting and JSON intent validation.

Uses mock Whisper / llama.cpp outputs only — no microphone, no GGUF weights,
no cloud APIs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import (
    SAMPLE_RATE,
    SLM_N_CTX,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_MODEL_NAME,
)
from models.intent_engine import (
    ALLOWED_INTENTS,
    INTENT_GBNF,
    INTENT_JSON_SCHEMA,
    Intent,
    IntentEngine,
    IntentParseError,
    LocalIntentParser,
    LocalTranscriber,
    SYSTEM_PROMPT,
    clean_transcript,
    find_local_gguf,
    find_local_whisper,
    parse_intent_json,
)


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------
def _pcm_sine(n: int = SAMPLE_RATE, freq: float = 440.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = amplitude * np.sin(2.0 * np.pi * freq * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _intent_json(
    intent: str = "TOGGLE_DEVICE",
    target: str = "lights",
    value: str | int | float = "on",
    confidence: float = 0.92,
) -> str:
    return json.dumps(
        {"intent": intent, "target": target, "value": value, "confidence": confidence}
    )


class ScriptedWhisper:
    """faster-whisper stand-in that yields scripted segment text."""

    def __init__(self, segments: list[str], *, as_generator: bool = False) -> None:
        self.segments = list(segments)
        self.as_generator = as_generator
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": np.asarray(audio), "kwargs": dict(kwargs)})

        def _gen():
            for text in self.segments:
                yield SimpleNamespace(text=text)

        info = SimpleNamespace(language="en", language_probability=1.0)
        if self.as_generator:
            return _gen(), info
        return [SimpleNamespace(text=t) for t in self.segments], info


class ScriptedLlama:
    """llama.cpp stand-in that returns scripted chat-completion JSON."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []
        self._i = 0

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        idx = min(self._i, len(self.outputs) - 1)
        self._i += 1
        return {
            "choices": [
                {"message": {"role": "assistant", "content": self.outputs[idx]}}
            ]
        }


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------
def test_clean_transcript_collapses_whitespace_and_strips():
    assert clean_transcript("  turn   on\nthe\tlights  ") == "turn on the lights"
    assert clean_transcript("") == ""
    assert clean_transcript("   ") == ""
    assert clean_transcript('"quoted utterance"') == "quoted utterance"


def test_transcriber_joins_segments_into_clean_text():
    whisper = ScriptedWhisper(["  Turn on ", " the   lights. "])
    stt = LocalTranscriber(model=whisper)
    text = stt.transcribe(_pcm_sine())
    assert text == "Turn on the lights."


def test_transcriber_accepts_segment_generator():
    whisper = ScriptedWhisper(["set", " volume ", " to 20"], as_generator=True)
    stt = LocalTranscriber(model=whisper)
    assert stt.transcribe(_pcm_sine(800)) == "set volume to 20"


def test_transcriber_empty_pcm_returns_empty_string():
    whisper = ScriptedWhisper(["should not run"])
    stt = LocalTranscriber(model=whisper)
    assert stt.transcribe(np.zeros(0, dtype=np.int16)) == ""
    assert whisper.calls == []


def test_transcriber_converts_int16_pcm_in_memory():
    whisper = ScriptedWhisper(["ok"])
    stt = LocalTranscriber(model=whisper)
    pcm = np.full(1600, 16384, dtype=np.int16)
    assert stt.transcribe(pcm) == "ok"
    audio = whisper.calls[0]["audio"]
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    np.testing.assert_allclose(float(audio.mean()), 0.5, atol=1e-3)
    assert whisper.calls[0]["kwargs"]["language"] == "en"
    assert whisper.calls[0]["kwargs"]["vad_filter"] is False


def test_transcriber_does_not_write_audio_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    whisper = ScriptedWhisper(["offline"])
    stt = LocalTranscriber(model=whisper)
    stt.transcribe(_pcm_sine(480))
    leftover = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftover == []


# ---------------------------------------------------------------------------
# Local model path resolution / offline constructors
# ---------------------------------------------------------------------------
def test_missing_stt_model_raises(tmp_path):
    missing = tmp_path / "stt"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="Whisper STT"):
        LocalTranscriber(download_root=missing, model_name="base.en")


def test_missing_gguf_raises(tmp_path):
    missing = tmp_path / "nope.gguf"
    with pytest.raises(FileNotFoundError, match="SLM GGUF"):
        LocalIntentParser(model_path=missing, search_dir=tmp_path)


def test_find_local_whisper_named_directory(tmp_path):
    model_dir = tmp_path / "base.en"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"stub")
    assert find_local_whisper(tmp_path, "base.en") == model_dir


def test_find_local_gguf_single_file(tmp_path):
    gguf = tmp_path / "Phi-3-mini-4k-instruct.Q4_K_M.gguf"
    gguf.write_bytes(b"stub")
    assert find_local_gguf(tmp_path / "model.gguf", tmp_path) == gguf


def test_stt_config_is_cpu_int8_english():
    assert STT_DEVICE == "cpu"
    assert STT_COMPUTE_TYPE == "int8"
    assert STT_MODEL_NAME in {"tiny.en", "base.en"}
    assert SLM_N_CTX == 2048


def test_whisper_constructed_offline_from_local_dir(monkeypatch, tmp_path):
    captured: dict = {}

    class FakeWhisper:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured.update(kwargs)

        def transcribe(self, audio, **kwargs):
            return [], None

    model_dir = tmp_path / "tiny.en"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"stub")
    monkeypatch.setattr("models.intent_engine.WhisperModel", FakeWhisper)

    stt = LocalTranscriber(model_name="tiny.en", download_root=tmp_path)
    assert captured["name"] == str(model_dir)
    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "int8"
    assert captured["local_files_only"] is True
    assert "download_root" not in captured
    assert stt.model_path == model_dir


def test_llama_constructed_offline_from_gguf(monkeypatch, tmp_path):
    captured: dict = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_chat_completion(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": _intent_json("UNKNOWN", "", "", 0.0)
                        }
                    }
                ]
            }

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"stub")
    monkeypatch.setattr("models.intent_engine.Llama", FakeLlama)
    monkeypatch.setattr("models.intent_engine.LlamaGrammar", None)

    parser = LocalIntentParser(model_path=gguf, search_dir=tmp_path)
    assert captured["model_path"] == str(gguf)
    assert captured["n_ctx"] == 2048
    assert captured["verbose"] is False
    assert parser.model_path == gguf


# ---------------------------------------------------------------------------
# JSON intent schema / GBNF
# ---------------------------------------------------------------------------
INTENT_ENUM_TOKENS = ("TOGGLE_DEVICE", "QUERY_STATUS", "ADJUST_SETTING", "UNKNOWN")


def test_gbnf_and_schema_cover_prompt_fields():
    for token in INTENT_ENUM_TOKENS:
        assert token in INTENT_GBNF
        assert token in INTENT_JSON_SCHEMA["properties"]["intent"]["enum"]
    for key in ("intent", "target", "value", "confidence"):
        assert key in INTENT_GBNF
        assert rf"\"{key}\"" in INTENT_GBNF
        assert key in INTENT_JSON_SCHEMA["required"]
    assert INTENT_JSON_SCHEMA["additionalProperties"] is False
    assert ALLOWED_INTENTS == set(INTENT_JSON_SCHEMA["properties"]["intent"]["enum"])


def test_parse_valid_intent_json():
    payload = parse_intent_json(_intent_json())
    assert payload == Intent(
        intent="TOGGLE_DEVICE", target="lights", value="on", confidence=0.92
    )
    assert payload.to_dict()["intent"] == "TOGGLE_DEVICE"


@pytest.mark.parametrize(
    "intent_name",
    ["TOGGLE_DEVICE", "QUERY_STATUS", "ADJUST_SETTING", "UNKNOWN"],
)
def test_all_allowed_intents_validate(intent_name):
    parsed = parse_intent_json(
        _intent_json(intent=intent_name, target="fan", value=1, confidence=1.0)
    )
    assert parsed.intent == intent_name
    assert parsed.value == 1
    assert parsed.confidence == pytest.approx(1.0)


def test_numeric_value_and_confidence_bounds():
    parsed = parse_intent_json(
        _intent_json(
            intent="ADJUST_SETTING", target="volume", value=20, confidence=0.0
        )
    )
    assert parsed.value == 20
    assert parsed.confidence == pytest.approx(0.0)
    parsed_f = parse_intent_json(
        _intent_json(
            intent="ADJUST_SETTING", target="volume", value=0.5, confidence=0.5
        )
    )
    assert parsed_f.value == pytest.approx(0.5)


def test_invalid_json_raises():
    with pytest.raises(IntentParseError, match="not JSON"):
        parse_intent_json("not json at all")


def test_conversational_text_is_rejected():
    wrapped = 'Sure! Here you go: {"intent":"TOGGLE_DEVICE","target":"lights","value":"on","confidence":0.9}'
    with pytest.raises(IntentParseError, match="not JSON"):
        parse_intent_json(wrapped)
    with pytest.raises(IntentParseError, match="not JSON"):
        parse_intent_json("```json\n" + _intent_json() + "\n```")
    with pytest.raises(IntentParseError, match="empty"):
        parse_intent_json("   ")


def test_unknown_intent_name_rejected():
    with pytest.raises(IntentParseError, match="intent must be one of"):
        parse_intent_json(_intent_json(intent="PLAY_MUSIC"))


def test_extra_keys_rejected():
    raw = json.dumps(
        {
            "intent": "UNKNOWN",
            "target": "",
            "value": "",
            "confidence": 0.1,
            "chatter": "hello",
        }
    )
    with pytest.raises(IntentParseError, match="unexpected"):
        parse_intent_json(raw)


def test_missing_fields_rejected():
    with pytest.raises(IntentParseError, match="missing"):
        parse_intent_json('{"intent":"UNKNOWN","target":"","value":""}')


def test_confidence_out_of_range_rejected():
    with pytest.raises(IntentParseError, match="confidence"):
        parse_intent_json(_intent_json(confidence=1.2))
    with pytest.raises(IntentParseError, match="confidence"):
        parse_intent_json(_intent_json(confidence=-0.01))


def test_value_rejects_bool_and_null():
    with pytest.raises(IntentParseError, match="value"):
        parse_intent_json(
            '{"intent":"UNKNOWN","target":"","value":true,"confidence":0.1}'
        )
    with pytest.raises(IntentParseError, match="value"):
        parse_intent_json(
            '{"intent":"UNKNOWN","target":"","value":null,"confidence":0.1}'
        )


def test_json_array_rejected():
    with pytest.raises(IntentParseError, match="object"):
        parse_intent_json("[]")


# ---------------------------------------------------------------------------
# LocalIntentParser with mocked GGUF
# ---------------------------------------------------------------------------
def test_parser_returns_validated_intent_from_mock_llm():
    llm = ScriptedLlama([_intent_json("QUERY_STATUS", "thermostat", "current", 0.81)])
    parser = LocalIntentParser(llm=llm)
    result = parser.parse("what is the thermostat set to")
    assert result.intent == "QUERY_STATUS"
    assert result.target == "thermostat"
    assert result.value == "current"
    assert result.confidence == pytest.approx(0.81)


def test_parser_passes_gbnf_grammar_and_json_only_system_prompt():
    llm = ScriptedLlama([_intent_json()])
    parser = LocalIntentParser(llm=llm)
    parser.parse("turn on the lights")
    call = llm.calls[0]
    assert call["kwargs"]["grammar"] == INTENT_GBNF
    assert call["kwargs"]["temperature"] == 0.0
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]
    assert "JSON" in call["messages"][0]["content"]
    assert "markdown" in SYSTEM_PROMPT.lower() or "commentary" in SYSTEM_PROMPT.lower()
    assert call["messages"][1]["content"] == "turn on the lights"


def test_parser_rejects_mock_conversational_output():
    llm = ScriptedLlama(["I think you want to toggle the lights."])
    parser = LocalIntentParser(llm=llm)
    with pytest.raises(IntentParseError):
        parser.parse("lights please")


def test_parser_empty_utterance_is_unknown_without_llm():
    llm = ScriptedLlama([_intent_json()])
    parser = LocalIntentParser(llm=llm)
    result = parser.parse("   ")
    assert result.intent == "UNKNOWN"
    assert result.confidence == pytest.approx(0.0)
    assert llm.calls == []


# ---------------------------------------------------------------------------
# End-to-end pipeline (mocked models)
# ---------------------------------------------------------------------------
def test_intent_engine_pcm_to_structured_json():
    whisper = ScriptedWhisper(["  turn off ", "the kitchen lights "])
    llm = ScriptedLlama(
        [_intent_json("TOGGLE_DEVICE", "kitchen lights", "off", 0.88)]
    )
    engine = IntentEngine(
        transcriber=LocalTranscriber(model=whisper),
        parser=LocalIntentParser(llm=llm),
    )
    intent = engine.process_pcm(_pcm_sine())
    assert intent.intent == "TOGGLE_DEVICE"
    assert intent.target == "kitchen lights"
    assert intent.value == "off"
    assert 0.0 <= intent.confidence <= 1.0
    dumped = json.loads(intent.to_json())
    assert dumped == intent.to_dict()
    assert llm.calls[0]["messages"][1]["content"] == "turn off the kitchen lights"
