"""Fetch lightweight local model weights into ./models for offline testing.

Usage (from repo root, after `pip install -r requirements.txt`):

    python scripts/download_models.py

Downloads:
  - CTranslate2 Whisper tiny.en  -> models/stt/whisper-tiny.en/
  - Qwen2.5-0.5B Instruct Q4_K_M -> models/slm/qwen2.5-0.5b-instruct-q4_k_m.gguf
  - KWS ONNX stub                -> models/kws/kws_model.onnx (if missing)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

STT_REPO = "Systran/faster-whisper-tiny.en"
STT_DIR = ROOT / "models" / "stt" / "whisper-tiny.en"
STT_MARKER = STT_DIR / "model.bin"

SLM_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
SLM_REMOTE_NAMES = (
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
)
SLM_DIR = ROOT / "models" / "slm"
SLM_DEST = SLM_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SLM_MIN_BYTES = 50 * 1024 * 1024  # incomplete / placeholder files are smaller

KWS_DIR = ROOT / "models" / "kws"
KWS_MODEL = KWS_DIR / "kws_model.onnx"
VAD_MODEL = KWS_DIR / "silero_vad.onnx"


def _print(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dirs() -> None:
    for path in (STT_DIR, SLM_DIR, KWS_DIR):
        path.mkdir(parents=True, exist_ok=True)
        _print(f"[dir] {path.relative_to(ROOT)}")


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def _download_stt() -> None:
    if _file_ok(STT_MARKER, min_bytes=1024):
        _print(f"[skip] STT already present: {STT_DIR.relative_to(ROOT)}")
        return

    _print(f"[stt] Downloading {STT_REPO} -> {STT_DIR.relative_to(ROOT)}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None

    if snapshot_download is not None:
        snapshot_download(
            repo_id=STT_REPO,
            local_dir=str(STT_DIR),
        )
    else:
        from faster_whisper.utils import download_model

        download_model("tiny.en", output_dir=str(STT_DIR))

    if not _file_ok(STT_MARKER):
        raise FileNotFoundError(
            f"STT download finished but {STT_MARKER} is missing. "
            "Check the Hugging Face repo contents."
        )
    size_mb = STT_MARKER.stat().st_size / (1024 * 1024)
    _print(f"[stt] Ready ({size_mb:.1f} MiB model.bin) at {STT_DIR.relative_to(ROOT)}")


def _download_slm() -> None:
    if _file_ok(SLM_DEST, min_bytes=SLM_MIN_BYTES):
        size_mb = SLM_DEST.stat().st_size / (1024 * 1024)
        _print(f"[skip] SLM already present ({size_mb:.1f} MiB): {SLM_DEST.relative_to(ROOT)}")
        return

    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    downloaded: Path | None = None
    for filename in SLM_REMOTE_NAMES:
        _print(f"[slm] Downloading {SLM_REPO}/{filename}")
        try:
            raw = hf_hub_download(
                repo_id=SLM_REPO,
                filename=filename,
                local_dir=str(SLM_DIR),
            )
            downloaded = Path(raw)
            break
        except Exception as exc:  # noqa: BLE001 — try the alternate remote name
            last_error = exc
            _print(f"[slm] {filename} not available ({exc})")

    if downloaded is None:
        raise RuntimeError(
            f"Could not download Q4_K_M GGUF from {SLM_REPO}"
        ) from last_error

    if downloaded.resolve() != SLM_DEST.resolve():
        _print(f"[slm] Saving as {SLM_DEST.name}")
        if SLM_DEST.exists():
            SLM_DEST.unlink()
        shutil.move(str(downloaded), str(SLM_DEST))

    if not _file_ok(SLM_DEST, min_bytes=SLM_MIN_BYTES):
        raise FileNotFoundError(f"SLM download finished but {SLM_DEST} is incomplete")
    size_mb = SLM_DEST.stat().st_size / (1024 * 1024)
    _print(f"[slm] Ready ({size_mb:.1f} MiB) at {SLM_DEST.relative_to(ROOT)}")


def _kws_graph_ok(path: Path) -> bool:
    """True when the ONNX graph is a single-tensor KWS model (not Silero VAD)."""
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return len(session.get_inputs()) == 1
    except Exception:
        return False


def _ensure_kws() -> None:
    if _file_ok(KWS_MODEL) and _kws_graph_ok(KWS_MODEL):
        _print(f"[skip] KWS already present: {KWS_MODEL.relative_to(ROOT)}")
        return

    if _file_ok(VAD_MODEL) and _kws_graph_ok(VAD_MODEL):
        _print(
            f"[kws] Copying {VAD_MODEL.relative_to(ROOT)} -> {KWS_MODEL.relative_to(ROOT)}"
        )
        shutil.copy2(VAD_MODEL, KWS_MODEL)
        _print(f"[kws] Ready at {KWS_MODEL.relative_to(ROOT)}")
        return

    if _file_ok(VAD_MODEL) and _file_ok(KWS_MODEL):
        _print(
            "[kws] Existing kws_model.onnx matches Silero VAD I/O; "
            "replacing with a KWS dummy so the loader accepts a single feature tensor"
        )

    _print(f"[kws] Writing ONNX dummy -> {KWS_MODEL.relative_to(ROOT)}")
    from onnx_stub import write_kws_model_stub

    write_kws_model_stub(KWS_MODEL)
    _print(f"[kws] Ready (dummy) at {KWS_MODEL.relative_to(ROOT)}")


def main() -> int:
    _print(f"Project root: {ROOT}")
    _ensure_dirs()
    try:
        _download_stt()
        _download_slm()
        _ensure_kws()
    except Exception as exc:  # noqa: BLE001 — surface a clear download error
        _print(f"[error] {exc}")
        return 1

    _print("")
    _print("Done. Local files:")
    for path in (STT_MARKER, SLM_DEST, KWS_MODEL):
        if path.is_file():
            _print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size} bytes)")
        else:
            _print(f"  MISSING {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
