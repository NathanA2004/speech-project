"""Default audio, VAD/KWS, and local-disk model paths.

All model loading must use these relative on-disk paths under ./models.
No cloud APIs or remote download roots are configured here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout (resolved from this file so cwd does not matter)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

KWS_DIR = MODELS_DIR / "kws"
STT_DIR = MODELS_DIR / "stt"
SLM_DIR = MODELS_DIR / "slm"

KWS_MODEL_PATH = KWS_DIR / "kws_model.onnx"
VAD_MODEL_PATH = KWS_DIR / "silero_vad.onnx"
SLM_MODEL_PATH = SLM_DIR / "model.gguf"
STT_DOWNLOAD_ROOT = STT_DIR  # faster-whisper local cache / model files

# ---------------------------------------------------------------------------
# Capture: 16 kHz, 16-bit, mono, 30 ms frames
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
SAMPLE_WIDTH_BYTES = 2
FRAME_DURATION_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480

# PortAudio input latency hint (seconds). Slightly above one frame is more
# stable on Windows than ultra-low latency, while still real-time.
INPUT_LATENCY_S = 0.05

# 3 s pre-trigger context for KWS / utterance start
RING_BUFFER_SECONDS = 3.0
RING_BUFFER_FRAMES = int(RING_BUFFER_SECONDS * 1000 / FRAME_DURATION_MS)  # 100

# Extra queued frames so the consumer can lag a bit without drops
QUEUE_MAX_FRAMES = RING_BUFFER_FRAMES

# ---------------------------------------------------------------------------
# VAD / KWS / intent (used by later modules)
# ---------------------------------------------------------------------------
VAD_SILENCE_TIMEOUT_S = 1.2
KWS_WINDOW_MS = 100
KWS_WAKE_THRESHOLD = 0.85
KWS_MAX_INFERENCE_MS = 15
SLM_N_CTX = 2048
STT_MODEL_NAME = "base.en"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
