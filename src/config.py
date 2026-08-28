"""Default audio, VAD/KWS, and local-disk model paths.

All model loading must use these paths under ./models on local disk.
No cloud APIs or remote download roots are configured here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout (resolved from this file so cwd does not matter)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR_RELATIVE = Path("models")
MODELS_DIR = PROJECT_ROOT / MODELS_DIR_RELATIVE

KWS_DIR_RELATIVE = MODELS_DIR_RELATIVE / "kws"
STT_DIR_RELATIVE = MODELS_DIR_RELATIVE / "stt"
SLM_DIR_RELATIVE = MODELS_DIR_RELATIVE / "slm"

KWS_MODEL_RELATIVE = KWS_DIR_RELATIVE / "kws_model.onnx"
VAD_MODEL_RELATIVE = KWS_DIR_RELATIVE / "silero_vad.onnx"
SLM_MODEL_RELATIVE = SLM_DIR_RELATIVE / "model.gguf"

KWS_DIR = PROJECT_ROOT / KWS_DIR_RELATIVE
STT_DIR = PROJECT_ROOT / STT_DIR_RELATIVE
SLM_DIR = PROJECT_ROOT / SLM_DIR_RELATIVE

KWS_MODEL_PATH = PROJECT_ROOT / KWS_MODEL_RELATIVE
VAD_MODEL_PATH = PROJECT_ROOT / VAD_MODEL_RELATIVE
SLM_MODEL_PATH = PROJECT_ROOT / SLM_MODEL_RELATIVE
STT_DOWNLOAD_ROOT = STT_DIR  # faster-whisper local files only: ./models/stt

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
# Silero VAD ONNX native window is 512 samples (32 ms) at 16 kHz. Capture
# still uses 30 ms / 480-sample frames; LocalVADDetector zero-pads to this.
VAD_WINDOW_SAMPLES = 512
VAD_LSTM_LAYERS = 2
VAD_LSTM_HIDDEN = 64  # v4 h/c; v5 uses a combined state of size 128
VAD_SPEECH_THRESHOLD = 0.5

KWS_WINDOW_MS = 100
KWS_WINDOW_SAMPLES = SAMPLE_RATE * KWS_WINDOW_MS // 1000  # 1600
KWS_WAKE_THRESHOLD = 0.85
KWS_MAX_INFERENCE_MS = 15

# MFCC / log-mel for sliding-window KWS (25 ms window, 10 ms hop)
MFCC_N_MFCC = 40
MFCC_N_MELS = 40
MFCC_N_FFT = 512
MFCC_WIN_MS = 25
MFCC_HOP_MS = 10
MFCC_WIN_LENGTH = SAMPLE_RATE * MFCC_WIN_MS // 1000  # 400
MFCC_HOP_LENGTH = SAMPLE_RATE * MFCC_HOP_MS // 1000  # 160
MFCC_PREEMPHASIS = 0.97
MFCC_FRAMES_PER_KWS_WINDOW = 1 + (KWS_WINDOW_SAMPLES - MFCC_WIN_LENGTH) // MFCC_HOP_LENGTH  # 8
SLM_N_CTX = 2048
STT_MODEL_NAME = "base.en"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
