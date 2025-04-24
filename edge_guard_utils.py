import io
from typing import Tuple

import numpy as np
import soundfile as sf
import webrtcvad, numpy as np, soundfile as sf, io

VAD = webrtcvad.Vad(2)           # mode 2 ≈ medium aggressiveness

def trim_silence_pcm16(wav_bytes: bytes) -> bytes:
    """Keep only voiced 30-ms frames -> less noise before ASR."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    if sr != 16000:
        return wav_bytes         # we resample later; skip VAD
    frame_len = int(0.03 * sr)
    voiced = []
    for i in range(0, len(data), frame_len):
        frame = data[i : i + frame_len]
        if len(frame) < frame_len:
            break
        if VAD.is_speech(frame.tobytes(), sample_rate=sr):
            voiced.append(frame)
    if not voiced:
        return wav_bytes
    processed = np.concatenate(voiced)
    buf = io.BytesIO()
    sf.write(buf, processed, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()

# --------------
# Configuration
# --------------
SILENCE_DB_THRESHOLD: float = -40.0  # dBFS; below this we treat as silence
MIN_DURATION_S: float = 0.6  # ignore utterances shorter than this
MAX_CLIPPING_RATIO: float = 0.05  # >5 % clipped samples → distorted recording

# --------------
# Core helpers
# --------------

def _audio_from_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes → mono float32 signal & sample‑rate."""
    data, sr = sf.read(io.BytesIO(wav_bytes))
    # down‑mix stereo → mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # always work in float32 for dB calc
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, sr


def _to_dbfs(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    db = 20 * np.log10(rms)
    return db


# --------------
# Public API
# --------------

def is_too_short(wav_bytes: bytes, min_duration: float = MIN_DURATION_S) -> bool:
    data, sr = _audio_from_bytes(wav_bytes)
    return len(data) / sr < min_duration


def is_silence(wav_bytes: bytes, thresh_db: float = SILENCE_DB_THRESHOLD) -> bool:
    data, _ = _audio_from_bytes(wav_bytes)
    return _to_dbfs(data) < thresh_db


def is_clipping(wav_bytes: bytes, clip_ratio: float = MAX_CLIPPING_RATIO) -> bool:
    data, _ = _audio_from_bytes(wav_bytes)
    clipped = np.sum(np.abs(data) >= 0.98)  # near‑full‑scale samples
    return clipped / len(data) > clip_ratio


def validate_audio(wav_bytes: bytes):
    """High‑level guard. Returns (valid: bool, error_msg: str|None)."""
    if is_too_short(wav_bytes):
        return False, "Audio too short; please speak the full phrase."
    if is_silence(wav_bytes):
        return False, "Silence detected; please record again."
    if is_clipping(wav_bytes):
        return False, "Recording is distorted; try speaking a bit softer."
    return True, None
