"""
azure_asr.py  ·  Azure Speech helpers (SDK)
-------------------------------------------
• transcribe_stt(bytes)              → transcript string
• assess_pronunciation(bytes, ref)   → dict with transcript, accuracy_score, words[]
"""

from __future__ import annotations
import os, io, json, numpy as np, soundfile as sf, librosa

# ─── credentials ─────────────────────────────────────────────────────────
try:
    import streamlit as st
    AZ_KEY    = st.secrets["azure_speech"]["key"]
    AZ_REGION = st.secrets["azure_speech"]["region"]
except Exception:
    AZ_KEY    = os.getenv("AZURE_SPEECH_KEY", "")
    AZ_REGION = os.getenv("AZURE_SPEECH_REGION", "")
if not AZ_KEY or not AZ_REGION:
    raise RuntimeError("Azure Speech creds missing – set secrets or env vars.")

# ─── audio helper: mono 16-kHz PCM bytes ─────────────────────────────────
def pcm16k(wav_bytes: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(wav_bytes))
    if sr != 16000:
        data = librosa.resample(data.T, orig_sr=sr, target_sr=16000).T
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, 16000, subtype="PCM_16", format="WAV")
    return buf.getvalue()

# ─── SDK common objects (lazy-load) ──────────────────────────────────────
import azure.cognitiveservices.speech as speechsdk

def _speech_cfg():
    cfg = speechsdk.SpeechConfig(subscription=AZ_KEY, region=AZ_REGION)
    cfg.speech_recognition_language = "de-DE"
    return cfg

# ─── 1. plain STT for lexical similarity ─────────────────────────────────
def transcribe_stt(wav_bytes: bytes) -> str:
    audio_stream = speechsdk.audio.PushAudioInputStream()
    audio_cfg    = speechsdk.audio.AudioConfig(stream=audio_stream)
    reco         = speechsdk.SpeechRecognizer(speech_config=_speech_cfg(), audio_config=audio_cfg)
    audio_stream.write(pcm16k(wav_bytes)); audio_stream.close()
    result = reco.recognize_once()
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise RuntimeError(f"Azure STT error: {result.reason}")
    return result.text.strip()

# ─── 2. Pronunciation-Assessment ─────────────────────────────────────────
def assess_pronunciation(wav_bytes: bytes, reference_text: str):
    # audio feed
    stream = speechsdk.audio.PushAudioInputStream()
    audio  = speechsdk.audio.AudioConfig(stream=stream)
    # recognizer + PA config
    reco   = speechsdk.SpeechRecognizer(speech_config=_speech_cfg(), audio_config=audio)
    pa_cfg = speechsdk.PronunciationAssessmentConfig(
        reference_text = reference_text,
        grading_system = speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity    = speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue  = False)
    pa_cfg.apply_to(reco)
    stream.write(pcm16k(wav_bytes)); stream.close()

    result = reco.recognize_once()
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise RuntimeError(f"Azure PA error: {result.reason}")

    pa_res  = speechsdk.PronunciationAssessmentResult(result)
    detail  = json.loads(result.json)
    words   = detail["NBest"][0]["Words"]

    return {
        "transcript":      result.text.strip(),
        "accuracy_score":  pa_res.accuracy_score,
        "fluency_score":   pa_res.fluency_score,
        "words":           words,
    }
