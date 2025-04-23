import streamlit as st

st.set_page_config(page_title="BorderPlus Pronunciation MVP", layout="centered")
st.title("🚀 BorderPlus Pronunciation Coach")
st.write("If you can read this on http://localhost:8501, your environment is set!")

try:
    from faster_whisper import WhisperModel
    st.success("faster-whisper imported OK ✅")
except Exception as e:
    st.error(f"Whisper import failed: {e}")

import random, json, pathlib
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
import hashlib
def demo_tts():
    st.subheader("TTS cache demo")
    corpus = json.loads(Path("phrases_de.json").read_text(encoding="utf8"))
    entry  = random.choice(corpus)
    st.write(entry["text"])
    mp3    = Path("tts_cache") / (hashlib.md5(entry["text"].encode()).hexdigest()[:10] + ".mp3")
    audio_bytes = mp3.read_bytes()
    st.audio(audio_bytes, format="audio/mp3")

if st.checkbox("🔊 Try a random cached phrase"):
    demo_tts()
