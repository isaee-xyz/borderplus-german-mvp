"""
app.py • BorderPlus Pronunciation Coach • Apr-2025
"""

from __future__ import annotations
import io, json, random, hashlib, warnings, os, logging
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
import altair as alt
from sentence_transformers import SentenceTransformer, util
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment
import urllib3
import base64, pathlib
from edge_guard_utils import validate_audio, trim_silence_pcm16
from azure_asr import transcribe_stt, assess_pronunciation

# ── mute noisy logs ────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="huggingface_hub.file_download")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["GRPC_VERBOSITY"] = "NONE"
logging.getLogger("azure").setLevel(logging.ERROR)
# ───────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BorderPlus Pronunciation Coach", layout="centered")

# ── helpers ────────────────────────────────────────────────────────────────
def clean_tokens(t: str) -> list[str]:
    import re; return re.sub(r"[^\wäöüß]", " ", t.lower()).split()

def token_ratio(a: list[str], b: list[str]) -> float:
    return SequenceMatcher(None, a, b).ratio()

def badge(v: int) -> str:
    return "🟢 Excellent" if v >= 85 else "🟡 Good" if v >= 70 else "🔴 Keep practising"

# ── model loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sbert():
    return SentenceTransformer("distiluse-base-multilingual-cased-v1")
SBERT = load_sbert()

WHISPER = None
def whisper_fallback(wav: bytes) -> str:
    global WHISPER
    if WHISPER is None:
        with st.spinner("Lade Offline-Modell …"):
            WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _ = WHISPER.transcribe(io.BytesIO(wav), language="de", beam_size=1)
    return "".join(s.text for s in segs).strip()

# ── corpus & TTS helpers ───────────────────────────────────────────────────
CORPUS = json.loads(Path("phrases_de.json").read_text(encoding="utf8"))
CACHE  = Path("tts_cache"); CACHE.mkdir(exist_ok=True)
def _hash(s): return hashlib.md5(s.encode()).hexdigest()[:10]

def tts_bytes(text: str) -> bytes:
    """Return MP3 bytes, generating & caching with gTTS if missing."""
    path = CACHE / f"{_hash(text)}.mp3"
    if not path.exists():
        gTTS(text, lang="de").save(path)
        try:
            AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1).export(path, format="mp3")
        except Exception:
            pass
    return path.read_bytes()

def sample_ids():
    g=[e["id"] for e in CORPUS if e["type"]=="general"]
    m=[e["id"] for e in CORPUS if e["type"]=="medical"]
    return random.sample(g,3) + random.sample(m,2)

# ── session bootstrap ──────────────────────────────────────────────────────
ss = st.session_state
if "preset_ids"  not in ss: ss.preset_ids  = sample_ids()
if "idx_in_set"  not in ss: ss.idx_in_set  = 0
if "phrase_id"   not in ss: ss.phrase_id   = ss.preset_ids[0]
if "hist_sim"    not in ss: ss.hist_sim    = defaultdict(list)
if "hist_prn"    not in ss: ss.hist_prn    = defaultdict(list)
mp = {e["id"]: e for e in CORPUS}

def load_phrase(pid: str):
    """Update current phrase & clear recorder."""
    ss.phrase_id = pid
    ss.phrase    = mp[pid]["text"]
    ss.pop("wav_bytes", None)

load_phrase(ss.phrase_id)            # ensure text loaded

# ── header ─────────────────────────────────────────────────────────────────
logo_b64 = base64.b64encode(pathlib.Path("assets/borderplus.png").read_bytes()).decode()   # keep the same path/filename
st.markdown(
    f"""
    <h1 style="display:flex; align-items:center; gap:0.5em; margin:0">
        <img src="data:image/png;base64,{logo_b64}" width="80">
        BorderPlus German Buddy
    </h1>
    """,
    unsafe_allow_html=True,
)
#st.title("🚀 BorderPlus Pronunciation Coach")
#st.caption("Listen, repeat, and improve both recall and pronunciation.")

if st.button("🎲 New set"):
    ss.preset_ids = sample_ids()
    ss.idx_in_set = 0
    load_phrase(ss.preset_ids[0])
    st.rerun()

# ── phrase radio ──────────────────────────────────────────────────────────
st.radio(
    "Choose phrase",
    ss.preset_ids,
    index=ss.idx_in_set,
    format_func=lambda i: mp[i]["text"],
    key="phrase_picker",
    on_change=lambda: (
        setattr(ss, "idx_in_set", ss.preset_ids.index(st.session_state["phrase_picker"])),
        load_phrase(st.session_state["phrase_picker"]),
        st.rerun(),
    ),
)

# ── playback & record ─────────────────────────────────────────────────────
st.markdown("---"); st.subheader("🎧 Listen & Repeat")
st.markdown(f"**{ss.phrase}**")
st.audio(tts_bytes(ss.phrase), format="audio/mp3")

st.markdown("### 🎙️ Record")
wav_raw = st_audiorec(); 
if wav_raw is not None:              # remember which phrase the clip belongs to
    ss.rec_phrase = ss.phrase_id
ss.wav_bytes = wav_raw

# ── stop & submit ─────────────────────────────────────────────────────────
disabled_btn = (wav_raw is None) or (ss.get("rec_phrase") != ss.phrase_id)
if st.button("Stop & Submit", disabled=disabled_btn):
    ok, err = validate_audio(wav_raw)
    if not ok: st.warning(err); st.stop()

    prog = st.progress(0, text="Uploading …")
    wav_vad = trim_silence_pcm16(wav_raw)
    prog.progress(50, text="Processing …")

    with st.spinner("Transcribing …"):
        try: hyp = transcribe_stt(wav_vad)
        except Exception: hyp = whisper_fallback(wav_vad)

        sim_pct = 0
        if hyp:
            sim = 0.6*token_ratio(clean_tokens(ss.phrase), clean_tokens(hyp)) + \
                  0.4*util.cos_sim(SBERT.encode(ss.phrase), SBERT.encode(hyp)).item()
            sim_pct = round(sim*100)

        acc, res = None, {}
        try:
            pa = assess_pronunciation(wav_vad, ss.phrase)
            acc, res = pa["accuracy_score"], pa
        except Exception: pass

    prog.progress(100, text="Done"); prog.empty()

    ss.hist_sim[ss.phrase_id].append(sim_pct)
    ss.hist_prn[ss.phrase_id].append(acc or 0)

    # ── feedback ─────────────────────────────────────────────────────────
    st.markdown("---"); st.subheader("🔍 Feedback")
    st.write(f"**Similarity:** {sim_pct}%  → {badge(sim_pct)}")

    if acc is not None:
        st.write(f"**Pronunciation:** {acc:.1f} / 100")
        def _col(v): return "#28a745" if v>=85 else "#ffc107" if v>=70 else "#dc3545"
        for w in res.get("words", []):
            sc = w["PronunciationAssessment"]["AccuracyScore"]
            st.markdown(
                f'<span style="color:{_col(sc)}; font-weight:600">{w["Word"]}</span> '
                f'<span style="opacity:0.6">({int(sc)})</span>',
                unsafe_allow_html=True,
            )

    # similarity explainer after first attempt
    if len(ss.hist_sim[ss.phrase_id]) == 1 and "sim_help" not in ss:
        st.sidebar.header("ℹ️ What is Similarity?")
        st.sidebar.write(
            "* **Recall** – word overlap\n"
            "* **Meaning** – semantic match\n\n"
            "Combined into a 0-100 score. Replay & focus on highlighted words to improve."
        )
        ss.sim_help = True

    # history chart
    st.markdown("### 📈 Attempt history")
    df = pd.DataFrame({
        "Attempt": range(1, len(ss.hist_sim[ss.phrase_id])+1),
        "Similarity": ss.hist_sim[ss.phrase_id],
        "Pronunciation": ss.hist_prn[ss.phrase_id],
    })
    st.altair_chart(
        alt.Chart(df.melt("Attempt", var_name="Metric", value_name="Score"))
        .mark_line(point=True).encode(
            x=alt.X("Attempt:O", title="Attempt"),
            y=alt.Y("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0,100])),
            color="Metric:N",
        ).properties(height=260),
        use_container_width=True,
    )
    st.divider()

    # ── bottom navigation (Retry | Next) ─────────────────────────────────
    col_retry, col_next = st.columns(2)

    with col_retry:
        if st.button("🔄 Retry"):
            load_phrase(ss.phrase_id)
            ss.pop("rec_phrase", None)   
            st.rerun()

    with col_next:
        if st.button("➡ Next"):
            ss.idx_in_set += 1
            if ss.idx_in_set >= len(ss.preset_ids):
                ss.preset_ids = sample_ids()
                ss.idx_in_set = 0
            load_phrase(ss.preset_ids[ss.idx_in_set])
            ss.pop("rec_phrase", None)
            st.rerun()

st.caption("MVP • BorderPlus © 2025")
