# German Buddy – AI Pronunciation Coach  
*A rapid MVP for BorderPlus nurses learning German (April 2025)*  

---

## ✨ What It Does  
Learners pick a phrase, record their reply, and receive—within a few seconds—

* a transcript of what they said  
* colour-coded, phoneme-level pronunciation scores  
* a blended Similarity % (token recall + semantic match)  
* a progress chart across attempts  

The loop turns tutor-driven drill into self-paced practice, freeing instructors to focus on clinical language and confidence-building.

---

## 🏗 Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Front-end** | Streamlit 1.33 + `st-audiorec` | Pure browser; zero JS build |
| **ASR + PA** | Azure Speech SDK | < 1 s latency; per-phoneme scores |
| **Offline** | `faster-whisper-base` (CPU) | Resilience when cloud unreachable |
| **Similarity** | 0.6 × token overlap + 0.4 × SBERT cosine (`distiluse-base-multilingual-cased-v1`) | Balances recall & meaning |
| **Noise Gate** | WebRTC-VAD | Removes silence/background noise |
| **Packaging** | Docker → Streamlit Cloud | One-click deploy, secrets via dashboard |

---

## ⚙️ Challenges Faced

* **Py 3.12 wheel gap** – Azure Speech SDK lacks wheels; pinned runtime to **Python 3.11** via `runtime.txt`.  
* **Recorder reset** – `st-audiorec` retains waveform;
* **Low-bandwidth resilience** – Whisper fallback + client-side VAD keep p95 latency < 5 s on 3 Mbps links.  
* **Key security** – `.streamlit/` ignored; Azure keys injected as encrypted **Streamlit Secrets**.  
* **Similarity accuracy ceiling** – current hybrid scoring works but mis-ranks some paraphrases; needs tuning (weighted tokens, LM embeddings, or BERTScore).  
* **Whisper-base accuracy** – acceptable for fallback yet struggles with strong Indian-accent German; requires fine-tuning on ~40 h accented corpus.

---

## 🚀 To Go Live 

1. **Structured Feedback Loop** – onboard first tutors & learners, capture friction points daily.  
2. **Expand content coverage** – build a larger database of nurse-specific phrases enriched with medical terminology and cultural tips, delivering context-relevant practice.  
3. **Secure Auth & Progress Tracking** – add login (Streamlit-Auth/Firebase) and Postgres to persist streaks & scores.  
4. **Dockerised Build & Scaling** – hardened image, push to Fargate/K8s; health checks + auto-scale.  
5. **Compliance Review** – confirm GDPR logging, audio-retention policy, DPIA for healthcare data.

