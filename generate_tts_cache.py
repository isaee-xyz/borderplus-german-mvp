import json, pathlib, hashlib
from gtts import gTTS
from pydub import AudioSegment

CORPUS_PATH = pathlib.Path("phrases_de.json")
CACHE_DIR   = pathlib.Path("tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

def text_to_filename(text: str) -> pathlib.Path:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{h}.mp3"

def main():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf8"))
    for entry in corpus:
        target = text_to_filename(entry["text"])
        if target.exists():
            continue
        print(f"Generating {target.name} …")
        tts = gTTS(entry["text"], lang="de")
        tts.save(target)

        # ensure 16 kHz mono for consistent playback
        wav = AudioSegment.from_file(target)
        wav = wav.set_frame_rate(16000).set_channels(1)
        wav.export(target, format="mp3")
    print("✅  TTS cache complete.")

if __name__ == "__main__":
    main()
