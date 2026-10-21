import os, whisper

_model = None

def load_model():
    global _model
    model_name = os.getenv("WHISPER_MODEL", "tiny")
    print(f"   Loading Whisper ({model_name})...")
    _model = whisper.load_model(model_name)
    print(f"   ✅ Whisper ready")

def transcribe(audio_path: str) -> dict:
    if _model is None:
        load_model()
    result    = _model.transcribe(audio_path)
    text      = result["text"].strip()
    language  = result.get("language", "en")
    word_count= len(text.split())
    duration  = result.get("duration", 60)
    return {
        "text"      : text,
        "language"  : language,
        "word_count": word_count,
        "duration_s": duration
    }
