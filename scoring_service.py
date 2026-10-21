def compute_scores(transcript: dict, nlp: dict, audio: dict, eye: dict) -> dict:

    # Grammar (0-100): penalise per error relative to word count
    words       = max(transcript.get("word_count", 1), 1)
    err_rate    = nlp.get("grammar_errors", 0) / words
    grammar     = max(0, 100 - err_rate * 500)

    # Vocabulary (0-100): lexical diversity × 100, bounded
    vocabulary  = min(nlp.get("lexical_diversity", 0) * 130, 100)

    # Filler words (0-100): penalise high filler rate
    filler_rate = nlp.get("filler_rate_per_min", 0)
    fillers     = max(0, 100 - filler_rate * 12)

    # Eye contact (0-100): directly from on-camera %
    eye_contact = min(eye.get("on_camera_pct", 0), 100)

    # Fluency (0-100): penalise pauses and extremes of speaking rate
    wpm         = audio.get("speaking_rate_wpm", 130)
    ideal_wpm   = 130
    wpm_penalty = abs(wpm - ideal_wpm) / 2
    pause_pen   = min(audio.get("long_pauses", 0) * 3, 30)
    fluency     = max(0, 100 - wpm_penalty - pause_pen)

    # Confidence (0-100): combination of pitch range, eye contact, fluency
    pitch_range = audio.get("pitch_range_hz", 0)
    confidence  = (
        0.4 * min(pitch_range / 1.5, 100) +
        0.3 * eye_contact +
        0.3 * fluency
    )

    # Overall weighted
    overall = (
        grammar    * 0.20 +
        vocabulary * 0.15 +
        fillers    * 0.15 +
        eye_contact* 0.20 +
        fluency    * 0.15 +
        confidence * 0.15
    )

    def r(x): return round(min(max(x, 0), 100), 1)

    return {
        "overall"    : r(overall),
        "grammar"    : r(grammar),
        "vocabulary" : r(vocabulary),
        "confidence" : r(confidence),
        "eye_contact": r(eye_contact),
        "fluency"    : r(fluency),
        "filler_words": r(fillers)
    }
