import os, httpx

OLLAMA_URL   = os.getenv("OLLAMA_URL",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","mistral")
ANTHROPIC_KEY= os.getenv("ANTHROPIC_API_KEY","")

def _build_prompt(transcript, nlp, audio, eye, scores):
    return f"""You are an expert communication coach. Analyze the following soft skill data and give structured feedback.

SCORES: Overall={scores['overall']}, Grammar={scores['grammar']}, Vocabulary={scores['vocabulary']}, 
Eye Contact={scores['eye_contact']}, Fluency={scores['fluency']}, Confidence={scores['confidence']}, 
Filler Words={scores['filler_words']}

TRANSCRIPT (first 400 chars): {transcript['text'][:400]}

NLP DATA: unique_words={nlp['unique_words']}, lexical_diversity={nlp['lexical_diversity']:.2f}, 
filler_count={nlp['filler_count']}, filler_rate_per_min={nlp['filler_rate_per_min']}, 
grammar_errors={nlp['grammar_errors']}

AUDIO: speaking_rate_wpm={audio['speaking_rate_wpm']}, avg_pitch_hz={audio['avg_pitch_hz']:.1f}, 
pause_count={audio['pause_count']}, long_pauses={audio['long_pauses']}

EYE CONTACT: on_camera_pct={eye['on_camera_pct']:.1f}%, look_away_events={eye['look_away_events']}

Respond with exactly three sections:
EVALUATION: (2-3 sentence overall assessment)
STRENGTHS: (bullet points of what they did well)
IMPROVEMENTS: (bullet points of specific, actionable improvements)"""


def _try_ollama(prompt):
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                       timeout=60)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception:
        return None


def _try_anthropic(prompt):
    if not ANTHROPIC_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg    = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 600,
            messages   = [{"role":"user","content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def _rule_based(transcript, nlp, audio, eye, scores):
    overall = scores['overall']
    grade   = "Excellent" if overall>=80 else "Good" if overall>=65 else "Average" if overall>=50 else "Needs Improvement"

    strengths = []
    improvements = []

    if scores['eye_contact'] >= 70: strengths.append("Good eye contact — you maintained camera presence well.")
    else: improvements.append(f"Improve eye contact — only {eye['on_camera_pct']:.0f}% on-camera. Practice looking directly at the lens.")

    if scores['grammar'] >= 75: strengths.append("Strong grammar usage with minimal errors.")
    else: improvements.append(f"Reduce grammar errors ({nlp['grammar_errors']} found). Review your speech before recording.")

    if scores['filler_words'] >= 70: strengths.append("Minimal filler word usage — your speech sounds polished.")
    else: improvements.append(f"Reduce filler words ({nlp['filler_count']} detected, {nlp['filler_rate_per_min']}/min). Practice pausing instead of saying 'um' or 'like'.")

    wpm = audio['speaking_rate_wpm']
    if 110 <= wpm <= 160: strengths.append(f"Good speaking pace ({wpm} WPM) — easy to follow.")
    elif wpm < 110: improvements.append(f"Speak slightly faster — {wpm} WPM is too slow. Aim for 120-150 WPM.")
    else: improvements.append(f"Slow down — {wpm} WPM is too fast. Aim for 120-150 WPM.")

    if scores['vocabulary'] >= 70: strengths.append(f"Good vocabulary diversity (lexical diversity: {nlp['lexical_diversity']:.0%}).")
    else: improvements.append("Expand your vocabulary — try using more varied and specific words.")

    s_txt = "\n".join(f"• {s}" for s in strengths) or "• Keep practising to build your strengths."
    i_txt = "\n".join(f"• {i}" for i in improvements) or "• Great job! Maintain your current performance."

    return (
        f"EVALUATION: Your overall soft skill score is {overall:.0f}/100 — {grade}. "
        f"You spoke {wpm} words per minute with {nlp['grammar_errors']} grammar errors and "
        f"{eye['on_camera_pct']:.0f}% camera presence.\n\n"
        f"STRENGTHS:\n{s_txt}\n\n"
        f"IMPROVEMENTS:\n{i_txt}"
    )


def generate_feedback(transcript, nlp, audio, eye, scores) -> str:
    prompt = _build_prompt(transcript, nlp, audio, eye, scores)

    fb = _try_ollama(prompt)
    if fb: return fb

    fb = _try_anthropic(prompt)
    if fb: return fb

    return _rule_based(transcript, nlp, audio, eye, scores)
