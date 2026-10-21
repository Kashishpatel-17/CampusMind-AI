"""
nlp_service.py  —  Pure built-in Python NLP (no spaCy required)
Analysed: vocabulary, filler words, grammar approximation
"""
import re
from collections import Counter

_READY = False

FILLER_WORDS = {
    "um", "uh", "like", "you know", "so", "basically", "actually",
    "literally", "right", "okay", "well", "kind of", "sort of",
    "i mean", "you see"
}

# Common English stop-words (small set for diversity calc)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of",
    "with","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may",
    "might","shall","can","i","you","he","she","it","we","they",
    "my","your","his","her","its","our","their","this","that","these",
    "those","not","no","nor","so","yet","both","either","neither",
    "just","also","than","then","when","where","who","which","what",
    "how","if","because","as","until","while","although","though"
}


def load_nlp():
    """No model to load — pure Python."""
    global _READY
    _READY = True
    print("   NLP using built-in (no spaCy needed) ✅")


def _tokenize(text: str):
    """Lowercase alpha tokens only."""
    return re.findall(r"[a-z']+", text.lower())


def analyze_nlp(text: str, duration_s: float) -> dict:
    """
    Returns vocabulary & filler-word metrics without any external NLP library.
    """
    tokens = _tokenize(text)
    content_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    unique_content  = set(content_tokens)

    total = max(len(tokens), 1)
    lexical_diversity = round(len(set(tokens)) / total, 4)

    # ── Filler-word detection ──────────────────────────────────────────────
    filler_counts: Counter = Counter()
    text_lower = text.lower()

    # Multi-word fillers first (longest match wins)
    multi_word = [fw for fw in FILLER_WORDS if " " in fw]
    single_word = [fw for fw in FILLER_WORDS if " " not in fw]

    for fw in sorted(multi_word, key=len, reverse=True):
        pattern = r'\b' + re.escape(fw) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count:
            filler_counts[fw] = count

    for fw in single_word:
        pattern = r'\b' + re.escape(fw) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count:
            filler_counts[fw] = count

    filler_count = sum(filler_counts.values())
    minutes      = max(duration_s / 60, 0.01)
    filler_rate  = round(filler_count / minutes, 2)

    # ── Grammar approximation (no external lib) ────────────────────────────
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    grammar_errors = 0
    for sentence in sentences:
        words = sentence.lower().split()
        if not words:
            continue
        # Heuristic: a sentence with < 3 words is likely incomplete
        if len(words) < 3:
            grammar_errors += 1
        # Very basic: sentence starts with a verb-ish word (no subject)
        # Just a rough flag — not a real grammar checker
        verb_starters = {"is","are","was","were","be","been","have","has","had",
                         "do","does","did","will","would","could","should","can"}
        if words[0] in verb_starters:
            grammar_errors += 1

    # ── Word-level stats ───────────────────────────────────────────────────
    avg_word_length = round(
        sum(len(t) for t in content_tokens) / max(len(content_tokens), 1), 2
    )

    return {
        "unique_words"        : len(set(tokens)),
        "unique_content_words": len(unique_content),
        "lexical_diversity"   : lexical_diversity,
        "avg_word_length"     : avg_word_length,
        "filler_words"        : dict(filler_counts),
        "filler_count"        : filler_count,
        "filler_rate_per_min" : filler_rate,
        "grammar_errors"      : grammar_errors,
        "sentence_count"      : len(sentences),
    }