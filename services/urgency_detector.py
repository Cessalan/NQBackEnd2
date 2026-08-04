"""
urgency_detector.py
Detects exam pressure, low confidence and distress in the student's message.

Why this exists
---------------
The tutor used to answer "I have an exam in 2 days and I know nothing" the same
way it answers "quiz me on cardiac drugs": by asking which topic, or by firing
straight into generation. The student's actual situation — no time, no baseline,
panicking — never reached the model in a form it reliably acted on, so the reply
read as a feature menu instead of support.

This module turns that situation into structured signals the orchestrator can
act on BEFORE any content tool runs.

Design choices
--------------
- Keyword/regex, not an LLM call. It runs on every message, so it must add no
  latency and cost nothing. The signals it looks for are stated plainly by
  students ("exam tomorrow", "I know nothing", "panicking"), which is exactly
  the case keyword matching handles well.
- Accent-insensitive and multilingual (en / fr / es), because the app answers in
  the language of the incoming message and French users routinely type without
  accents ("stresse", "je sais rien").
- `has_explicit_request` is a deliberate brake: a student who already said what
  they want ("quiz me on X") must not be intercepted by a support flow they did
  not ask for.

The orchestrator consumes `is_crisis` (hijack the turn, acknowledge first) and
feeds the rest of the dict into the system prompt so ordinary turns are still
calibrated to the deadline.
"""

import re
import unicodedata

# ── Vocabulary ───────────────────────────────────────────────────────────────

_NUMBER_WORDS = {
    "a": 1, "an": 1,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}

# A timeline only counts when the message is actually about an exam — otherwise
# "I study 2 days a week" would read as a deadline.
_EXAM_WORDS = (
    "exam", "examen", "test", "midterm", "final", "finals", "partiel",
    "evaluation", "controle", "prueba", "nclex", "oiiq", "licensure", "boards",
)

_TODAY_WORDS = ("today", "tonight", "aujourd", "ce soir", "hoy", "esta noche")
_TOMORROW_WORDS = ("tomorrow", "demain", "manana")
_NEXT_WEEK_WORDS = ("next week", "semaine prochaine", "proxima semana", "semana proxima")

_DAY_UNIT = r"(?:day|days|jour|jours|dia|dias)"
_WEEK_UNIT = r"(?:week|weeks|semaine|semaines|semana|semanas)"

_IN_DAYS = re.compile(rf"\b(?:in|dans|en|within|d ?ici)\s+(\w+)\s+{_DAY_UNIT}\b")
_BARE_DAYS = re.compile(rf"\b(\w+)\s+{_DAY_UNIT}\b")
_IN_WEEKS = re.compile(rf"\b(?:in|dans|en|within|d ?ici)\s+(\w+)\s+{_WEEK_UNIT}\b")

# "I have not opened the book yet" — no baseline to build on.
_LOW_CONFIDENCE = (
    "know nothing", "don t know anything", "dont know anything", "know none",
    "no idea", "clueless", "from scratch", "havent studied", "haven t studied",
    "have not studied", "never studied", "didnt study", "didn t study",
    "did not study", "understand nothing", "dont understand anything",
    "don t understand anything", "so behind", "way behind", "totally lost",
    "completely lost", "starting from zero", "zero prep", "not prepared",
    "unprepared",
    "je ne sais rien", "je sais rien", "sais rien", "aucune idee",
    "je n ai rien etudie", "j ai rien etudie", "rien etudie", "pas etudie",
    "jamais etudie", "je comprends rien", "je ne comprends rien", "a zero",
    "depuis zero", "pas prete", "pas pret", "completement perdu",
    "completement perdue", "je suis perdu", "je suis perdue",
    "no se nada", "ni idea", "no he estudiado", "nunca he estudiado",
    "desde cero", "no entiendo nada", "estoy perdido", "estoy perdida",
    "no estoy preparado", "no estoy preparada",
)

_DISTRESS = (
    "panic", "panicking", "freaking out", "freaking", "stressed", "stressing",
    "so stressed", "anxious", "anxiety", "overwhelmed", "scared", "terrified",
    "desperate", "crying", "can t cope", "cant cope", "losing it", "help me",
    "i m screwed", "im screwed", "gonna fail", "going to fail", "afraid",
    "panique", "je panique", "stresse", "stressee", "du stress", "anxieux",
    "anxieuse", "angoisse", "angoissee", "deborde", "debordee", "j ai peur",
    "jai peur", "peur de rater", "desespere", "desesperee", "au secours",
    "je vais couler", "je vais rater",
    "panico", "estresado", "estresada", "ansiedad", "ansioso", "ansiosa",
    "agobiado", "agobiada", "tengo miedo", "desesperado", "desesperada",
    "voy a reprobar",
)

# The student already said what they want. Do not hijack their turn.
_EXPLICIT_REQUEST = (
    "quiz me", "test me", "give me questions", "make me a quiz", "create a quiz",
    "practice questions", "mcq", "sata", "case study", "flashcard", "flash card",
    "study sheet", "study guide", "cheat sheet", "summarize", "summarise",
    "explain", "study plan",
    "teste moi", "quiz moi", "fais moi un quiz", "des questions", "carte",
    "cartes", "fiche", "resume", "explique", "plan d etude", "plan d etudes",
    "examine me", "examiname", "hazme un quiz", "preguntas", "tarjeta",
    "tarjetas", "resumen", "explica", "plan de estudio",
)


def _normalize(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace so one pattern list
    matches 'stressé', 'stresse' and 'STRESSE' alike."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Apostrophes vary by keyboard (' vs ’); treat them all as spaces so
    # "j'ai rien etudie" and "j ai rien etudie" match the same pattern.
    stripped = re.sub(r"['’ʼ]", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _to_int(token: str):
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _extract_days(text: str):
    """Days until the exam, or None. 0 means today."""
    if not any(w in text for w in _EXAM_WORDS):
        return None

    if any(w in text for w in _TODAY_WORDS):
        return 0
    if any(w in text for w in _TOMORROW_WORDS):
        return 1

    for pattern in (_IN_DAYS, _BARE_DAYS):
        match = pattern.search(text)
        if match:
            value = _to_int(match.group(1))
            if value is not None:
                return value

    match = _IN_WEEKS.search(text)
    if match:
        value = _to_int(match.group(1))
        if value is not None:
            return value * 7

    if any(w in text for w in _NEXT_WEEK_WORDS):
        return 7

    return None


def detect_urgency(user_message: str) -> dict:
    """
    Read exam pressure out of a single message.

    Returns:
        days_to_exam        int | None   0 = today, 1 = tomorrow
        confidence_baseline "very_low" | None
        emotional_state     "panicking" | None
        has_explicit_request bool        student already named what they want
        is_crisis           bool         acknowledge before generating anything
        signals             list[str]    matched phrases, for logs
    """
    text = _normalize(user_message or "")
    if not text:
        return {
            "days_to_exam": None,
            "confidence_baseline": None,
            "emotional_state": None,
            "has_explicit_request": False,
            "is_crisis": False,
            "signals": [],
        }

    signals = []

    days = _extract_days(text)
    if days is not None:
        signals.append(f"days_to_exam={days}")

    low_confidence_hits = [p for p in _LOW_CONFIDENCE if p in text]
    distress_hits = [p for p in _DISTRESS if p in text]
    request_hits = [p for p in _EXPLICIT_REQUEST if p in text]

    signals.extend(f"low_confidence:{p}" for p in low_confidence_hits[:3])
    signals.extend(f"distress:{p}" for p in distress_hits[:3])
    signals.extend(f"explicit_request:{p}" for p in request_hits[:2])

    confidence_baseline = "very_low" if low_confidence_hits else None
    emotional_state = "panicking" if distress_hits else None
    has_explicit_request = bool(request_hits)

    urgent_timeline = days is not None and days <= 3

    # Two ways in: a near deadline the student feels unready for, or an explicit
    # "I know nothing and I'm panicking" with no deadline named yet. A deadline
    # alone is not a crisis — plenty of students are calm and organised about it.
    is_crisis = (not has_explicit_request) and (
        (urgent_timeline and (confidence_baseline or emotional_state))
        or (confidence_baseline and emotional_state)
    )

    return {
        "days_to_exam": days,
        "confidence_baseline": confidence_baseline,
        "emotional_state": emotional_state,
        "has_explicit_request": has_explicit_request,
        "is_crisis": bool(is_crisis),
        "signals": signals,
    }
