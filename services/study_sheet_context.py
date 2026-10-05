"""Bounded, attributable study-sheet evidence. Pure: no database or model calls."""
import re

from services import student_emphasis

_PRIORITY = re.compile(
    r"teacher|professor|instructor|must know|important|on the exam|focus on|skip|exclude|only |"
    r"more detail|handwritten|struggl|confus|don't understand|do not understand|"
    r"prof(?:esseur)?|enseignant|important|examen|insist|uniquement|seulement|"
    r"ne comprends|difficult|approfond|moins de|plus de", re.I)


def _text(value, limit=1800):
    return str(value or "")[:limit]


def quiz_evidence(message):
    """Normalize old chat quizzes, saved practice, and study-mode attempts.

    Correctness is copied from recorded grading, never inferred from an empty
    answer. Keep first and latest attempts so a corrected retry is visible.
    """
    practice = message.get("practice") or {}
    progress = message.get("quizProgress") or {}
    content = message.get("studyContent") or {}
    questions = message.get("quizData") or content.get("questions") or []
    if isinstance(questions, list) and questions and isinstance(questions[0], dict) and "questions" in questions[0]:
        questions = questions[0]["questions"]
    if not isinstance(questions, list):
        return None
    questions = list(questions)
    stems = {q.get("question") for q in questions if isinstance(q, dict)}
    for q in practice.get("questions") or []:
        if isinstance(q, dict) and q.get("question") not in stems:
            questions.append(q)
            stems.add(q.get("question"))
    latest = practice.get("answers") or progress.get("answers") or {}
    first = practice.get("firstAnswers") or progress.get("firstAttemptAnswers") or {}
    first_statuses = progress.get('firstAttemptStatuses') or (practice.get('snapshot') or {}).get('firstAttemptStatuses') or {}
    latest_statuses = progress.get('questionStatuses') or (practice.get('snapshot') or {}).get('questionStatuses') or {}

    def attempt(raw):
        if raw in ('correct', 'incorrect'):
            return {'isCorrect': raw == 'correct'}
        if not isinstance(raw, dict) or not raw:
            return None
        result = {k: raw[k] for k in (
            "selectedOption", "selectedOptionText", "selectedIndex", "selectedOptions",
            "selectedAnswers", "selectedIndices", "selection", "score", "maxScore", "percentage", "partial", "isPartial") if k in raw}
        correct = raw.get("isCorrect", raw.get("correct"))
        if isinstance(correct, bool):
            result["isCorrect"] = correct
        return result or None

    records = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        original = attempt(first.get(str(i), first.get(i))) or attempt(first_statuses.get(str(i), first_statuses.get(i)))
        answer = (attempt(latest.get(str(i), latest.get(i)))
                  or attempt(latest_statuses.get(str(i), latest_statuses.get(i)))
                  or attempt(q.get("userSelection")) or original)
        records.append({
            "index": i, "topic": _text(q.get("topic") or q.get("category") or (q.get('metadata') or {}).get('topic'), 180),
            "question": _text(q.get("question") or q.get("questionText")),
            "options": q.get("options") or [],
            "correctAnswer": q.get("correctAnswer", q.get("correctIndex", q.get("correctAnswers"))),
            "rationale": _text(q.get("rationale") or q.get("explanation")),
            "latestAttempt": answer, "firstAttempt": original,
        })
    if not records:
        return None
    # Preserve the whole-quiz counts; retain the most useful evidence in a
    # bounded payload, with corrected retries ahead of unattempted questions.
    answered = [q for q in records if q["latestAttempt"]]
    review = lambda q: any(a and a.get("isCorrect") is False for a in (q["latestAttempt"], q["firstAttempt"]))
    chosen = sorted(records, key=lambda q: (not review(q), not bool(q["latestAttempt"])))[:30]
    return {
        "messageId": message.get("id"), "timestamp": _text(message.get("timestamp"), 80),
        "title": _text(message.get("quizTopic") or content.get("title") or "Practice quiz", 240),
        "total": len(records), "answered": len(answered),
        "incorrect": sum(q["latestAttempt"].get("isCorrect") is False for q in answered),
        "questions": chosen,
    }


def build_chat_evidence(messages, profile=None):
    visible = [m for m in messages if not m.get("hidden")]
    users = [m for m in visible if m.get("role") == "user" and isinstance(m.get("content"), str)]
    signals = student_emphasis.gather([m["content"] for m in users], profile)
    priorities = []
    for m in users:
        # Scan every line before clipping: instructor flags can sit at the
        # end of a long paste or far back in a long chat.
        for line in re.split(r"\n|(?<=[.!?])\s+", m["content"]):
            for hit in _PRIORITY.finditer(line):
                start = max(0, hit.start() - 100)
                quote = line[start:start + 500].strip()
                if quote and not any(p["quote"] == quote for p in priorities):
                    priorities.append({"messageId": m.get("id"), "quote": quote})
                break
    conversation = [{"role": m["role"], "content": _text(m["content"], 2400), "messageId": m.get("id")}
                    for m in visible if m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)
                    and m.get("type") not in ("studysheet", "study_sheet", "quiz", "study_quiz", "study_exam")]
    quizzes = [quiz_evidence(m) for m in visible if m.get("type") in ("quiz", "study_quiz", "study_exam")]
    previous = [{"messageId": m.get("id"), "title": m.get("topic"), "content": _text(m.get("content"), 12000)}
                for m in visible if m.get("type") in ("studysheet", "study_sheet")]
    return {"conversation": conversation[-20:], "studentSignals": signals,
            "priorities": priorities[-24:], "quizzes": [q for q in quizzes if q][-8:],
            "previousSheets": previous[-2:]}
