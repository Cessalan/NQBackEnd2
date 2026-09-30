"""
Read-only audit: does the same question ever carry two different answer keys?

On 2026-09-22 a Pro student got one perineal-care question nine times with
identical options, keyed B four times and A five times. The fix
(tools/mcq_answer_position.py, commit 60cc1ea) makes the model write the key
as A and moves the option in code. This checks production for the failure it
fixed, split at the fix, so "fixed" is a measurement rather than a belief.

Two checks, per question type (mcq, sata):
  * CONFLICT  - identical stem + identical option texts, different correct
                option TEXT(s), anywhere in the scanned chats.
  * INTERNAL  - `answer` names one option while the stored index points at
                another (the explanation/key contradiction in chat C348).

Writes nothing. Prints counts; `--out FILE` writes the offending items as JSON
(it contains question text, no student identifiers beyond chat ids).

    venv/Scripts/python.exe tools/answer_key_audit.py --since 2026-09-01 --fix 2026-09-23T01:43:00Z
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

_LABEL = re.compile(r"^\s*([A-Ha-h])\s*[\).:\-]\s*")


def strip_label(text):
    return _LABEL.sub("", str(text or ""), count=1).strip()


def norm(text):
    return re.sub(r"\s+", " ", strip_label(text)).strip().lower()


def correct_texts(q):
    """The option text(s) this stored question marks correct, or None if unreadable."""
    options = q.get("options") or []
    texts = [strip_label(o) for o in options]
    if isinstance(q.get("correctIndices"), list) or q.get("questionType") == "sata":
        indices = q.get("correctIndices") or []
        if not indices and isinstance(q.get("correctAnswers"), list):
            return sorted(norm(a) for a in q["correctAnswers"])
        picked = [texts[i] for i in indices if isinstance(i, int) and 0 <= i < len(texts)]
        return sorted(norm(t) for t in picked) or None
    index = q.get("correctIndex")
    if not isinstance(index, int) or index < 0:
        index = (q.get("metadata") or {}).get("correctAnswerIndex")
    if not isinstance(index, int):
        m = _LABEL.match(str(q.get("answer") or ""))
        index = ord(m.group(1).upper()) - 65 if m else None
    if isinstance(index, int) and 0 <= index < len(texts):
        return [norm(texts[index])]
    return None


def internal_mismatch(q):
    """`answer` text names a different option than the stored index."""
    options = [strip_label(o) for o in (q.get("options") or [])]
    answer_text = strip_label(q.get("answer"))
    index = q.get("correctIndex")
    if not isinstance(index, int) or index < 0:
        index = (q.get("metadata") or {}).get("correctAnswerIndex")
    if not answer_text or not isinstance(index, int) or not (0 <= index < len(options)):
        return False
    if norm(answer_text) not in {norm(o) for o in options}:
        return False  # a letter-only answer, or free text; nothing to compare
    return norm(answer_text) != norm(options[index])


def iter_questions(value, depth=0):
    """Every stored question inside a message, whatever its shape.

    Chat quizzes keep a flat `quizData` list plus `practice.questions`; plan
    quizzes nest them (`quizData[0].questions`), and exams and other cards use
    their own wrappers. Anything with a question stem and an options list is
    a question.
    """
    if depth > 6:
        return
    if isinstance(value, dict):
        if isinstance(value.get("question"), str) and isinstance(value.get("options"), list):
            yield value
            return
        for child in value.values():
            yield from iter_questions(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from iter_questions(child, depth + 1)


def to_dt(value):
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return datetime.fromtimestamp(value.timestamp(), tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-09-01", help="only chats updated on/after this date")
    parser.add_argument("--fix", default="2026-09-23T01:43:00Z", help="when the answer-position fix shipped (UTC)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import firebase_admin
    from firebase_admin import credentials, firestore
    key = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service-account-key.json")
    firebase_admin.initialize_app(credentials.Certificate(key))
    db = firestore.client()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    fix = to_dt(args.fix)
    groups = defaultdict(list)
    internal = []
    scanned = 0
    per_era = defaultdict(int)
    chats = db.collection("chats").where(filter=firestore.FieldFilter("updatedAt", ">=", since)).stream()
    for chat in chats:
        for message in chat.reference.collection("messages").stream():
            data = message.to_dict()
            if data.get("role") == "user":
                continue
            when = to_dt(data.get("timestamp"))
            era = "after_fix" if when and fix and when >= fix else "before_fix"
            seen = set()
            for q in iter_questions(data):
                if not q.get("options"):
                    continue
                stem = norm(q.get("question"))
                if not stem or stem in seen:
                    continue
                seen.add(stem)
                scanned += 1
                per_era[era] += 1
                qtype = "sata" if (q.get("questionType") == "sata" or isinstance(q.get("correctIndices"), list)) else "mcq"
                key = (qtype, stem, tuple(sorted(norm(o) for o in q["options"])))
                keyed = correct_texts(q)
                if keyed is not None:
                    groups[key].append({"era": era, "chat": chat.id, "message": message.id, "type": data.get("type"), "correct": keyed})
                if qtype == "mcq" and internal_mismatch(q):
                    internal.append({"era": era, "chat": chat.id, "message": message.id, "question": q.get("question"),
                                     "answer": q.get("answer"), "correctIndex": q.get("correctIndex")})

    conflicts = []
    for (qtype, stem, _), rows in groups.items():
        if len({tuple(r["correct"]) for r in rows}) > 1:
            conflicts.append({"type": qtype, "question": stem[:200], "instances": rows,
                              "after_fix": any(r["era"] == "after_fix" for r in rows)})

    repeated = sum(1 for rows in groups.values() if len(rows) > 1)
    print(f"questions scanned: {scanned}   distinct items repeated: {repeated}")
    for era in ("before_fix", "after_fix"):
        c = sum(1 for x in conflicts if all(r["era"] == era for r in x["instances"]))
        i = sum(1 for x in internal if x["era"] == era)
        print(f"{era:>11}: {per_era[era]} questions, conflicting keys {c}, internal key/answer mismatches {i}")
    spanning = sum(1 for x in conflicts if x["after_fix"] and any(r["era"] == "before_fix" for r in x["instances"]))
    print(f"  spanning the fix: {spanning}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"conflicts": conflicts, "internal": internal}, f, ensure_ascii=False, indent=1, default=str)
        print(f"details written to {args.out}")


if __name__ == "__main__":
    main()
