"""
The practice settings a chat remembers between question batches.

WHY THIS EXISTS

Every chat quiz used to be decided from the one message that asked for it. On
2026-09-28 a Pro student wrote "i dont want the in order questions"; the next
batch obeyed, then "more questions" brought ordering questions straight back,
and the next day she had to say it again. Nothing had forgotten anything: there
was simply nowhere to remember it. The per-quiz `practice.settings` only lives
on one quiz message, and each new request starts from defaults.

So the chat document carries one `practiceProfile`, and every generation path
(chat quiz, practice continuation, the in-quiz tutor, plan quizzes) reads it.

THE RULES

- Only what the student EXPLICITLY changes is written. "more", "continue" and
  "NCLEX-style" change nothing.
- An excluded format stays excluded until the student names it again without a
  negation. A model's guess (the router's or the analyzer's question_types) can
  fill a gap but can never bring an excluded format back.
- A field the message doesn't mention keeps its saved value.

"casestudy" IS the ordering question here: an NGN case with drag-and-drop
ordered actions. Students call it "in order", "put things in order",
"ordering", "drag and drop". The old keyword check listed "ordering" only as a
REQUEST, so "no ordering questions" added the format it was refusing.

Pure on purpose (no LLM, no Firebase) so every rule is pinned by
tests/test_practice_profile.py, which replays the real messages.
"""
import re
from datetime import datetime, timezone

PROFILE_VERSION = 1
FORMATS = ("mcq", "sata", "casestudy")
# What a chat quiz generates when nobody has said anything about format and
# no model suggested one. Ordering (casestudy) is opt-in: it is the format
# students push back on, and a request for NGN / case studies still adds it.
DEFAULT_FORMATS = ["mcq", "sata"]
MAX_SOURCE_CHARS = 12000
MAX_SCOPE_CHARS = 600
# A message this long with no upload is study material, not a request.
PASTE_MIN_CHARS = 800

_FORMAT_PATTERNS = {
    "casestudy": re.compile(
        r"case[\s-]?stud(?:y|ies)|casestudy|\bordering\b|\bin\s+order\b|\bin\s+the\s+(?:right|correct)\s+order\b"
        r"|\b(?:put|place|arrange|rank|sort)\w*\s+(?:\w+\s+){0,3}in\s+(?:the\s+)?(?:right\s+|correct\s+)?order"
        r"|drag[\s-]*(?:and|&|n)?[\s-]*drop|\bbow[\s-]?tie\b|\bngn\b|next[\s-]gen",
        re.I),
    "sata": re.compile(r"\bsatas?\b|select[\s-]+all|multiple\s+(?:correct|answers)", re.I),
    "mcq": re.compile(r"\bmcqs?\b|multiple[\s-]*choice|single[\s-]answer", re.I),
}
_NEGATION = re.compile(
    r"\b(?:no|not|don'?t|dont|do\s+not|stop|without|never|avoid|exclude|skip|remove|less|fewer|hate|dislike|none\s+of)\b"
    r"|\bno\s+more\b|\bwant\s+no\b", re.I)
_ONLY = re.compile(r"\b(?:only|just|exclusively|nothing\s+but)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s*(?:#{1,4}\s+|\*\*|__)(.+?)(?:\*\*|__)?\s*:?\s*$")
_CLAUSE_SPLIT = re.compile(r"[.;!?\n,]+|\bbut\b|\binstead\b|\bplus\b", re.I)

# A number counts as a request only next to a verb of asking, "make it N" or
# "N more": "my exam has 50 questions" describes the exam, not the practice.
_TOTAL = re.compile(
    r"\b(?:give|make|create|generate|ask|want|need|do|write|send)\s+(?:me\s+|us\s+)?(?:a\s+|another\s+|like\s+)?(\d{1,3})\b"
    r"|\bmake\s+it\s+(\d{1,3})\b|\b(\d{1,3})\s+more\b", re.I)
# Difficulty is a change only when it describes the QUESTIONS she wants, not
# how she feels about a topic ("I find SATA difficult").
_HARD = re.compile(
    r"\b(?:harder|tougher|more\s+(?:difficult|challenging|advanced))\b"
    r"|\b(?:hard|difficult|challenging|tough|advanced)\s+(?:\w+[\s-]+){0,2}?(?:questions?|quiz|ones|practice|items?)\b"
    r"|\bmake\s+(?:it|them)\s+(?:hard|difficult|challenging)\b", re.I)
_EASY = re.compile(
    r"\b(?:easier|simpler)\b"
    r"|\b(?:easy|simple|basic)\s+(?:\w+[\s-]+){0,2}?(?:questions?|quiz|ones|practice|items?)\b"
    r"|\bmake\s+(?:it|them)\s+(?:easy|simple)\b", re.I)

# Product buttons that aim ONE quiz at a topic. Their topic is a focus for that
# quiz, never the chat's new scope: in chat Ub1B the next "50 questions based
# on the notes" inherited "Serotonin Syndrome Actions" from such a button.
FOCUS_PROMPT = re.compile(
    r"^\s*(?:create a short targeted practice on|crée une courte pratique ciblée sur)\s*:\s*(.+)$", re.I | re.S)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _clauses(text):
    return [c.strip() for c in _CLAUSE_SPLIT.split(text or "") if c and c.strip()]


def parse_changes(text):
    """What this message explicitly changes. Empty dict when it changes nothing.

    Keys (all optional): exclude, include, only (format lists),
    requested_total (int), difficulty ('easy'|'hard').
    """
    text = str(text or "")
    if not text.strip():
        return {}
    if len(text) > PASTE_MIN_CHARS * 2:
        # A long message is mostly material: notes mention "select all that
        # apply" without asking for it. Instructions sit at the start or end,
        # written as prose rather than as the notes' own bullets and headings.
        edges = (text[:400] + "\n" + text[-400:]).splitlines()
        text = "\n".join(l for l in edges if not _BULLET.match(l) and not _HEADING.match(l))
    changes = {}
    exclude, include = [], []
    only_hit = False
    for clause in _clauses(text):
        negated_spans = [m.start() for m in _NEGATION.finditer(clause)]
        for fmt, pattern in _FORMAT_PATTERNS.items():
            for match in pattern.finditer(clause):
                # Negation governs a format only when it comes BEFORE it in the
                # same clause: "no ordering", "stop giving questions where i
                # have to place things in order".
                negated = any(pos < match.start() for pos in negated_spans)
                (exclude if negated else include).append(fmt)
                break
        if _ONLY.search(clause):
            only_hit = True
    exclude = [f for f in FORMATS if f in exclude]
    include = [f for f in FORMATS if f in include and f not in exclude]
    if exclude:
        changes["exclude"] = exclude
    if include:
        if only_hit or exclude:
            # "MCQ and SATA only, no ordering" is a complete statement of
            # what she wants, not just an addition.
            changes["only"] = include
        changes["include"] = include

    for match in _TOTAL.finditer(text):
        value = next((g for g in match.groups() if g), None)
        if value and 1 <= int(value) <= 200:
            changes["requested_total"] = int(value)
            break

    for clause in _clauses(text):
        if _NEGATION.search(clause):
            continue
        if _HARD.search(clause):
            changes["difficulty"] = "hard"
        elif _EASY.search(clause):
            changes["difficulty"] = "easy"
    return changes


def _clean_formats(values):
    if not isinstance(values, (list, tuple)):
        return []
    out = []
    for value in values:
        value = str(value).lower()
        value = "casestudy" if value in ("ordering", "bowtie", "case_study", "case study") else value
        if value in FORMATS and value not in out:
            out.append(value)
    return out


def normalize(profile):
    """A profile in canonical shape. Unknown keys are dropped; missing ones None."""
    profile = profile if isinstance(profile, dict) else {}
    source = profile.get("source") if isinstance(profile.get("source"), dict) else {}
    total = profile.get("requestedTotal")
    return {
        "version": PROFILE_VERSION,
        "source": {
            "kind": source.get("kind") if source.get("kind") in ("uploads", "pasted", "general") else None,
            "files": [str(f) for f in (source.get("files") or [])][:20],
            "pastedText": str(source.get("pastedText") or "")[:MAX_SOURCE_CHARS] or None,
            "pastedMessageId": source.get("pastedMessageId"),
        },
        "scope": (str(profile.get("scope") or "").strip()[:MAX_SCOPE_CHARS]) or None,
        "emphasis": (str(profile.get("emphasis") or "").strip()[:300]) or None,
        "sourceTopics": [str(t)[:120] for t in (profile.get("sourceTopics") or []) if str(t).strip()][:40],
        "formats": _clean_formats(profile.get("formats")) or None,
        "excludedFormats": _clean_formats(profile.get("excludedFormats")),
        "difficulty": profile.get("difficulty") if profile.get("difficulty") in ("easy", "medium", "hard") else None,
        "requestedTotal": max(1, min(200, int(total))) if isinstance(total, int) and not isinstance(total, bool) else None,
        "updatedAt": profile.get("updatedAt"),
        "updatedBy": profile.get("updatedBy"),
    }


def merge(profile, changes, *, analyzer_changes=None, updated_by="chat"):
    """Apply explicit changes to a saved profile. Returns (profile, changed_fields).

    `analyzer_changes` is the intent analyzer's `practice_changes` block: it can
    set scope and emphasis (things keywords can't read) and add format
    exclusions it heard. It can never re-enable a format on its own; only the
    student's own words, parsed in `changes`, can.
    """
    before = normalize(profile)
    after = normalize(profile)
    changes = changes or {}
    analyzer_changes = analyzer_changes if isinstance(analyzer_changes, dict) else {}

    excluded = list(after["excludedFormats"])
    formats = list(after["formats"] or [])

    analyzer_excluded = _clean_formats(analyzer_changes.get("excluded_formats"))
    for fmt in list(changes.get("exclude", [])) + analyzer_excluded:
        if fmt not in excluded:
            excluded.append(fmt)
        if fmt in formats:
            formats.remove(fmt)
    for fmt in changes.get("include", []):
        if fmt in excluded:
            excluded.remove(fmt)
    if changes.get("only"):
        formats = list(changes["only"])
    elif formats:
        formats += [f for f in changes.get("include", []) if f not in formats]
    after["excludedFormats"] = excluded
    after["formats"] = formats or None

    if changes.get("requested_total"):
        after["requestedTotal"] = changes["requested_total"]
    if changes.get("difficulty"):
        after["difficulty"] = changes["difficulty"]

    scope = str(analyzer_changes.get("scope") or "").strip()
    if scope:
        after["scope"] = scope[:MAX_SCOPE_CHARS]
    if "emphasis" in analyzer_changes:
        emphasis = str(analyzer_changes.get("emphasis") or "").strip()
        if emphasis:
            after["emphasis"] = emphasis[:300]
    if analyzer_changes.get("clear_emphasis"):
        after["emphasis"] = None

    changed = [k for k in after if k not in ("updatedAt", "updatedBy") and after[k] != before[k]]
    if changed:
        after["updatedAt"] = now_iso()
        after["updatedBy"] = updated_by
    return after, changed


def effective_formats(profile, guess=None, text=None):
    """The formats one batch should be generated in.

    Saved allow-list if there is one, otherwise the model's guess (or the
    default); plus any format this message asks for; minus every exclusion.
    Never empty.
    """
    profile = normalize(profile)
    changes = parse_changes(text) if text else {}
    excluded = set(profile["excludedFormats"]) | set(changes.get("exclude", []))
    excluded -= set(changes.get("include", []))
    if changes.get("only"):
        base = list(changes["only"])
    elif profile["formats"]:
        base = list(profile["formats"])
    else:
        base = _clean_formats(guess) or list(DEFAULT_FORMATS)
    base += [f for f in changes.get("include", []) if f not in base]
    result = [f for f in base if f not in excluded]
    if not result:
        result = [f for f in ("mcq", "sata", "casestudy") if f not in excluded][:1] or ["mcq"]
    return result


def seed_from_quiz_settings(settings):
    """A first profile for a chat created before profiles existed.

    Only the settings a student could have chosen are carried over. Formats
    are NOT: an old quiz's format list was the model's guess, and freezing it
    would turn one guess into a permanent rule.
    """
    settings = settings if isinstance(settings, dict) else {}
    seeded = {}
    if settings.get("difficulty") in ("easy", "medium", "hard"):
        seeded["difficulty"] = settings["difficulty"]
    total = settings.get("requested_total")
    if isinstance(total, int) and not isinstance(total, bool) and total > 5:
        seeded["requestedTotal"] = total
    return normalize(seeded)


def focus_topic(text):
    """The topic of a 'targeted practice' button prompt, else None."""
    match = FOCUS_PROMPT.match(str(text or ""))
    return match.group(1).strip()[:200] if match else None


def looks_like_pasted_material(text):
    """A long message is study material when it reads like notes, not a request."""
    text = str(text or "")
    if len(text) < PASTE_MIN_CHARS:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) >= 4 or len(text) >= PASTE_MIN_CHARS * 2


def outline_topics(text, limit=30):
    """Headings of pasted notes, used as the topics the material covers.

    Deliberately literal: markdown headings, bold lines and short title-like
    lines. It never invents a topic; if the paste has no visible structure it
    returns fewer items and coverage falls back to the questions' own labels.
    """
    topics = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or len(line) > 90:
            continue
        match = _HEADING.match(line)
        if match:
            candidate = match.group(1)
        elif _BULLET.match(line):
            continue
        elif line.endswith(".") or line.count(" ") > 9:
            continue
        else:
            candidate = line
        candidate = re.sub(r"[*_#`]+", "", candidate).strip(" :-–—\t")
        if len(candidate) < 3 or re.search(r"\b(questions?|points?)\b.*\d|\d+\s*(?:questions?|points?)", candidate, re.I):
            continue
        if candidate.lower() not in [t.lower() for t in topics]:
            topics.append(candidate)
        if len(topics) >= limit:
            break
    return topics


def summary_line(profile):
    """One line describing the saved profile, for the analyzer and the logs."""
    p = normalize(profile)
    parts = []
    if p["scope"]:
        parts.append(f"scope: {p['scope'][:160]}")
    if p["formats"]:
        parts.append("formats: " + "+".join(p["formats"]))
    if p["excludedFormats"]:
        parts.append("never: " + "+".join(p["excludedFormats"]))
    if p["difficulty"]:
        parts.append(f"difficulty: {p['difficulty']}")
    if p["requestedTotal"]:
        parts.append(f"total: {p['requestedTotal']}")
    if p["emphasis"]:
        parts.append(f"emphasis: {p['emphasis']}")
    if p["source"]["kind"]:
        parts.append(f"source: {p['source']['kind']}")
    return "; ".join(parts) or "none saved"


def generation_guidance(profile):
    """Student preferences as text for the question generator's context."""
    p = normalize(profile)
    lines = []
    if p["emphasis"]:
        lines.append(f"- Emphasis the student asked for: {p['emphasis']}")
    if "casestudy" in p["excludedFormats"]:
        lines.append("- The student does not want ordering / drag-and-drop questions.")
    return ("STUDENT PRACTICE PREFERENCES (follow these):\n" + "\n".join(lines)) if lines else ""
