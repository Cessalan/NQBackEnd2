"""
The student's request, kept in her own words: what she asked this practice to be.

WHY THIS EXISTS (2026-10-11)

A Pro student who had paid 17 minutes earlier wrote, over four messages:
"a super tough practice exam nclex style with so much clinical reasoning on
the material i gave you", "a full exam on both these chapters with medications
questions also. the doctor expects us to know medication class not medications
name ... she gives us very hard exams", "i said i need a full exam", "i need
harder questions on medications". Each time she got 5 questions. By her fourth
quiz the only trace of all of it was one saved line, "harder medication
questions": `generationInstructions` was overwritten by her newest message, the
question writer saw only that message, and the loading screen showed none of
her words. She scored 5/5 and wrote "these are very easy".

A read of every chat on the same day found the pattern was the norm, not her
bad luck: 88 of 139 quizzes where a student named a number delivered fewer
questions, and 26 students answered every question they were given and then
nothing more came.

So a request is kept as a short list of requirements, each one in her words:

- WHAT SHE ASKED FOR is never paraphrased into something else. `said` is her
  own clause; `label` is a short restatement shown on the loading screen;
  `instruction` is what the question writer is told.
- DURABLE kinds (scope, focus, rule, style) are saved on the chat's practice
  profile and survive "more", resume and a later "harder ones". A new message
  adds to them; it never silently erases them. A new SUBJECT (scope change)
  retires the old subject's scope and focus, and keeps her rules and style.
- PER-REQUEST facts (how many questions, exam conditions, things the app
  cannot do) belong to one request and are not saved as standing wishes.
- Code reads what keywords can read reliably (a number next to a verb of
  asking, "full exam", NCLEX / clinical reasoning). The intent analyzer adds
  what only reading can (a focus like "medication classes", a rule like "ask
  about the class, not the drug name"). Either alone produces a usable brief,
  so a skipped or failed analyzer never empties the loading screen.

Pure on purpose (no LLM, no Firebase); every rule is pinned by
tests/test_request_brief.py, which replays the real messages.
"""
import re

# A request for "an exam" with no number. The most common number students ask
# for (25 of 88 short-delivered requests asked for 50), and close to a unit
# exam. Shown on the loading screen as an assumption she can change.
FULL_EXAM_TOTAL = 50
MAX_TOTAL = 200

DURABLE_KINDS = ("scope", "focus", "rule", "style")
ANALYZER_KINDS = ("scope", "focus", "rule", "style", "unsupported")
MAX_DURABLE = 8
MAX_SAID = 160
MAX_LABEL = 90
MAX_INSTRUCTION = 320
DEFAULT_FOCUS_SHARE = 0.3

# The clinical-reasoning style, as students actually write it. "ATI" and
# "HESI" are exam vendors whose questions are scenario-based like NCLEX.
_STYLE = re.compile(
    r"\bnclex\w*|\bnxlex\b|\bnext[\s-]?gen\w*|\bngn\b|\bati\b|\bhesi\b"
    r"|clinic\w*\s+(?:reasoning|judg(?:e)?ment|scenarios?|thinking|situations?)"
    r"|critical\s+thinking|situational|patient\s+scenarios?|scenario[\s-]based|case[\s-]based"
    r"|application[\s-](?:level|based)|raisonnement\s+clinique|mises?\s+en\s+situation",
    re.I)

# Asking FOR an exam, as opposed to mentioning one ("my exam is tomorrow",
# "help me prepare for my exam"). The exam word must be the object of a verb
# of asking, with no "my/for/on/about" in between.
_ASK_VERB = (r"(?:give|make|create|generate|build|write|need|want|do|prepare|set|send|design|"
             r"can\s+you\s+(?:make|give|create|generate|do|write)|could\s+you\s+(?:make|give|create|generate|do|write)|"
             r"i'?d\s+like|i\s+would\s+like|fais|fait|cr[ée]e|g[ée]n[èe]re|donne)")
_EXAM_NOUN = r"(?:exams?|examen|tests?|mock|quiz\s+exam)"
_EXAM_ASK = re.compile(
    rf"\b{_ASK_VERB}\s+(?:me\s+|us\s+|moi\s+)?(?:a|an|another|one|some|the|un|une)?\s*"
    rf"(?:(?!(?:my|for|on|about|to|mon|ma|pour)\b)[\w'-]+\s+){{0,6}}?{_EXAM_NOUN}\b"
    r"|\b(?:practice|mock|full|complete|whole|entire|comprehensive|sample)\s+(?:length\s+)?(?:practice\s+)?(?:exams?|tests?)\b"
    r"|\bexamen\s+blanc\b",
    re.I)
# "test" is also a verb: "make sure the questions test application" is not a
# request for an exam, and neither is the product's own "Generate a knowledge
# test quiz about X". A matched "test" followed by one of these is the verb.
_TEST_AS_VERB = re.compile(
    r"\s+(?:me|my|your|their|the|whether|understanding|application|knowledge|for|if|how|quiz)\b", re.I)
_FULL = re.compile(
    r"\b(?:full|complete|whole|entire|comprehensive|real)\s+(?:length\s+)?(?:practice\s+)?(?:exams?|tests?|mock)\b"
    r"|\bexamen\s+(?:complet|entier)\b",
    re.I)

# A number that is a request without a verb next to it: "can you make me a
# nclex style test using the material i gave you, minimum 20 questions".
_AT_LEAST = re.compile(r"\b(?:minimum|min|at\s+least|no\s+less\s+than)\s+(?:of\s+)?(\d{1,3})\s+(?:questions?|items?|qs)\b", re.I)

_CLAUSE = re.compile(r"[.;!?\n]+")
_SLUG = re.compile(r"[^a-z0-9]+")


def _clip(text, limit):
    return " ".join(str(text or "").split())[:limit].strip()


def slug(text):
    return _SLUG.sub("-", str(text or "").lower()).strip("-")[:40] or "item"


def item_id(kind, label):
    return f"{kind}:{slug(label)}"


def said_for(text, match):
    """The clause of her message that contains a match: her words, not ours."""
    text = str(text or "")
    if not match:
        return None
    start = max((m.end() for m in _CLAUSE.finditer(text, 0, match.start())), default=0)
    end_match = _CLAUSE.search(text, match.end())
    end = end_match.start() if end_match else len(text)
    return _clip(text[start:end], MAX_SAID) or None


def floor(text, explicit_total=None):
    """What code can read from one message without a model.

    Returns {total, full, exam, style, said: {total, exam, style}}.
    `explicit_total` is practice_profile.parse_changes' requested_total: a
    number next to a verb of asking, which is the only number code trusts.
    """
    text = str(text or "")
    result = {"total": explicit_total if isinstance(explicit_total, int) and explicit_total > 0 else None,
              "full": False, "exam": False, "style": None, "said": {}}
    if not text.strip():
        return result
    if not result["total"]:
        at_least = _AT_LEAST.search(text)
        if at_least and 1 <= int(at_least.group(1)) <= MAX_TOTAL:
            result["total"] = int(at_least.group(1))
    full = _FULL.search(text)
    exam = None
    for match in _EXAM_ASK.finditer(text):
        noun = match.group(0).rstrip().split()[-1].lower()
        if noun in ("test", "tests") and _TEST_AS_VERB.match(text, match.end()):
            continue
        exam = match
        break
    if full or exam:
        result["exam"] = True
        result["full"] = bool(full)
        result["said"]["exam"] = said_for(text, full or exam)
    style = _STYLE.search(text)
    if style:
        result["style"] = {"id": "style:clinical", "kind": "style", "label": "Clinical reasoning, NCLEX style",
                           "said": said_for(text, style),
                           "instruction": ("Every question is a realistic patient scenario that requires clinical "
                                           "judgment (assess, prioritize, act, evaluate), not recall of a fact.")}
    if result["total"]:
        number = re.search(rf"\b{result['total']}\b", text)
        result["said"]["total"] = said_for(text, number)
    return result


def normalize_item(raw):
    """One requirement in canonical shape, or None when it is not usable."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in ANALYZER_KINDS:
        return None
    label = _clip(raw.get("label"), MAX_LABEL)
    said = _clip(raw.get("said"), MAX_SAID)
    instruction = _clip(raw.get("instruction"), MAX_INSTRUCTION)
    if not (label or said):
        return None
    item = {"id": raw.get("id") if isinstance(raw.get("id"), str) and raw["id"].startswith(kind + ":")
            else item_id(kind, label or said),
            "kind": kind, "label": label or said, "said": said or None,
            "instruction": instruction or None}
    if kind == "focus":
        share = raw.get("share")
        try:
            share = float(share)
        except (TypeError, ValueError):
            share = DEFAULT_FOCUS_SHARE
        item["share"] = round(min(1.0, max(0.1, share)), 2)
    return item


def normalize_brief(items):
    """A saved brief: durable items only, deduplicated by id, newest last."""
    out = []
    for raw in items if isinstance(items, list) else []:
        item = normalize_item(raw)
        if not item or item["kind"] not in DURABLE_KINDS:
            continue
        out = [i for i in out if i["id"] != item["id"]]
        out.append(item)
    return out[-MAX_DURABLE:]


def analyzer_items(practice_changes):
    """The analyzer's requirements for THIS message, normalized."""
    changes = practice_changes if isinstance(practice_changes, dict) else {}
    return [i for i in (normalize_item(r) for r in (changes.get("requirements") or [])) if i]


def merge(saved, new_items, *, scope_action=None, drop=None, floor_style=None):
    """The durable brief after one message. Returns (brief, request_only).

    `request_only` are durable-kind items that apply to this request but are
    not saved: a one-off focus ("Create a short targeted practice on: X")
    must not become the chat's standing wish.
    """
    brief = normalize_brief(saved)
    drop = {d for d in (drop or []) if isinstance(d, str)}
    if scope_action == "change":
        # A new subject retires the old subject's scope and focus. Her rules
        # ("class, not drug name") and style ("NCLEX") are about how she wants
        # questions written, and carry over.
        brief = [i for i in brief if i["kind"] not in ("scope", "focus")]
    brief = [i for i in brief if i["id"] not in drop and i["kind"] not in drop]
    incoming = [i for i in (new_items or []) if i and i["kind"] in DURABLE_KINDS]
    if floor_style and not any(i["kind"] == "style" for i in incoming):
        incoming.append(dict(floor_style))
    request_only = []
    for item in incoming:
        if scope_action == "focus" and item["kind"] in ("scope", "focus"):
            request_only.append(item)
            continue
        if item["kind"] in ("scope", "style"):
            # One scope and one style at a time: the newest statement wins.
            brief = [i for i in brief if i["kind"] != item["kind"]]
        brief = [i for i in brief if i["id"] != item["id"]]
        brief.append(item)
    return normalize_brief(brief), request_only


def resolve_total(*, explicit=None, analyzer_total=None, floor_result=None, is_continuation=False,
                  saved_total=None, router_total=None, focus=False):
    """How many questions this request is for. Returns (total, assumed).

    Her number wins; then the analyzer's reading of her number; then an exam
    with no number is a full exam; a "more" keeps the practice's saved size;
    otherwise the router's choice. `assumed` marks a number she did not say,
    which the loading screen shows as changeable.
    """
    floor_result = floor_result or {}

    def clamp(value):
        return max(1, min(MAX_TOTAL, int(value)))

    if isinstance(explicit, int) and explicit > 0:
        return clamp(explicit), False
    if isinstance(analyzer_total, int) and not isinstance(analyzer_total, bool) and analyzer_total > 0:
        return clamp(analyzer_total), False
    if floor_result.get("exam") and not focus:
        return FULL_EXAM_TOTAL, True
    if is_continuation and isinstance(saved_total, int) and saved_total > 0:
        return clamp(saved_total), False
    if isinstance(router_total, int) and router_total > 0:
        return clamp(router_total), True
    return 5, True


def request_view(brief, request_only=(), unsupported=(), *, total, assumed, exam, said=None,
                 difficulty=None, formats=None, excluded=None, files=None, scope=None):
    """Everything this request promises, in display order, for the student and the writer.

    The shape the frontend renders (requestBriefModel.js) and the planner
    reads. Durable free-form items keep their labels; fixed facts (count,
    difficulty, formats, conditions) carry values the frontend translates.
    """
    said = said or {}
    items = [{"id": "count", "kind": "count", "value": total, "assumed": bool(assumed),
              "said": said.get("total")}]
    durable = list(brief or []) + list(request_only or [])
    scope_items = [i for i in durable if i["kind"] == "scope"]
    if scope_items:
        items.append({**scope_items[-1], "files": list(files or [])})
    elif files or scope:
        items.append({"id": "scope:material", "kind": "scope", "label": scope or None,
                      "said": None, "files": list(files or []), "derived": True})
    items += [i for i in durable if i["kind"] == "focus"]
    items += [i for i in durable if i["kind"] == "style"]
    if difficulty in ("easy", "hard"):
        items.append({"id": "difficulty", "kind": "difficulty", "value": difficulty, "said": said.get("difficulty")})
    if formats:
        items.append({"id": "formats", "kind": "format", "value": list(formats),
                      "excluded": list(excluded or []), "said": said.get("formats")})
    items += [i for i in durable if i["kind"] == "rule"]
    if exam:
        items.append({"id": "conditions", "kind": "conditions", "value": "exam", "said": said.get("exam")})
    items += [dict(i) for i in unsupported or [] if i and i.get("kind") == "unsupported"]
    return {"items": items, "total": total, "exam": bool(exam)}


# What the writer is told when she asks for hard. The same standard is then
# CHECKED on every question (services/requirement_check.HARD_STANDARD), so
# this is prevention and the check is enforcement.
HARD_WRITING = """SHE ASKED FOR HARD QUESTIONS. Every question must meet this standard or it is rejected:
- Test a decision (what to do first, which finding matters, whether care worked), never recall of one fact.
- Give at least two findings that must be combined (for example a medication class, a lab value and the patient's state); no single finding should give the answer away.
- Make the deciding findings need interpretation: borderline or mixed values, a change from baseline, a finding that matters only because of another one. Do not use extreme values that make the priority obvious.
- Every wrong option is a specific, real nursing action that would be right for a different patient or at a different moment. No option may be unsafe or absurd on its face.
- Write all options at the same length and level of detail. The correct option must not be the longest, the most complete, or the only one that assesses or notifies, and must not repeat the stem's words."""


def is_hard(view):
    return any(i.get("kind") == "difficulty" and i.get("value") == "hard" for i in (view or {}).get("items", []))


def writer_requirements(view):
    """The requirements every question writer must follow, as a text block.

    Placed before the source material with an explicit precedence statement,
    so her words are not read as just more document content.
    """
    lines = []
    for item in (view or {}).get("items", []):
        if item.get("kind") in ("focus", "rule", "style", "scope") and item.get("instruction"):
            said = f' (she said: "{item["said"]}")' if item.get("said") else ""
            lines.append(f"- {item['instruction']}{said}")
    blocks = []
    if lines:
        blocks.append("THE STUDENT'S OWN REQUIREMENTS FOR THIS PRACTICE. Follow every one; they outrank "
                      "every instruction about style below, and a question that breaks one is rejected:\n" + "\n".join(lines))
    if is_hard(view):
        blocks.append(HARD_WRITING)
    return "\n\n".join(blocks)


# A batch is made harder when she is getting at least this share right on her
# first try over at least ESCALATE_MIN answers. "Hard" is relative to her: the
# Pro student of 2026-10-11 scored 18 of 20 on questions labelled hard.
ESCALATE_SHARE = 0.8
ESCALATE_MIN = 5


def escalation(performance):
    """Writer instructions for a harder next batch, or None. Pure; tested."""
    performance = performance if isinstance(performance, dict) else {}
    answered, correct = performance.get("answered"), performance.get("correct")
    if not isinstance(answered, int) or not isinstance(correct, int) or isinstance(answered, bool) \
            or answered < ESCALATE_MIN or correct < 0 or correct > answered:
        return None
    if correct / answered < ESCALATE_SHARE:
        return None
    return (f"SHE IS FINDING THESE TOO EASY: {correct} of {answered} right on the first try. Make this question "
            "clearly harder than a standard exam item: three findings to combine instead of two, a subtler "
            "deciding cue, and distractors that are each the right action in a closely related situation.")


def checkable(view):
    """Requirements a checker can judge from one question, with what to test."""
    from services.requirement_check import HARD_ID, HARD_STANDARD
    checks = []
    if is_hard(view):
        checks.append({"id": HARD_ID, "test": HARD_STANDARD, "applies": "all"})
    for item in (view or {}).get("items", []):
        if item.get("kind") == "rule" and item.get("instruction"):
            checks.append({"id": item["id"], "test": item["instruction"], "applies": "all"})
        elif item.get("kind") == "style" and item.get("instruction"):
            checks.append({"id": item["id"], "test": item["instruction"], "applies": "all"})
        elif item.get("kind") == "focus" and item.get("instruction"):
            # A focus is a share of the exam, so only the questions planned
            # for it are held to it.
            checks.append({"id": item["id"], "test": item["instruction"], "applies": "serves"})
    return checks


def saved_summary(brief):
    """The saved brief for the analyzer's context, with ids it can drop."""
    items = normalize_brief(brief)
    if not items:
        return ""
    return "; ".join(f"[{i['id']}] {i['label']}" for i in items)
