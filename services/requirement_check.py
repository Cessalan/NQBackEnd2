"""
Checks each generated question against the student's own requirements before
she sees it.

WHY THIS EXISTS (2026-10-11)

Putting her words in the writer's prompt is necessary and not sufficient: a
writer told "ask about the medication class, never a specific drug name" will
still write a digoxin question some of the time. The student had told the app
exactly that her professor tests classes, not names; a question about one
named drug tells her the app did not listen.

So every question is checked against the requirements that apply to it (her
rules and style on every question; a focus only on the questions planned for
it). A question that fails is rewritten once with the checker's reason; a
rewrite that still fails is delivered WITHOUT the claim that it meets that
requirement, so the per-question tag and the loading-screen tick stay true.
The checker never decides the answer key and never edits a question; it only
judges, which keeps it cheap and keeps it out of the correctness path.

Claude Haiku 5.5 at low effort with structured output: one short call per
question, run in parallel with the other questions of the batch. Any failure
of the checker itself (network, refusal, junk) FAILS OPEN to "not verified",
never to "verified", and never blocks the question.
"""
import asyncio
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-5-5"
MAX_WHY = 200
_TAG = re.compile(r"<[^>]+>")

_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def applicable(checks, serves=()):
    """The checks one question is held to: every 'all' check, and a focus it serves."""
    serves = set(serves or ())
    return [c for c in checks or [] if c.get("applies") == "all" or c.get("id") in serves]


def _plain(value, limit=900):
    return " ".join(_TAG.sub(" ", str(value or "")).split())[:limit]


def render(question):
    """One question as the checker reads it: stem, scenario, options and key."""
    q = question or {}
    lines = [f"Format: {q.get('questionType') or 'mcq'}", f"Question: {_plain(q.get('question'))}"]
    case = q.get("caseStudy")
    if isinstance(case, dict):
        for tab, text in case.items():
            lines.append(f"{tab}: {_plain(text, 600)}")
    options = q.get("options") or []
    for option in options:
        lines.append(f"- {_plain(option.get('text') if isinstance(option, dict) else option, 240)}")
    answer = q.get("correctOrder") or q.get("answer")
    if isinstance(answer, list):
        answer = "; ".join(_plain(a.get("text") if isinstance(a, dict) else a, 160) for a in answer)
    if answer:
        lines.append(f"Keyed answer: {_plain(answer, 600)}")
    for row in q.get("rows") or []:
        if isinstance(row, dict):
            lines.append(f"Row: {_plain(row.get('text'), 200)}")
    return "\n".join(lines)


def output_schema():
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "meets": {"type": "boolean"},
                        "why": {"type": "string"},
                    },
                    "required": ["id", "meets", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }


PROMPT = """You check one nursing practice question against requirements a student set for her own practice. You do not judge whether the question is good in general, and you never change it.

For EACH requirement below, decide whether this question meets it. Be strict and literal about what the student asked: a requirement to avoid specific drug names is broken by any brand or generic drug name in the stem, options or keyed answer; a requirement for clinical-judgment scenarios is broken by a question that only asks to recall a fact. If a requirement does not apply to this kind of question at all, it is met.

In "why", give one short sentence a question writer can act on (what exactly breaks it), or "meets" when it is met.

REQUIREMENTS:
{requirements}

QUESTION:
{question}"""


def build_prompt(question, checks):
    requirements = "\n".join(f"[{c['id']}] {c['test']}" for c in checks)
    return PROMPT.format(requirements=requirements, question=render(question))


def validate(raw, checks):
    """{id: {meets, why}} for every check, or None when the output is unusable.

    A check the model skipped is NOT counted as met.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        return None
    wanted = {c["id"] for c in checks}
    out = {}
    for row in raw["results"]:
        if isinstance(row, dict) and row.get("id") in wanted and isinstance(row.get("meets"), bool):
            out[row["id"]] = {"meets": row["meets"], "why": " ".join(str(row.get("why") or "").split())[:MAX_WHY]}
    for check in checks:
        out.setdefault(check["id"], {"meets": None, "why": ""})
    return out


def failures(results):
    return [{"id": cid, "why": r["why"]} for cid, r in (results or {}).items() if r.get("meets") is False]


def repair_note(failed):
    """Appended to the writer's content for the one rewrite."""
    reasons = "\n".join(f"- {f['why'] or f['id']}" for f in failed)
    return ("A PREVIOUS DRAFT OF THIS QUESTION WAS REJECTED because it broke the student's requirements:\n"
            f"{reasons}\nWrite a NEW question on the same concept that meets every requirement above.")


def verified_map(results):
    """{id: True|False|None} as stored on the question: True only when checked and met."""
    return {cid: r.get("meets") for cid, r in (results or {}).items()}


# ── Hard, as something that can be checked ───────────────────────────────
#
# On 2026-10-11 ten "very hard" questions written for a Pro student were
# scenario questions that a test-wise student could answer without the
# content: the keyed option was the longest in 9 of 10, most had an option
# that was unsafe on its face, and the deciding findings were extreme (HR 38,
# SpO2 82%). Across all students since September, "hard" multiple choice was
# answered right first time 73% of the time against 78% for "medium": the
# word changed almost nothing. So when she asks for hard, "hard" is three
# checks, and a question that fails them is rewritten:
#   1. length: the key is not the stand-out longest option (code, exact);
#   2. options only: a model shown the options WITHOUT the question cannot
#      pick the key; if it can, a test-wise student can too;
#   3. the standard (HARD_STANDARD), judged by the checker with her other
#      requirements.
HARD_ID = "difficulty"
HARD_STANDARD = (
    "The question is genuinely hard for a well-prepared nursing student: answering it requires a "
    "decision (prioritize, analyze or evaluate), not recall of one fact; the stem gives at least two "
    "findings that must be combined, and no single finding gives the answer away; the deciding findings "
    "need interpretation (borderline or mixed values, a change from baseline) rather than extreme values "
    "that make the priority obvious; every wrong option is a specific nursing action that would be right "
    "for another patient or at another moment, and none is unsafe or absurd on its face; and nothing in "
    "the wording points to the key (it is not the only option that assesses or escalates, and it does "
    "not repeat the stem's words).")
LENGTH_MARGIN = 1.15
_LETTER_INDEX = re.compile(r"^\s*([A-H])(?:[).:\s]|$)", re.I)
_LETTER_PREFIX = re.compile(r"^[A-H][).:]\s*")


def _option_text(option):
    return _plain(option.get("text") if isinstance(option, dict) else option, 600)


def key_indices(question):
    """The keyed option(s) as indices, whatever shape the writer returned."""
    q = question or {}
    options = [_option_text(o) for o in q.get("options") or []]
    strip = lambda s: re.sub(r"^[A-H][).:]\s*", "", _plain(s, 600)).strip().lower()
    stripped = [strip(o) for o in options]
    if q.get("questionType") == "sata":
        answer = q.get("answer") if isinstance(q.get("answer"), list) else q.get("correctAnswers") or []
        return [stripped.index(strip(a)) for a in answer if strip(a) in stripped]
    index = (q.get("metadata") or {}).get("correctAnswerIndex")
    if isinstance(index, int) and 0 <= index < len(options):
        return [index]
    if isinstance(q.get("correctIndex"), int) and 0 <= q["correctIndex"] < len(options):
        return [q["correctIndex"]]
    answer = q.get("answer")
    if isinstance(answer, str):
        if strip(answer) in stripped:
            return [stripped.index(strip(answer))]
        letter = _LETTER_INDEX.match(answer)
        if letter and ord(letter.group(1).upper()) - 65 < len(options):
            return [ord(letter.group(1).upper()) - 65]
    return []


def length_cue(question):
    """A reason when option length gives the key away, else None.

    Single answer: the key is the longest option by LENGTH_MARGIN over the
    next longest. Select all: the keyed options are on average LENGTH_MARGIN
    longer than the rest. Ordering and matrix questions have no such cue.
    """
    q = question or {}
    if q.get("questionType") not in (None, "mcq", "sata"):
        return None
    options = [_option_text(o) for o in q.get("options") or []]
    keys = key_indices(q)
    if len(options) < 3 or not keys:
        return None
    lengths = [len(o) for o in options]
    if q.get("questionType") == "sata":
        right = [lengths[i] for i in keys]
        wrong = [l for i, l in enumerate(lengths) if i not in keys]
        if right and wrong and (sum(right) / len(right)) > LENGTH_MARGIN * (sum(wrong) / len(wrong)):
            return ("The correct options are noticeably longer and more detailed than the wrong ones; "
                    "write every option at the same length and level of detail.")
        return None
    key = keys[0]
    others = [l for i, l in enumerate(lengths) if i != key]
    if others and lengths[key] > LENGTH_MARGIN * max(others):
        return ("The correct option is clearly the longest and most detailed; rewrite so every option has "
                "the same length and specificity and the key does not stand out.")
    return None


def guess_schema():
    return {"type": "object", "properties": {"letter": {"type": "string", "enum": list("ABCDEFGH")}},
            "required": ["letter"], "additionalProperties": False}


GUESS_PROMPT = """These are the answer options of a nursing exam question. The question itself is hidden from you. Using only the options, pick the one most likely to be the keyed correct answer, the way a test-wise student would (the most complete, the safest-sounding, the one unlike the others). Answer with its letter.

{options}"""


async def option_only_guess(question):
    """The option a test-wise reader picks with the question hidden, or None."""
    options = [_option_text(o) for o in (question or {}).get("options") or []]
    if len(options) < 3 or (question or {}).get("questionType") not in (None, "mcq"):
        return None
    bare = [_LETTER_PREFIX.sub("", o) for o in options]
    listed = "\n".join(f"{chr(65 + i)}) {text}" for i, text in enumerate(bare))
    try:
        response = await _get_client().messages.create(
            model=MODEL, max_tokens=50,
            messages=[{"role": "user", "content": GUESS_PROMPT.format(options=listed)}],
            extra_body={"output_config": {"effort": "low",
                                          "format": {"type": "json_schema", "schema": guess_schema()}}},
        )
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        letter = json.loads(text).get("letter")
        return ord(letter) - 65 if letter and ord(letter) - 65 < len(options) else None
    except Exception as e:  # noqa: BLE001 - never blocks a question
        print(f"⚠️ requirement_check: option-only guess failed: {type(e).__name__}: {e}")
        return None


# The hard standard as specific findings, decided in code. A single holistic
# "is this hard?" verdict was unreliable when calibrated on two hand-written
# hard questions (2026-10-11): it failed one of them three times out of three,
# for reasons that contradicted the standard (it objected that other options
# also escalated, which is what the standard asks for). Asking for the facts
# separately and applying fixed thresholds is steadier and explains itself.
HARD_PROMPT = """You review ONE nursing exam question for how hard it is for a well-prepared nursing student. Report each item about THIS question exactly; do not judge whether it is a good question overall.

- decision: true if answering requires a clinical decision (what to do first, which finding matters, whether care is working); false if it asks to recall a fact or definition.
- findings_to_combine: how many separate findings in the stem must be weighed TOGETHER to reach the keyed answer (0 if the answer needs none).
- extreme_finding: true if one finding is so extreme that it alone makes the answer obvious (for example heart rate 38 with blood pressure 70/40, oxygen saturation 80%, stridor).
- giveaway_options: letters of options a prepared student rejects instantly WITHOUT reading the stem, because they are unsafe, lazy (only wait, document or reassure) or nonsensical. An option that is a reasonable action for some client is not a giveaway.
- lone_active_key: true if the keyed option is the only one that assesses, intervenes or escalates while every other option is passive.
- key_names_conclusion: true if a keyed option states the diagnosis or conclusion the stem leaves the student to infer.
- note: one sentence a question writer can act on to make it harder.

QUESTION:
{question}"""


def hard_schema():
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "boolean"},
            "findings_to_combine": {"type": "integer"},
            "extreme_finding": {"type": "boolean"},
            "giveaway_options": {"type": "array", "items": {"type": "string"}},
            "lone_active_key": {"type": "boolean"},
            "key_names_conclusion": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["decision", "findings_to_combine", "extreme_finding", "giveaway_options",
                     "lone_active_key", "key_names_conclusion", "note"],
        "additionalProperties": False,
    }


async def judge_hard(question):
    """The specific findings of HARD_PROMPT for one question, or None when the call failed."""
    try:
        response = await _get_client().messages.create(
            model=MODEL, max_tokens=800,
            messages=[{"role": "user", "content": HARD_PROMPT.format(question=render(question))}],
            extra_body={"output_config": {"effort": "low",
                                          "format": {"type": "json_schema", "schema": hard_schema()}}},
        )
        if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
            return None
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception as e:  # noqa: BLE001 - never blocks a question
        print(f"⚠️ requirement_check: hard judge failed: {type(e).__name__}: {e}")
        return None


def hard_verdict(sub, length_reason=None, guessed=False):
    """(meets, why) for the hard standard. Pure; tested.

    Fails on any objective cue (length, options alone) whatever the judge
    said. Otherwise it needs the judge's findings: a decision, at least two
    findings to combine, nothing extreme, no give-away option, a key that is
    not the lone active option and does not name the conclusion. Without the
    judge and without an objective failure the verdict is unknown (None),
    never a pass.
    """
    reasons = []
    if length_reason:
        reasons.append(length_reason)
    if guessed:
        reasons.append("With the question hidden, the correct option can still be picked from the options alone; "
                       "make every option equally specific and plausible on its own so only the stem decides.")
    if isinstance(sub, dict):
        if sub.get("decision") is not True:
            reasons.append("It tests recall, not a decision; ask what to do first, which finding matters, or whether care worked.")
        if not isinstance(sub.get("findings_to_combine"), int) or sub["findings_to_combine"] < 2:
            reasons.append("No two findings have to be combined; give at least two findings that only together point to the answer.")
        if sub.get("extreme_finding") is True:
            reasons.append("One finding is extreme enough to give the answer away; use a borderline value or a change from baseline.")
        giveaways = [str(g).strip()[:2] for g in sub.get("giveaway_options") or [] if str(g).strip()]
        if giveaways:
            reasons.append(f"Option(s) {', '.join(giveaways)} can be rejected without reading the stem; replace each with "
                           "an action that would be right in a nearby situation.")
        if sub.get("lone_active_key") is True:
            reasons.append("The key is the only option that assesses or escalates; make at least one wrong option active too.")
        if sub.get("key_names_conclusion") is True:
            reasons.append("The key names the conclusion the student should infer; describe the action without naming it.")
    if reasons:
        note = str((sub or {}).get("note") or "").strip() if isinstance(sub, dict) else ""
        return False, " ".join(reasons + ([note] if note else []))[:700]
    if not isinstance(sub, dict):
        return None, ""
    return True, "meets"


async def judge(question, checks):
    """Her requirements, plus the hard standard when she asked for hard. None when the checker failed."""
    hard = any(c.get("id") == HARD_ID for c in checks)
    other = [c for c in checks if c.get("id") != HARD_ID]
    if not hard:
        return await check_question(question, other)

    async def nothing():
        return {}

    results, sub, guess = await asyncio.gather(
        check_question(question, other) if other else nothing(), judge_hard(question), option_only_guess(question))
    keys = key_indices(question)
    meets, why = hard_verdict(sub, length_cue(question), guess is not None and bool(keys) and guess == keys[0])
    results = dict(results) if results is not None else {c["id"]: {"meets": None, "why": ""} for c in other}
    results[HARD_ID] = {"meets": meets, "why": why}
    return results


async def check_question(question, checks):
    """Judge one question. Returns validate()'s map, or None when the checker failed."""
    if not checks:
        return {}
    try:
        response = await _get_client().messages.create(
            model=MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": build_prompt(question, checks)}],
            # Same route as services/quiz_walkthrough.py: SDK 0.68 predates
            # the output_config parameter.
            extra_body={"output_config": {"effort": "low",
                                          "format": {"type": "json_schema", "schema": output_schema()}}},
        )
    except Exception as e:  # noqa: BLE001 - the checker never blocks a question
        print(f"⚠️ requirement_check: call failed: {type(e).__name__}: {e}")
        return None
    if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
        return None
    text = "".join(getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text")
    try:
        return validate(json.loads(text), checks)
    except (ValueError, TypeError):
        return None
