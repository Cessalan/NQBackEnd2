"""
Order walkthrough: "Who first?", the teaching layer behind a missed ordering
(case-study) question.

WHY THIS EXISTS
───────────────
Prioritization is the gap Pro students drill without closing: one student ran
11 "Targeted practice: prioritization" drills at 33-67% with no trend
(2026-09-26 read). Those drills are ordering questions ("place the nursing
actions in order of priority"), and the card only ever showed the right order
next to hers. Seeing the answer never taught the ladder that produces it.

Real misses (2026-10-10 read) showed two things the design rests on:
  * Students usually get number 1 right and break on number 2 (an EKG placed
    before oxygen, an interpreter before a quiet room). So the walkthrough
    demonstrates 1, does 2 together, and leaves the rest to her.
  * Not every ordering question is airway-breathing-circulation. Communication
    and procedure questions climb a different ladder ("can he hear you, can he
    focus, does he understand the words"). So the model PICKS a ladder from a
    closed set instead of forcing ABC onto everything.

WHAT THE MODEL WRITES, AND WHAT IT DOES NOT
───────────────────────────────────────────
The model is handed the stored correct order and never decides it. It writes
the explanation layer: which ladder, which rung each action sits on, the named
rule that decides each step (closed set, so a student learns the same few
rules everywhere), a short label per action, why each comes where it does,
and a nudge for a student who places it too early.

The walkthrough is DISCARDED, never softened, when:
  * the model says the key is not defensible from the chart (`key_supported`).
    A real 0% miss had a key that put "acknowledge the patient's emotions"
    last; a tutor confidently teaching a shaky key does more harm than none.
  * the rungs contradict the key (an action placed later sits on a higher
    rung). That is the model and the key disagreeing.
  * an action is missing or duplicated, or no breakdown quote survives.
Breakdown quotes must be literal substrings of the chart notes or the question
she saw, or they are dropped rather than highlighted.

The ladder and rule ids are a cross-repo contract with the frontend's
`ChatInerface/orderWalkthroughModel.js` (LADDER_RUNGS, RULES): the frontend
owns the words, this side owns the ids and rung counts.

Claude Haiku 5.5, effort low, structured output. Cached per question.
"""
import hashlib
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-5-5"

MIN_ITEMS, MAX_ITEMS = 3, 8
MAX_LABEL_CHARS = 40       # what an action does, at a glance
MAX_WHY_CHARS = 140        # why it comes where it does
MAX_NUDGE_CHARS = 140
MAX_THINK_CHARS = 110      # one line of the demonstrated thinking
MAX_MEANING_CHARS = 90
MAX_QUOTE_CHARS = 110
MIN_THINK, MAX_THINK = 2, 3
ATTEMPTS = 2               # a rejected walkthrough gets one fresh try

BREAKDOWN_ROLES = ("problem", "risk", "task")

# Ladder id → rung ids, most urgent first. MIRRORED in the frontend
# (orderWalkthroughModel.js LADDER_RUNGS); the frontend renders the words.
LADDERS = {
    "abc": ["airway", "breathing", "circulation", "safety", "comfort", "teaching"],
    "process": ["assess", "act", "report", "record", "teach"],
    "barriers": ["hear", "focus", "understand", "teach", "check"],
    "procedure": ["prepare", "protect", "perform", "confirm", "record"],
    "maslow": ["body", "safety", "belonging", "esteem", "growth"],
}

# The named rule that decides a step. MIRRORED in the frontend (RULES).
RULES = (
    "abc", "maslow", "assess_first", "unstable_first", "acute_first", "safety_first",
    "barrier_first", "prepare_first", "treat_before_record", "teach_last", "sequence",
)

PROMPT_VERSION = 2

_CACHE: dict[str, dict] = {}
_client: Optional[AsyncAnthropic] = None

_LANGUAGE_NAMES = {"en": "English", "fr": "French"}
_FILLER = re.compile(r"^\s*(so\s+)?(yes|no|true|false|correct|right)\b[\s,.:!]*$", re.I)


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cache_key(question: str, chart: str, items: list, language: str) -> str:
    low = lambda s: _norm(s).lower()
    payload = "|".join([f"v{PROMPT_VERSION}", language, low(question), low(chart),
                        "::".join(low(i) for i in items)]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "key_supported": {"type": "boolean"},
            "ladder": {"type": "string", "enum": list(LADDERS)},
            "breakdown": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": list(BREAKDOWN_ROLES)},
                        "quote": {"type": "string"},
                        "meaning": {"type": "string"},
                    },
                    "required": ["role", "quote", "meaning"],
                    "additionalProperties": False,
                },
            },
            "first_thinking": {"type": "array", "items": {"type": "string"}},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "integer"},
                        "rung": {"type": "integer"},
                        "rule": {"type": "string", "enum": list(RULES)},
                        "label": {"type": "string"},
                        "why_here": {"type": "string"},
                        "nudge": {"type": "string"},
                    },
                    "required": ["position", "rung", "rule", "label", "why_here", "nudge"],
                    "additionalProperties": False,
                },
            },
            "method_line": {"type": "string"},
        },
        "required": ["key_supported", "ladder", "breakdown", "first_thinking", "items", "method_line"],
        "additionalProperties": False,
    }


PROMPT = """You are a nursing tutor. A student just got this ordering question wrong. You will teach her how to find the order, step by step. The correct order below is the answer key: do not change it, explain it.

Chart notes:
{chart}
{vitals_block}
Question: {question}

Correct order (1 comes first):
{items_block}

Ladders. Pick the ONE that explains this order best. Rungs are listed most urgent first; rung numbers are 0-based.
- abc: 0 airway, 1 breathing, 2 circulation, 3 safety, 4 comfort, 5 teaching. For clinical "who/what first" questions about physical threats.
- process: 0 assess, 1 act (treat or intervene), 2 report, 3 record (document or tests that only record), 4 teach. For a single patient where you gather data, act, call, then document.
- barriers: 0 hear (sensory), 1 focus (environment, distractions), 2 understand (language, literacy), 3 teach, 4 check (teach-back, confirm). For communication and patient-teaching questions.
- procedure: 0 prepare (explain, gather, position), 1 protect (safety, infection control, interference), 2 perform, 3 confirm (verify, check), 4 record. For carrying out a procedure.
- maslow: 0 body (physiological), 1 safety, 2 belonging, 3 esteem, 4 growth. For psychosocial priority questions.

Rules (name the one that decides each step):
abc (airway, breathing, circulation first), maslow (body needs before feelings), assess_first (assess before you act), unstable_first (unstable before stable), acute_first (new and sudden before chronic and expected), safety_first (safety first), barrier_first (remove what blocks the message first), prepare_first (prepare before you perform), treat_before_record (treat and report before tests that only record), teach_last (teach once the patient is safe and able), sequence (this step needs the one before it).

Write in {language_name}, in plain words a tired student reads at a glance. Return JSON with:
- key_supported: this order is the answer key and is normally right. Priority questions always have room for debate; a step that is debatable but reasonable is supported. Set false ONLY if following this order would be unsafe for this patient (for example acting before a needed assessment, or delaying care for an unstable finding), or the chart directly contradicts a step. When true, explain the order with reasons the chart actually gives, and pick the ladder and rules that honestly fit it.
- ladder: the ladder id.
- breakdown: 1 to 3 parts, at most one per role:
    - role: "problem" (what is wrong now), "risk" (what could go wrong), or "task" (what the question asks).
    - quote: words COPIED EXACTLY from the chart notes or the question, at most 12 words. Not from the vital signs.
    - meaning: what it tells the nurse about THIS patient, at most 12 words, plain words.
  Always include "problem". Include "risk" only when it decides the order. Include "task" only when the question narrows the goal (e.g. "to overcome communication barriers"), never for a plain "place in order of priority".
- first_thinking: {min_think} to {max_think} short lines (at most 16 words each) that think out loud about why number 1 comes first, using THIS patient's details. The last line names action 1 and ONLY action 1: the student places number 2 herself next, so never mention it. Example: ["Before you do anything for him, do you know he is stable?", "Not yet. He is dizzy and his rate jumped to 106.", "So assess first."]
- items: one entry per action, position 1-based as in the correct order, with:
    - rung: the 0-based rung on your ladder. Rungs must never go back UP the ladder as positions go down: a later action sits on the same rung or a lower one (higher number).
    - rule: the rule that puts this action at this position rather than later. It must honestly describe the action: teach_last only for teaching, treat_before_record only for a test or note that records.
    - label: what this action does for THIS patient, at most 5 words, often a question (e.g. "Is he stable?", "Eases the heart's work", "Can he hear you?").
    - why_here: one sentence, at most 22 words, that names the action and says why it comes before the ones after it (e.g. "Oxygen comes before the call: it eases the heart's work right now."). For the last action, why it can wait.
    - nudge: shown when a student picks THIS action while an earlier one is still waiting. One or two short sentences, at most 22 words, that point her back to what still has to come first, without naming it and without arguing for this action. Examples: the EKG picked too early, "The EKG records the rhythm. Does that help him right now, or only the chart?"; an interpreter picked before the room is quiet, "An interpreter helps with words. Can he hear the interpreter over the announcements?". For position 1 write "" (it can never be early).
- method_line: the method in at most 9 words (e.g. "Climb the ladder: most urgent need first.").

Never overstate a finding: a value in the normal range is not a problem (SpO2 96% is normal, do not call it a drop), and never add a finding the chart does not state.

No markdown, no emojis, no em dashes."""


def build_prompt(question: str, chart: str, vitals: str, items: list, language: str) -> str:
    vitals = _norm(vitals)
    return PROMPT.format(
        chart=_norm(chart) or "(none)",
        vitals_block=f"\nVital signs and labs:\n{vitals}\n" if vitals else "",
        question=_norm(question),
        items_block="\n".join(f"{n}. {_norm(t)}" for n, t in enumerate(items, start=1)),
        language_name=_LANGUAGE_NAMES.get(language, "English"),
        min_think=MIN_THINK, max_think=MAX_THINK,
    )


def _clip(text, limit) -> str:
    text = _norm(text).replace("—", ",").replace("–", "-")
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:") + "…"


def validate_order_walkthrough(raw, question: str, chart: str, items: list) -> Optional[dict]:
    """The model's output, made safe to show, or None when it must not be shown."""
    if not isinstance(raw, dict) or raw.get("key_supported") is not True:
        return None
    ladder = raw.get("ladder")
    if ladder not in LADDERS:
        return None
    rung_count = len(LADDERS[ladder])

    entries = raw.get("items")
    if not isinstance(entries, list) or len(entries) != len(items):
        return None
    by_pos = {}
    for e in entries:
        if not isinstance(e, dict):
            return None
        pos, rung = e.get("position"), e.get("rung")
        if not isinstance(pos, int) or isinstance(pos, bool) or not 1 <= pos <= len(items) or pos in by_pos:
            return None
        if not isinstance(rung, int) or isinstance(rung, bool) or not 0 <= rung < rung_count:
            return None
        label, why, nudge = (_clip(e.get("label"), MAX_LABEL_CHARS), _clip(e.get("why_here"), MAX_WHY_CHARS),
                             _clip(e.get("nudge"), MAX_NUDGE_CHARS))
        if not label or not why or (not nudge and pos != 1):
            return None
        by_pos[pos] = {"position": pos, "rung": rung,
                       "rule": e.get("rule") if e.get("rule") in RULES else "sequence",
                       "label": label, "why_here": why, "nudge": nudge}
    ordered = [by_pos[p] for p in range(1, len(items) + 1)]
    # The key is final. A later action on a MORE urgent rung means the model's
    # ladder and the key disagree, and the walkthrough would teach a
    # contradiction ("teaching sits below circulation" next to an order that
    # puts it above).
    if any(b["rung"] < a["rung"] for a, b in zip(ordered, ordered[1:])):
        return None

    source = (_norm(chart) + " \n " + _norm(question)).lower()
    raw_parts = raw.get("breakdown") if isinstance(raw.get("breakdown"), list) else []
    kept, seen = [], set()
    for part in raw_parts:
        if not isinstance(part, dict) or part.get("role") not in BREAKDOWN_ROLES or part["role"] in seen:
            continue
        quote = _norm(part.get("quote")).strip('"“”')
        meaning = _clip(part.get("meaning"), MAX_MEANING_CHARS)
        if not quote or len(quote) > MAX_QUOTE_CHARS or quote.lower() not in source or not meaning:
            continue
        seen.add(part["role"])
        kept.append({"role": part["role"], "quote": quote, "meaning": meaning})
    if not kept:
        return None
    kept.sort(key=lambda p: BREAKDOWN_ROLES.index(p["role"]))

    thinking = [_clip(line, MAX_THINK_CHARS) for line in (raw.get("first_thinking") or [])
                if _norm(line) and not _FILLER.match(_norm(line))]
    if len(thinking) < MIN_THINK:
        return None

    return {
        "ladder": ladder,
        "breakdown": kept,
        "first_thinking": thinking[:MAX_THINK],
        "items": ordered,
        "method_line": _clip(raw.get("method_line"), 80) or None,
    }


async def generate_order_walkthrough(question: str, chart: str, items: list, vitals: str = "",
                                     language: str = "en") -> Optional[dict]:
    """A validated walkthrough, or None (the card then keeps today's
    explanation). `items` are the action texts in the CORRECT order.
    Never raises for a model-side problem."""
    question, chart = _norm(question), _norm(chart)
    items = [_norm(i) for i in (items or [])]
    if not question or not MIN_ITEMS <= len(items) <= MAX_ITEMS or any(not i for i in items):
        return None
    if len({i.lower() for i in items}) != len(items):
        return None
    language = (language or "en").split("-")[0].lower()
    language = language if language in _LANGUAGE_NAMES else "en"

    key = cache_key(question, chart, items, language)
    if key in _CACHE:
        return _CACHE[key]

    # A decline is mostly noise, not a verdict on the key: on chat
    # XbPrJdZgXEcRmlGNhgfY's 15 case questions, run twice each, 13 of 30 came
    # back null, and most of those questions passed on the other run. So a
    # rejection gets one fresh attempt, and only a usable walkthrough is
    # cached: remembering a rejection had made one unlucky roll permanent
    # for the server's lifetime, and students saw "no walkthrough" every
    # time (owner, 2026-10-11). Validation is unchanged, so a ladder that
    # contradicts the key is still never shown.
    for attempt in range(ATTEMPTS):
        try:
            response = await _get_client().messages.create(
                model=MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": build_prompt(question, chart, vitals, items, language)}],
                # SDK 0.68 predates output_config; same route as quiz_walkthrough.py.
                extra_body={"output_config": {
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": output_schema()},
                }},
            )
        except Exception as e:
            print(f"⚠️ Order walkthrough LLM call failed: {e}")
            return None

        if response.stop_reason in ("refusal", "max_tokens"):
            print(f"⚠️ Order walkthrough not usable: stop_reason={response.stop_reason}")
            continue
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            print("⚠️ Order walkthrough returned non-JSON text")
            continue

        result = validate_order_walkthrough(raw, question, chart, items)
        if result is not None:
            _CACHE[key] = result
            return result
        supported = raw.get("key_supported") if isinstance(raw, dict) else None
        print(f"⚠️ Order walkthrough rejected (attempt {attempt + 1}/{ATTEMPTS}, "
              f"key_supported={supported}) for {question[:60]!r}")
    return None
