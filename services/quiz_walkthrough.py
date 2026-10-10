"""
Quiz walkthrough: the teaching layer behind "Show me how" on a missed
select-all question.

WHY THIS EXISTS
───────────────
Nearly every paying student has the same gap: ~80-89% on multiple choice and
~22-42% on select-all (2026-10-09 read of the 16 payers since paywall
telemetry began). The app already shows which options were right; what it
never showed is HOW to get there. More drills of the same format did not move
it. So the walkthrough teaches one method on the question she just missed:

    find the one fact that matters → turn it into one test question → ask it
    of every option, alone.

WHAT THE MODEL WRITES, AND WHAT IT DOES NOT
───────────────────────────────────────────
The model never decides what is correct. It is handed the stored answer key
and writes only the explanation layer: the key fact (quoted from the stem),
the test question, a 2-3 link reasoning chain and a hint per option. It also
states its own yes/no per option, and if any of those disagrees with the key
the whole walkthrough is DISCARDED: a tutor explaining the wrong answer the
night before an exam costs more than no tutor. The key fact must be a literal
substring of the question, or it is dropped rather than highlighted.

No new questions are generated here. The question, options and key are the
ones she already answered, produced by the existing generator from her notes.

Claude Haiku 5.5, effort low, structured output. Cached per question (same
key shape as quiz_rationale.py), so a question many students miss is paid for
once. Pure validation lives in `validate_walkthrough` so it is testable
without the API.
"""
import hashlib
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-5-5"

MAX_FACT_CHARS = 120
MAX_SHORT_CHARS = 90      # meaning, test question, method line
MAX_LINK_CHARS = 64       # one link of a reasoning chain
MAX_HINT_CHARS = 110
MIN_LINKS, MAX_LINKS = 2, 3
MAX_OPTIONS = 8
MAX_QUOTE_CHARS = 110     # a breakdown quote; must be a literal substring of the stem

# The question is broken down before the options, into at most one part per
# role, shown in this order. Not every question has a risk; only real parts
# are shown. v4 (2026-10-10): one key fact could not explain why options fail
# for a DIFFERENT reason (a trauma airway question whose wrong options move
# the neck), so the stem is now pulled apart the way NCLEX teaches it.
BREAKDOWN_ROLES = ("problem", "risk", "task")

# Part of the cache key: a prompt change must not keep serving walkthroughs
# written by the previous prompt. v2 (2026-10-10): the owner found v1's key
# fact named the topic ("breathing spontaneously") instead of the deciding
# finding, and its test question was jargon ("assess the breathing patient
# without overriding it").
PROMPT_VERSION = 5

# Reasoning steps that only restate the verdict or the test question ("Answer:
# yes, true", "checking breathing answers the test question"). v2 produced
# them as a filler third step; they are dropped rather than shown.
_FILLER_LINK = re.compile(
    r"^\s*(answer\b|so\s+(yes|no)\b|(yes|no|true|false)\b[\s,.:]*((yes|no|true|false)\b)?\s*$)"
    r"|answers the test question|test question|\bfits? the test\b|\bso it (fits|does not fit)\b|\bdoes not fit\b",
    re.I,
)

_CACHE: dict[str, dict] = {}
_client: Optional[AsyncAnthropic] = None

_LANGUAGE_NAMES = {"en": "English", "fr": "French"}
_LETTER_PREFIX = re.compile(r"^[A-Z]\)\s*")


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def clean_option(text) -> str:
    return _LETTER_PREFIX.sub("", str(text or "")).strip()


def cache_key(question: str, options: list, correct: list, language: str) -> str:
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    payload = "|".join([f"v{PROMPT_VERSION}", language, ",".join(str(i) for i in sorted(correct)), norm(question),
                        "::".join(norm(clean_option(o)) for o in options)]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def output_schema(option_count: int) -> dict:
    return {
        "type": "object",
        "properties": {
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
            "test_question": {"type": "string"},
            "method_line": {"type": "string"},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "verdict": {"type": "boolean"},
                        "part": {"type": "integer"},
                        "chain": {"type": "array", "items": {"type": "string"}},
                        "hint": {"type": "string"},
                    },
                    "required": ["index", "verdict", "part", "chain", "hint"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["breakdown", "test_question", "method_line", "options"],
        "additionalProperties": False,
    }


PROMPT = """You are a nursing tutor teaching ONE method for select-all-that-apply questions to a student who just got this question wrong. The answer key below is final and correct. Do not re-decide it; explain it.

The method: break the question down first (what is wrong with the patient, what could go wrong, what the question asks the nurse to do), turn that into a single yes/no test question, then ask that test of every option on its own, top to bottom.

Question: {question}

Options:
{options_block}

Answer key (these options are TRUE, all others FALSE): {key_letters}

Write in {language_name}, in plain words a tired student reads at a glance. Return JSON with:
- breakdown: 1 to 3 parts of the question, at most one per role, each with:
    - role: "problem" (the finding that shows what is wrong now), "risk" (what could go wrong or must be protected, e.g. a possible neck injury after a high-speed collision), or "task" (what the question asks the nurse to do, e.g. "during the initial airway assessment").
    - quote: words COPIED EXACTLY from the question text above, at most 12 words: a finding, vital sign, lab value, mechanism of injury or the task phrase. Never the whole question.
    - meaning: what it tells the nurse about THIS patient, in plain words a first-year student uses, at most 12 words (e.g. "airway partly blocked by blood or fluid", "possible neck injury, so don't move the neck").
  Include a "risk" part ONLY when some options are wrong because of it. Always include the "problem". Include "task" when the question names a phase or priority.
- test_question: one plain yes/no question to ask of every option, at most 14 words, built from the breakdown so it cleanly separates the right options from the wrong ones. No jargon; name the actual problem and any risk (e.g. "Does this clear or check the airway without moving the neck?", "Does this protect a slowed heart?").
- options: one entry per option, index 0-based in the order given, with:
    - verdict: true if the option is in the answer key, false otherwise.
    - part: the 0-based position in your breakdown list of the part this option's verdict turns on (the risk for an option that is wrong because of the risk), or -1 if none.
    - chain: {min_links}-{max_links} very short reasoning steps (at most 6 words each). Step 1 says what this option does for THIS patient; the last step says why that is or is not what the test question asks for. Every step must add a new clinical reason. Never restate the verdict ("Answer: yes", "true"), never repeat the test question, and never say an option "fits" or "does not fit"; the last step must be a clinical reason. Example: ["Already slow", "slowing more is dangerous"]; ["88% means low oxygen now", "waiting for imaging delays the fix"].
    - hint: one sentence (at most 16 words) nudging a student who answered this option wrong, without giving the answer away.
- method_line: the method in at most 8 words (e.g. "One fact, one question, every option.").

No markdown, no emojis, no em dashes."""


def build_prompt(question: str, options: list, correct: list, language: str) -> str:
    letters = lambda i: chr(ord("A") + i)
    return PROMPT.format(
        question=question.strip(),
        options_block="\n".join(f"{letters(i)}) {clean_option(o)}" for i, o in enumerate(options)),
        key_letters=", ".join(letters(i) for i in sorted(correct)),
        language_name=_LANGUAGE_NAMES.get(language, "English"),
        min_links=MIN_LINKS, max_links=MAX_LINKS,
    )


def _clip(text, limit) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().replace("—", ",")
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:") + "…"


def validate_walkthrough(raw, question: str, options: list, correct: list) -> Optional[dict]:
    """The model's output, made safe to show, or None when it must not be shown.

    Rejects outright when any verdict disagrees with the answer key, when an
    option is missing or duplicated, when a chain is too short to show the
    reasoning, or when no breakdown part survives. Drops (rather than
    invents) any breakdown part whose quote is not a literal substring of the
    question, and any second part claiming a role already used.
    """
    if not isinstance(raw, dict):
        return None
    correct = set(correct)
    entries = raw.get("options")
    if not isinstance(entries, list) or len(entries) != len(options):
        return None

    by_index = {}
    for e in entries:
        if not isinstance(e, dict):
            return None
        idx = e.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(options) or idx in by_index:
            return None
        if bool(e.get("verdict")) != (idx in correct):
            return None  # the model disagrees with the key: never teach that
        chain = [_clip(link, MAX_LINK_CHARS) for link in (e.get("chain") or [])
                 if str(link or "").strip() and not _FILLER_LINK.search(str(link))]
        if len(chain) < MIN_LINKS:
            return None
        part = e.get("part")
        by_index[idx] = {"index": idx, "verdict": idx in correct, "chain": chain[:MAX_LINKS],
                         "part": part if isinstance(part, int) and not isinstance(part, bool) else -1,
                         "hint": _clip(e.get("hint"), MAX_HINT_CHARS)}

    # The breakdown: literal quotes only, one part per role, shown in role
    # order. Positions the model referred to are remapped to what survives.
    raw_parts = raw.get("breakdown") if isinstance(raw.get("breakdown"), list) else []
    kept, remap, seen = [], {}, set()
    for pos, part in enumerate(raw_parts):
        if not isinstance(part, dict) or part.get("role") not in BREAKDOWN_ROLES or part["role"] in seen:
            continue
        quote = re.sub(r"\s+", " ", str(part.get("quote") or "")).strip().strip('"\u201c\u201d')
        meaning = _clip(part.get("meaning"), MAX_SHORT_CHARS)
        if not quote or len(quote) > MAX_QUOTE_CHARS or quote.lower() not in question.lower() or not meaning:
            continue
        seen.add(part["role"])
        kept.append({"role": part["role"], "quote": quote, "meaning": meaning, "_pos": pos})
    if not kept:
        return None
    kept.sort(key=lambda p: BREAKDOWN_ROLES.index(p["role"]))
    for new_pos, p in enumerate(kept):
        remap[p.pop("_pos")] = new_pos
    for entry in by_index.values():
        entry["part"] = remap.get(entry["part"], -1)

    test_question = _clip(raw.get("test_question"), MAX_SHORT_CHARS)
    if not test_question:
        return None
    return {
        "breakdown": kept,
        # The first part, under the v1-v3 field names, for any client that
        # predates the breakdown.
        "key_fact": kept[0]["quote"],
        "fact_meaning": kept[0]["meaning"],
        "test_question": test_question,
        "method_line": _clip(raw.get("method_line"), MAX_SHORT_CHARS) or None,
        "options": [by_index[i] for i in range(len(options))],
    }


async def generate_walkthrough(question: str, options: list, correct: list, language: str = "en") -> Optional[dict]:
    """A validated walkthrough, or None (the frontend then keeps today's
    explanation). Never raises for a model-side problem."""
    question = (question or "").strip()
    if not question or not 2 <= len(options) <= MAX_OPTIONS:
        return None
    correct = sorted({i for i in correct if isinstance(i, int) and 0 <= i < len(options)})
    if not correct:
        return None
    language = (language or "en").split("-")[0].lower()
    language = language if language in _LANGUAGE_NAMES else "en"

    key = cache_key(question, options, correct, language)
    if key in _CACHE:
        return _CACHE[key]

    try:
        response = await _get_client().messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": build_prompt(question, options, correct, language)}],
            # SDK 0.68 predates the output_config parameter; same route as
            # services/exam_debrief.py. Effort low: a short, fully specified
            # writing task gains nothing from deep thinking.
            extra_body={"output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": output_schema(len(options))},
            }},
        )
    except Exception as e:
        print(f"⚠️ Walkthrough LLM call failed: {e}")
        return None

    if response.stop_reason in ("refusal", "max_tokens"):
        print(f"⚠️ Walkthrough not usable: stop_reason={response.stop_reason}")
        return None
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        print("⚠️ Walkthrough returned non-JSON text")
        return None

    result = validate_walkthrough(raw, question, options, correct)
    if result is None:
        print(f"⚠️ Walkthrough rejected by validation for {question[:60]!r}")
        return None
    _CACHE[key] = result
    return result
