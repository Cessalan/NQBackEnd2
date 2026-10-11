"""
Writes a question that is hard by construction, for a student who asked for hard.

WHY THIS EXISTS (2026-10-11)

Measured on a student's own two chapters: ten "very hard" questions from the
general writer could each be answered by a model shown ONLY the options (20 of
20 tries), the key was the longest option in 9 of 10, and most had an option
no prepared student would consider. Telling the general writer "hard" and
checking afterwards removed the length cue but not the rest: its prompts ask
for distractors that are "reasonable but less optimal", which in practice means
one sensible answer and three weak ones.

So a hard question is BUILT in the order an item writer would: the decision it
tests, the findings that must be combined (borderline, not extreme), then for
each wrong option the nearby situation in which it WOULD be right. A distractor
that is right somewhere nearby is one a well-prepared student can be tempted
by; that is what makes the question hard rather than long. The plan is part of
the JSON so it can be inspected, and the result still goes through every check
in services/requirement_check.

The key's position is drawn here, not by the model, and the options are
written without letters, so neither the model nor the shuffle can leak it.
"""
import json
import os
import random
import re

from core.quiz_model import quiz_chat_model

REASONING = "medium"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

PROMPT = """You write ONE hard {kind} nursing exam question, in {language}, for a nursing student who asked for hard questions.

Build it in this order, and return every step in the JSON:
1. "decision": the one clinical decision the question tests: what to do first, which finding matters most, or whether care is working. Never the recall of a single fact.
2. "findings": 2 to 4 findings in the stem that must be COMBINED to reach the answer. Use values that need interpretation: borderline, a change from the client's own baseline, or a finding that matters only because of another one. No finding may be so extreme that it gives the answer away on its own.
3. "near_misses": for EACH wrong option, the nearby situation in which it would be the right action (for example "if the potassium were normal", "if the client were hypotensive"). Every wrong option must be a real, specific nursing action that a well-prepared student could defend. Never an unsafe, lazy or absurd option.
4. "question": the stem. It gives the findings and asks the decision. It must not name the conclusion the student is meant to reach.
5. "options": {n_options} options WITHOUT letters, each the same kind of action at the same length and level of detail. At least one wrong option also assesses or escalates. The correct option{s} must not name the diagnosis or conclusion, must not be the longest, and must not reuse a phrase that appears only in the stem.
6. "correct": the zero-based {correct_desc}.
7. "rationale": one or two sentences: why the key is right for THIS client, and which finding rules out the near-misses.
8. "topic": 2 to 4 words.

Test only what the source passage supports. Follow every requirement in it.

CONCEPT: {concept}

{content}

Return only the JSON object."""


def _parse(text):
    text = _FENCE.sub("", str(text or "").strip())
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    return data


def build_question(data, fmt, rng=random):
    """The app's question shape from the writer's JSON, or None when it is malformed.

    Pure: the key position is drawn here, so neither the model nor the order
    it wrote the options in can give the answer away.
    """
    options = [re.sub(r"^[A-H][).:]\s*", "", str(o)).strip() for o in data.get("options") or []]
    correct = data.get("correct")
    correct = [correct] if isinstance(correct, int) else correct
    if not isinstance(correct, list) or not all(isinstance(i, int) and 0 <= i < len(options) for i in correct):
        return None
    if fmt == "sata":
        if len(options) < 5 or not 2 <= len(set(correct)) <= len(options) - 1:
            return None
    elif len(options) != 4 or len(set(correct)) != 1:
        return None
    if not str(data.get("question") or "").strip() or len(set(o.lower() for o in options)) != len(options):
        return None
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    keys = sorted(order.index(i) for i in set(correct))
    lettered = [f"{chr(65 + i)}) {text}" for i, text in enumerate(shuffled)]
    rationale = str(data.get("rationale") or "").strip()
    question = {"question": str(data["question"]).strip(), "options": lettered,
                "questionType": "sata" if fmt == "sata" else "mcq",
                "topic": str(data.get("topic") or "").strip()[:60] or None,
                "correct_blurb": rationale, "justification": rationale,
                "hardPlan": {"decision": data.get("decision"), "findings": data.get("findings"),
                             "nearMisses": data.get("near_misses")}}
    if fmt == "sata":
        question["answer"] = [lettered[i] for i in keys]
        question["scoringType"] = "partial"
    else:
        question["answer"] = lettered[keys[0]]
        question["metadata"] = {"correctAnswerIndex": keys[0]}
    return question


def item_schema():
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "near_misses": {"type": "array", "items": {"type": "string"}},
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "correct": {"type": "array", "items": {"type": "integer"}},
            "rationale": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": ["decision", "findings", "near_misses", "question", "options", "correct", "rationale", "topic"],
        "additionalProperties": False,
    }


async def _write_claude(prompt, model):
    from services.requirement_check import _get_client
    response = await _get_client().messages.create(
        model=model, max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": item_schema()}}},
    )
    if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
        raise ValueError(f"stopped: {response.stop_reason}")
    return "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")


async def write_hard(fmt, concept, content, language="en", model=None):
    """One hard multiple-choice or select-all question, or None.

    `model` (or HARD_ITEM_MODEL) names a Claude model to write with; otherwise
    the quiz writer at REASONING effort."""
    sata = fmt == "sata"
    prompt = PROMPT.format(
        kind="select-all-that-apply" if sata else "single-answer multiple-choice",
        language={"fr": "French"}.get(str(language)[:2].lower(), "English"),
        n_options="6" if sata else "4", s="s" if sata else "",
        correct_desc="indices of the 2 to 4 correct options" if sata else "index of the one correct option",
        concept=concept, content=content)
    model = model or os.getenv("HARD_ITEM_MODEL")
    try:
        if model and model.startswith("claude"):
            text = await _write_claude(prompt, model)
        else:
            text = (await quiz_chat_model(timeout=120, reasoning=REASONING).ainvoke(prompt)).content
        return build_question(_parse(text), "sata" if sata else "mcq")
    except Exception as e:  # noqa: BLE001 - the caller falls back to the general writer
        print(f"⚠️ hard_items: write failed: {type(e).__name__}: {e}")
        return None
