# -*- coding: utf-8 -*-
"""
============================================
Framework Question Generation
============================================

Questions that test a CLASSIFICATION SKILL rather than a topic.

    "The nurse and patient agree the patient will ambulate 50 feet by Friday.
     Which step of the nursing process is the nurse in?"
        A) Assessment   B) Diagnosis   C) Planning   D) Implementation

The clinical content is almost incidental; the skill is placing an action into a
framework. Every other generator in this codebase is topic-first and therefore
cannot produce this item — the item is not about a topic.

Frameworks come from the closed set in `constants/nursing_frameworks.py`, which
also supplies the categories used as answer options. See that module for why the
set is closed and how detection decides a document teaches one.

WHY THE OPTIONS ARE BUILT IN PYTHON, NOT BY THE MODEL
------------------------------------------------------

For a classify item the answer options ARE the framework's categories — we know
them before we ask for anything. So the model is given the finished option list
and asked only to write the scenario and say which letter it matches. Asking it
to emit the options too invites drift ("Assessment" becoming "Assessment phase"),
and a drifted option silently breaks marking, because the frontend compares the
answer string to the option string.

That constraint is also what makes `validate_framework_question` a real gate
rather than a shape check: in English we can assert the options ARE the registry's
categories, exactly. No other generator here can check its own output that hard.

TWO ITEM MODES, CHOSEN BY THE FRAMEWORK
----------------------------------------

`categories` — one scenario, the categories as options. Needs >= 4 categories to
    make a respectable MCQ. Erikson has 8, so we sample the correct one plus
    confusable distractors rather than listing all of them.

`stems` — several scenarios as options, one category in the question. Used when
    a framework has too few categories for the above (subjective/objective is a
    coin flip as a 2-option item) and for every `prioritize` framework, where the
    options are competing patients and the framework is the ordering rule, not a
    label set.

COVERAGE
--------

`plan_category_coverage` rotates through every category and then weights the
confusable pairs. This exists because of a specific request from a real student
after her exam: "covering all steps including planning". Planning is the step
students lose, and a generator left to itself keeps writing Assessment items.

Usage:
    from tools.framework_prompts import generate_framework_question, plan_category_coverage

    targets = plan_category_coverage("nursing_process", 5)
    for i, target in enumerate(targets):
        q = await generate_framework_question(
            framework_id="nursing_process",
            target_category=target,
            question_num=i + 1,
            difficulty="medium",
            language="english",
            content_context=document_text,
        )
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from constants.nursing_frameworks import (
    SHAPE_PRIORITIZE,
    get_framework,
)

# A classify item needs at least this many options to be worth asking. Below it
# the framework switches to `stems` mode instead of shipping a coin flip.
MIN_OPTIONS = 4

# Upper bound on options. Erikson's eight stages as eight choices is a wall of
# text on a phone; five is the most any question here should ask a student to read.
MAX_OPTIONS = 5

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ============================================
# MODE + OPTION CONSTRUCTION
# ============================================

def option_mode(framework_id: str) -> Optional[str]:
    """
    'categories' or 'stems' for this framework, or None if the id is unknown.

    Prioritize frameworks are always `stems`: "which patient do you see first"
    has patients as its options, and Airway/Breathing/Circulation is the reason
    for the answer rather than the answer itself.
    """
    fw = get_framework(framework_id)
    if not fw:
        return None
    if fw["item_shape"] == SHAPE_PRIORITIZE:
        return "stems"
    return "categories" if len(fw["categories"]) >= MIN_OPTIONS else "stems"


def _confusable_partners(fw: Dict[str, Any], category: str) -> List[str]:
    """Categories this one is specifically mistaken for, per the registry."""
    partners = []
    for a, b in fw.get("confusable", []):
        if a == category and b not in partners:
            partners.append(b)
        elif b == category and a not in partners:
            partners.append(a)
    return partners


def build_option_set(
    framework_id: str,
    target_category: str,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], str]:
    """
    The finished option list for a classify item, and the correct option string.

    Distractors prefer the target's confusable partners — an item whose wrong
    answers are the categories nobody confuses with it tests nothing. Remaining
    slots are filled at random from the rest, then the whole list is shuffled so
    the answer is not always in the same position.

    Returns ([...'A) Assessment'...], 'C) Planning').
    """
    rng = rng or random
    fw = get_framework(framework_id)
    if not fw or target_category not in fw["categories"]:
        raise ValueError(f"{framework_id} has no category {target_category!r}")

    others = [c for c in fw["categories"] if c != target_category]
    preferred = [c for c in _confusable_partners(fw, target_category) if c in others]
    rest = [c for c in others if c not in preferred]
    rng.shuffle(rest)

    picked = (preferred + rest)[: MAX_OPTIONS - 1]
    chosen = picked + [target_category]
    rng.shuffle(chosen)

    options = [f"{LETTERS[i]}) {c}" for i, c in enumerate(chosen)]
    answer = options[chosen.index(target_category)]
    return options, answer


def plan_category_coverage(
    framework_id: str,
    count: int,
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Which categories the next `count` questions should target.

    Every category appears once before any repeats — a set of five nursing-process
    items that never asks about Evaluation has not tested the nursing process.
    Extra questions beyond one full pass go to the confusable categories, which is
    where students actually lose marks.
    """
    rng = rng or random
    fw = get_framework(framework_id)
    if not fw or count <= 0:
        return []

    categories = list(fw["categories"])
    rng.shuffle(categories)

    plan: List[str] = []
    while len(plan) < count:
        remaining = count - len(plan)
        if remaining >= len(categories):
            plan.extend(categories)
            continue
        # Part of a pass left: spend it on the confusable ones first.
        hard = [c for c in categories
                if any(c in pair for pair in fw.get("confusable", []))]
        pool = hard or categories
        pool = [c for c in pool if c not in plan[-len(categories):]] or pool
        plan.extend(rng.sample(pool, min(remaining, len(pool))))

    return plan[:count]


# ============================================
# PROMPTS
# ============================================

# Model writes ONLY the scenario, the answer letter, and a one-line blurb. The
# options are already decided — see the module docstring.
FRAMEWORK_CLASSIFY_TEMPLATE = """
You are a {language}-speaking nursing question writer.

Write EXACTLY ONE multiple-choice question that asks the student to classify
{stem_subject} using this framework:

FRAMEWORK: {framework_name}
THE QUESTION MUST BE: "{question_ask}"

The answer options are FIXED and already chosen. Do not invent, reword, reorder
or add options:
{options_block}

THE CORRECT ANSWER MUST BE: {answer}

So: write a scenario that unambiguously belongs to "{target_category}" and to
none of the other options.

Difficulty: {difficulty}
Question number: {question_num}

DO NOT repeat any of these previously-asked questions:
{questions_to_avoid}

Ground the scenario in this source material where you can — use its clinical
setting, patient types and vocabulary so the question feels like the student's
own course:
{content}

REQUIREMENTS

1. The scenario is 1-2 sentences describing {stem_subject}. Concrete and
   clinical: a real action, a real patient, real numbers where they help.
2. It must fit "{target_category}" and NOT the other options. A student who knows
   the framework should have no doubt; a student who does not should be tempted by
   {confusable_note}
3. Do NOT name the category in the scenario. "The nurse plans to..." gives away
   Planning. Describe what the nurse DOES, and let the student classify it.
4. Do NOT mention the framework itself in the scenario.
5. `correct_blurb` is ONE plain-text sentence, 25 words maximum, saying what makes
   it that category. No HTML. No listing the other options.
6. Write everything in {language}.

Return ONLY valid JSON, no markdown wrapper:
{{
    "question": "The scenario, ending with: {question_ask}",
    "answer": "{answer}",
    "correct_blurb": "One sentence, max 25 words, on what makes it this category.",
    "topic": "2-4 word specific topic in {language}"
}}
"""


# Here the model DOES write the options, because they are scenarios rather than
# a known label set. Validation falls back to "answer is one of the options".
FRAMEWORK_STEMS_TEMPLATE = """
You are a {language}-speaking nursing question writer.

Write EXACTLY ONE multiple-choice question built on this framework:

FRAMEWORK: {framework_name}
Its categories: {categories_list}

TASK: {stems_task}

Difficulty: {difficulty}
Question number: {question_num}

DO NOT repeat any of these previously-asked questions:
{questions_to_avoid}

Ground the options in this source material where you can:
{content}

REQUIREMENTS

1. Write EXACTLY 4 options, labelled "A) " through "D) ".
2. Each option is a short, concrete clinical item — an action, a finding, a
   patient, or a task. One sentence at most.
3. EXACTLY ONE option is correct. The other three must be genuinely plausible:
   a student who does not know the framework should find them all reasonable.
4. Do NOT name the framework or its categories inside the options — that gives
   the answer away. The categories may appear in the question stem.
5. `correct_blurb` is ONE plain-text sentence, 25 words maximum, naming the
   category that decides it. No HTML.
6. Write everything in {language}.

Return ONLY valid JSON, no markdown wrapper:
{{
    "question": "The question stem",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "B) ... (must match one option EXACTLY, character for character)",
    "correct_blurb": "One sentence, max 25 words, naming the deciding category.",
    "topic": "2-4 word specific topic in {language}"
}}
"""


def _question_ask(fw: Dict[str, Any]) -> str:
    """The sentence the stem ends on, phrased per framework."""
    name = fw["name"]
    if fw["item_shape"] == SHAPE_PRIORITIZE:
        return f"Which should the nurse address FIRST?"
    if fw["id"] == "delegation_scope":
        return "Who should perform this task?"
    return f"Which part of {name} does this represent?"


def _stems_task(fw: Dict[str, Any], target_category: str) -> str:
    """What the model is being asked to build in stems mode."""
    if fw["item_shape"] == SHAPE_PRIORITIZE:
        return (
            f"Give four items competing for the nurse's attention. Exactly one is "
            f"the correct priority under {fw['name']}. Ask which the nurse should "
            f"address first."
        )
    return (
        f'Give four items of the kind described as "{fw["stem_subject"]}". '
        f'Exactly ONE of them belongs to the category "{target_category}"; the '
        f'other three belong to different categories of {fw["name"]}. Ask which '
        f'one is "{target_category}".'
    )


# ============================================
# GENERATION
# ============================================

async def generate_framework_question(
    framework_id: str,
    target_category: str,
    question_num: int,
    difficulty: str = "medium",
    language: str = "english",
    content_context: str = "",
    questions_to_avoid: Optional[List[str]] = None,
    nclex_framing: bool = False,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """
    One framework question, in the MCQ shape the frontend already renders.

    Deliberately emits `questionType: "mcq"` rather than a new type: the item is
    a normal single-answer multiple choice and every existing renderer, scorer and
    rating surface handles it unchanged. What makes it a framework question lives
    in `metadata.framework`, which is what later analysis groups on.

    Returns None on failure, matching `generate_sata_question` — the caller drops
    the question rather than shipping a broken one.
    """
    rng = rng or random
    fw = get_framework(framework_id, nclex_framing=nclex_framing)
    if not fw:
        print(f"❌ Unknown framework: {framework_id}")
        return None

    if target_category not in fw["categories"]:
        print(f"❌ {framework_id} has no category {target_category!r}")
        return None

    questions_to_avoid = questions_to_avoid or []
    avoid_text = ("\n".join(f"- {q}" for q in questions_to_avoid)
                  if questions_to_avoid else "None - this is the first question")

    if not content_context:
        content_context = (
            "(No source excerpt available — write a generic but clinically "
            "realistic scenario appropriate to a nursing fundamentals course.)"
        )
    content_context = content_context[:3000]

    mode = option_mode(framework_id)
    ask = _question_ask(fw)

    print(f"\n{'='*60}")
    print(f"🧭 Generating framework question {question_num}")
    print(f"   Framework: {fw['name']} ({framework_id}), mode={mode}")
    print(f"   Target category: {target_category}")
    print(f"   Difficulty: {difficulty} | Language: {language}")
    print(f"{'='*60}\n")

    if mode == "categories":
        options, answer = build_option_set(framework_id, target_category, rng=rng)
        partners = _confusable_partners(fw, target_category)
        confusable_note = (
            ", ".join(partners) + "." if partners else "one of the other options."
        )
        prompt = PromptTemplate(
            input_variables=[
                "language", "framework_name", "question_ask", "options_block",
                "answer", "target_category", "difficulty", "question_num",
                "questions_to_avoid", "content", "stem_subject", "confusable_note",
            ],
            template=FRAMEWORK_CLASSIFY_TEMPLATE,
        )
        payload = {
            "language": language,
            "framework_name": fw["name"],
            "question_ask": ask,
            "options_block": "\n".join(options),
            "answer": answer,
            "target_category": target_category,
            "difficulty": difficulty,
            "question_num": question_num,
            "questions_to_avoid": avoid_text,
            "content": content_context,
            "stem_subject": fw["stem_subject"],
            "confusable_note": confusable_note,
        }
    else:
        options, answer = None, None
        prompt = PromptTemplate(
            input_variables=[
                "language", "framework_name", "categories_list", "stems_task",
                "difficulty", "question_num", "questions_to_avoid", "content",
            ],
            template=FRAMEWORK_STEMS_TEMPLATE,
        )
        payload = {
            "language": language,
            "framework_name": fw["name"],
            "categories_list": " | ".join(fw["categories"]),
            "stems_task": _stems_task(fw, target_category),
            "difficulty": difficulty,
            "question_num": question_num,
            "questions_to_avoid": avoid_text,
            "content": content_context,
        }

    llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    try:
        raw = await chain.ainvoke(payload)
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)

        if mode == "categories":
            # The model was told the options and the answer; it does not get to
            # change either. Overwriting rather than trusting is the whole point.
            parsed["options"] = options
            parsed["answer"] = answer
        else:
            parsed["options"] = list(parsed.get("options") or [])

        question = _assemble(parsed, fw, target_category, mode, difficulty,
                             language, question_num)

        ok, errors = validate_framework_question(question, framework_id,
                                                 nclex_framing=nclex_framing)
        if not ok:
            print(f"❌ Framework question {question_num} failed validation: {errors}")
            return None

        print(f"✅ Framework question {question_num} generated ({fw['name']} → {target_category})")
        return question

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse framework question {question_num}: {e}")
        if "raw" in locals():
            print(f"Raw output: {raw[:500]}...")
        return None
    except Exception as e:
        print(f"❌ Error generating framework question {question_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _assemble(parsed, fw, target_category, mode, difficulty, language, question_num):
    """Model output + what we already knew → the MCQ dict the frontend expects."""
    options = parsed.get("options", [])
    answer = parsed.get("answer", "")
    try:
        answer_index = options.index(answer)
    except ValueError:
        answer_index = -1

    return {
        "question": (parsed.get("question") or "").strip(),
        "questionType": "mcq",
        "quizMode": "knowledge",
        "options": options,
        "answer": answer,
        "correct_blurb": (parsed.get("correct_blurb") or "").strip(),
        "topic": (parsed.get("topic") or fw["name"]).strip(),
        "metadata": {
            "sourceLanguage": language,
            "topic": parsed.get("topic") or fw["name"],
            "category": "nursing",
            "difficulty": difficulty,
            "quizMode": "knowledge",
            "correctAnswerIndex": answer_index,
            "sourceDocument": "framework_generation",
            # What makes this a framework item. Analysis groups on these, and the
            # post-exam debrief's calibration findings need them to say "your exam
            # had five nursing-process items and you practised none".
            "framework": fw["id"],
            "frameworkName": fw["name"],
            "frameworkCategory": target_category,
            "frameworkMode": mode,
            "itemShape": fw["item_shape"],
        },
    }


# ============================================
# VALIDATION
# ============================================

def validate_framework_question(
    question: Dict[str, Any],
    framework_id: str,
    *,
    nclex_framing: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Is this a well-formed, markable framework question?

    In `categories` mode and English this is unusually strict: the options must BE
    the registry's categories, so a drifted label ("Assessment phase") is caught
    before it reaches a student and silently breaks marking. Other languages get
    the structural checks only, because the categories are translated at
    generation time and the registry holds English — see the note below.

    Returns (is_valid, errors).
    """
    errors: List[str] = []
    fw = get_framework(framework_id, nclex_framing=nclex_framing)
    if not fw:
        return False, [f"unknown framework: {framework_id}"]

    for field in ("question", "questionType", "options", "answer", "correct_blurb"):
        if not question.get(field):
            errors.append(f"missing or empty field: {field}")

    if question.get("questionType") != "mcq":
        errors.append(f"questionType must be 'mcq', got {question.get('questionType')!r}")

    options = question.get("options") or []
    answer = question.get("answer")

    if not isinstance(options, list):
        errors.append("options must be a list")
    else:
        if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS + 1):
            errors.append(f"expected {MIN_OPTIONS}-{MAX_OPTIONS + 1} options, got {len(options)}")
        if len(set(options)) != len(options):
            errors.append("duplicate options")

    # The invariant that matters: a student's click is compared to this string.
    if isinstance(options, list) and answer not in options:
        errors.append(f"answer {answer!r} is not one of the options")

    meta = question.get("metadata") or {}
    if meta.get("framework") != framework_id:
        errors.append("metadata.framework does not match the requested framework")

    target = meta.get("frameworkCategory")
    if target not in fw["categories"]:
        errors.append(f"metadata.frameworkCategory {target!r} is not a category of {framework_id}")

    if meta.get("frameworkMode") == "categories":
        stripped = [o.split(") ", 1)[-1] for o in options if isinstance(o, str)]
        unknown = [s for s in stripped if s not in fw["categories"]]
        # Only enforced for English: for other languages the option labels are
        # translated at generation time while the registry stays English. Adding
        # per-language category labels to the registry would make this strict
        # everywhere — worth doing before promoting non-English framework items.
        if unknown and str(meta.get("sourceLanguage", "")).lower() in ("english", "en", ""):
            errors.append(f"options are not registry categories: {unknown}")
        if answer and isinstance(answer, str):
            answer_label = answer.split(") ", 1)[-1]
            if answer_label != target and not unknown:
                errors.append(
                    f"answer {answer_label!r} does not match target category {target!r}")

    blurb = question.get("correct_blurb") or ""
    if len(blurb.split()) > 40:
        errors.append(f"correct_blurb is {len(blurb.split())} words, expected <= 25ish")
    if "<" in blurb and ">" in blurb:
        errors.append("correct_blurb must be plain text, no HTML")

    return (len(errors) == 0), errors
