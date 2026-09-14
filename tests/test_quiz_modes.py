# -*- coding: utf-8 -*-
"""
Applied quiz mode — the parts that run without an LLM.

Three post-exam debriefs described the same question shape their fundamentals
exams used: a named condition plus what the nurse monitors or does about it.
Neither existing template can produce it — knowledge mode bans patient
scenarios outright, NCLEX mode always asks what comes FIRST. "applied" is the
middle rung, and this file covers everything about it that is decidable without
calling the model.

The distribution test is the important one. `distribute_quiz_modes` decides how
much of every study quiz node is applied; if it silently returns the wrong
count, or blocks all the recall questions before all the applied ones, nothing
downstream notices — the quiz still renders, it just tests the wrong thing.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_quiz_modes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sata_prompts import (  # noqa: E402
    APPLIED_FRACTION,
    SATA_APPLIED_PROMPT_TEMPLATE,
    SATA_KNOWLEDGE_PROMPT_TEMPLATE,
    SATA_PROMPT_TEMPLATE,
    distribute_quiz_modes,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
print("\nMode distribution — counts")

# The ratio a student actually meets on a 5-question node.
check("difficulty 1 gives 1 applied of 5",
      distribute_quiz_modes(5, 1).count("applied") == 1,
      distribute_quiz_modes(5, 1))
check("difficulty 2 gives 3 applied of 5",
      distribute_quiz_modes(5, 2).count("applied") == 3,
      distribute_quiz_modes(5, 2))
check("difficulty 3 gives 4 applied of 5",
      distribute_quiz_modes(5, 3).count("applied") == 4,
      distribute_quiz_modes(5, 3))

# Half-up rounding, not Python's banker's rounding. round(0.5 * 5) is 2; the
# node is supposed to lean applied at difficulty 2, so it must be 3.
check("difficulty 2 rounds half UP, not to even",
      distribute_quiz_modes(5, 2).count("applied") == 3,
      "banker's rounding would give 2")

for n in (1, 2, 3, 5, 12):
    for diff in (1, 2, 3):
        seq = distribute_quiz_modes(n, diff)
        check(f"length preserved n={n} difficulty={diff}", len(seq) == n, seq)
        check(f"only known modes n={n} difficulty={diff}",
              set(seq) <= {"knowledge", "applied"}, seq)


# ---------------------------------------------------------------------------
print("\nMode distribution — unknown difficulty falls back to recall-leaning")

# A node with no difficulty must never land on the applied-heavy end. Being
# wrong in the recall direction is a duller quiz; being wrong the other way
# tests application before anything has been taught.
recall_leaning = APPLIED_FRACTION[1]
for missing in (None, "x", "", [], {}):
    seq = distribute_quiz_modes(5, missing)
    check(f"difficulty {missing!r} falls back to the recall-leaning fraction",
          seq.count("applied") == round(recall_leaning * 5), seq)

check("difficulty given as a numeric string still works",
      distribute_quiz_modes(5, "2").count("applied") == 3,
      distribute_quiz_modes(5, "2"))

# Real plans contain difficulty-4 quiz nodes. A number outside the table must be
# CLAMPED, never dropped to the default — dropping would give the hardest node in
# the plan the most recall-heavy mix.
check("difficulty 4 clamps UP to the difficulty-3 mix",
      distribute_quiz_modes(5, 4).count("applied")
      == distribute_quiz_modes(5, 3).count("applied"),
      distribute_quiz_modes(5, 4))
check("difficulty 99 clamps to the difficulty-3 mix",
      distribute_quiz_modes(5, 99).count("applied") == 4,
      distribute_quiz_modes(5, 99))
check("difficulty 0 clamps DOWN to the difficulty-1 mix",
      distribute_quiz_modes(5, 0).count("applied") == 1,
      distribute_quiz_modes(5, 0))
check("negative difficulty clamps to the difficulty-1 mix",
      distribute_quiz_modes(5, -2).count("applied") == 1,
      distribute_quiz_modes(5, -2))

check("zero questions returns an empty plan", distribute_quiz_modes(0, 2) == [])
check("negative question count returns an empty plan",
      distribute_quiz_modes(-3, 2) == [])


# ---------------------------------------------------------------------------
print("\nMode distribution — interleaved, not blocked")


def is_blocked(seq):
    """True if every applied question is clustered together in one run.

    Only meaningful once there are at least two of each mode — a lone applied
    question among recall ones cannot be 'blocked' anywhere.
    """
    if seq.count("applied") < 2 or seq.count("knowledge") < 2:
        return False
    first_applied = seq.index("applied")
    last_applied = len(seq) - 1 - seq[::-1].index("applied")
    applied_run = seq[first_applied:last_applied + 1]
    return "knowledge" not in applied_run


for n in (5, 8, 12):
    for diff in (1, 2, 3):
        seq = distribute_quiz_modes(n, diff)
        if seq.count("applied") >= 2 and seq.count("knowledge") >= 2:
            check(f"mixed batch is interleaved n={n} difficulty={diff}",
                  not is_blocked(seq), seq)

# A single applied question should not be stranded in the last slot, where a
# student who quits early never reaches the rung the node exists to teach.
lone = distribute_quiz_modes(5, 1)
check("a lone applied question is not stranded in the final slot",
      lone.index("applied") < len(lone) - 1, lone)


# ---------------------------------------------------------------------------
print("\nSATA template selection")

check("applied template is distinct from the other two",
      SATA_APPLIED_PROMPT_TEMPLATE not in (SATA_KNOWLEDGE_PROMPT_TEMPLATE,
                                           SATA_PROMPT_TEMPLATE))
check("applied SATA template declares its mode",
      '"quizMode": "applied"' in SATA_APPLIED_PROMPT_TEMPLATE)

# The bans are the whole contract of this rung. If they drift out of the
# template the model reverts to whichever neighbouring rung it likes.
for banned in ("FIRST", "PRIORITY", "MOST IMPORTANT", "IMMEDIATE", "INITIAL"):
    check(f"applied SATA template bans '{banned}' in the stem",
          banned in SATA_APPLIED_PROMPT_TEMPLATE)

check("applied SATA template carries the grounding rule",
      "GROUNDING RULE" in SATA_APPLIED_PROMPT_TEMPLATE)
check("applied SATA template requires a grounding field",
      '"grounding"' in SATA_APPLIED_PROMPT_TEMPLATE)

# The knowledge template must keep its own ban — an applied question leaking
# into a recall slot is the regression this guards.
check("knowledge SATA template still bans patient scenarios",
      "DO NOT use patient scenarios" in SATA_KNOWLEDGE_PROMPT_TEMPLATE)
check("applied SATA template does NOT ban patient scenarios",
      "DO NOT use patient scenarios" not in SATA_APPLIED_PROMPT_TEMPLATE)

# PromptTemplate fills these by name; a missing one raises at generation time.
for field in ("{topic}", "{difficulty}", "{question_num}", "{language}",
              "{questions_to_avoid}", "{content}", "{num_correct}"):
    check(f"applied SATA template keeps the {field} placeholder",
          field in SATA_APPLIED_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
print("\nMCQ mode normalisation")

import tools.quiztools as quiztools  # noqa: E402

src = open(quiztools.__file__.replace(".pyc", ".py"), encoding="utf-8").read()

check("'applied' survives normalisation instead of being downgraded",
      '["nclex", "knowledge", "applied"]' in src)
check("MCQ generator has an applied branch",
      'elif quiz_mode == "applied":' in src)
check("MCQ applied template declares its mode",
      '"quizMode": "applied"' in src)
check("MCQ applied branch carries the grounding rule",
      "GROUNDING RULE" in src)
check("MCQ applied branch bans priority wording",
      "NO PRIORITY RANKING" in src)

# Applied questions have ONE right answer, not a best one. Reusing the NCLEX
# answer instruction here is what would quietly turn rung 2 into rung 3, so
# check the applied blocks themselves rather than counting occurrences globally
# (the NCLEX branch legitimately uses that phrasing twice).
applied_starts = [i for i in range(len(src))
                  if src.startswith('elif quiz_mode == "applied":', i)]
check("both applied branches exist (answer instruction + template)",
      len(applied_starts) == 2, f"found {len(applied_starts)}")

applied_blocks = []
for start in applied_starts:
    nxt = src.find("\n        else:", start)
    alt = src.find("\n    else:", start)
    applied_blocks.append(src[start:min(x for x in (nxt, alt) if x != -1)])

answer_block, template_block = applied_blocks

# The answer instruction must demand the ONLY correct answer, never the best
# one — "best" is what makes a question a ranking exercise.
check("applied answer instruction demands the ONLY correct answer",
      "ONLY correct answer" in answer_block)
check("applied answer instruction does not tell the model to pick the best option",
      "should be the BEST choice" not in answer_block)
check("applied answer instruction rejects 'less optimal' distractors",
      'not "less optimal", actually wrong' in answer_block)

# In the template, BEST appears several times — the banned-word list, the STEP 5
# validation line and the JSON check field. Every one must be a prohibition, so
# assert the ban is present and that nothing ever asks for "the BEST" option.
check("applied template lists BEST among the banned stem words",
      "IMMEDIATE, INITIAL, BEST." in template_block)
check("applied template never asks for 'the BEST' option",
      "the BEST" not in template_block)


# ---------------------------------------------------------------------------
print("\nConcept extraction grounding")

import services.quiz_with_bank as qwb  # noqa: E402

qwb_src = open(qwb.__file__.replace(".pyc", ".py"), encoding="utf-8").read()

check("concept extractor has an applied branch",
      'elif quiz_mode == "applied":' in qwb_src)
check("applied extractor may return fewer concepts than asked",
      "return FEWER" in qwb_src)
check("applied extractor forbids inventing a condition",
      "Do not introduce a condition the content never mentions" in qwb_src)
check("mode plan helper exists",
      "async def _extract_concepts_for_mode_plan" in qwb_src)
check("generators receive the per-slot mode, not the batch mode",
      "quiz_mode=current_quiz_mode" in qwb_src
      and "quiz_mode=quiz_mode\n" not in qwb_src)


# ---------------------------------------------------------------------------
print("\nExam endpoint — applied is opt-in, not endpoint-wide")

from models.requests import StudyExamRequest, StudyItemRequest  # noqa: E402

req = StudyExamRequest(chat_id="c", topic="t")
# /study/generate-exam is shared by three callers. The adaptive priority drill
# sends instructions demanding "what the nurse does FIRST", which the applied
# template bans — so the DEFAULT must stay knowledge and only the study exam
# node may opt in. A default of "applied" here would silently break drills.
check("exam request defaults to knowledge mode", req.quiz_mode == "knowledge")
check("exam request carries a difficulty default", req.difficulty == 2)
check("exam request accepts applied when asked",
      StudyExamRequest(chat_id="c", topic="t", quiz_mode="applied").quiz_mode
      == "applied")

item = StudyItemRequest(chat_id="c", node_type="quiz", node_label="l")
check("study item difficulty defaults to the recall-leaning end",
      item.difficulty == 1)

main_src = open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "main.py"), encoding="utf-8").read()
check("exam endpoint honours the requested mode rather than hardcoding it",
      'quiz_mode=request.quiz_mode or "knowledge"' in main_src)
check("exam endpoint forwards node difficulty",
      "node_difficulty=request.difficulty" in main_src)
# The diagnostic runs before anything has been taught — it must stay recall.
check("diagnostic is carved out of applied mode",
      main_src.count('quiz_mode="knowledge" if request.is_diagnostic else "applied"') == 2)


# ---------------------------------------------------------------------------

print("\n" + "=" * 58)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("All quiz mode tests passed.")
