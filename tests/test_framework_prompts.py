# -*- coding: utf-8 -*-
"""
Framework question construction tests — the parts that run without an LLM.

Everything here is the half of generation that must NOT depend on the model:
which categories get covered, which distractors are chosen, and whether a
finished question is markable. `generate_framework_question` itself needs an
OpenAI call and is exercised by hand, not here.

The validation cases are the important ones. A framework question whose answer
string is not exactly one of its option strings scores every student wrong, and
that is precisely the failure the fixed-options design exists to prevent — so it
is worth a test that would catch it coming back.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_framework_prompts.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.nursing_frameworks import FRAMEWORKS, get_framework  # noqa: E402
from tools.framework_prompts import (  # noqa: E402
    MAX_OPTIONS,
    MIN_OPTIONS,
    build_option_set,
    option_mode,
    plan_category_coverage,
    validate_framework_question,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def make_question(framework_id, target, options, answer, *, mode="categories",
                  qtype="mcq", blurb="Setting a measurable goal with the patient is planning.",
                  language="english"):
    """A finished question dict, so tests can corrupt one field at a time."""
    return {
        "question": "The nurse and patient agree on a goal. Which part does this represent?",
        "questionType": qtype,
        "quizMode": "knowledge",
        "options": options,
        "answer": answer,
        "correct_blurb": blurb,
        "topic": "Nursing Process Steps",
        "metadata": {
            "sourceLanguage": language,
            "framework": framework_id,
            "frameworkCategory": target,
            "frameworkMode": mode,
        },
    }


# ---------------------------------------------------------------------------
print("\nOPTION MODE")
# ---------------------------------------------------------------------------

check("nursing process uses category options",
      option_mode("nursing_process") == "categories", option_mode("nursing_process"))
check("subjective/objective falls back to stems (2 categories is a coin flip)",
      option_mode("data_type") == "stems", option_mode("data_type"))
check("therapeutic communication falls back to stems",
      option_mode("therapeutic_communication") == "stems")
check("delegation (3 roles) falls back to stems",
      option_mode("delegation_scope") == "stems")
check("ABC is stems because it is a prioritize framework",
      option_mode("abc_priority") == "stems", option_mode("abc_priority"))
check("maslow is stems because it is a prioritize framework",
      option_mode("maslow") == "stems")
check("unknown framework has no mode", option_mode("nope") is None)


# ---------------------------------------------------------------------------
print("\nOPTION CONSTRUCTION")
# ---------------------------------------------------------------------------

rng = random.Random(4)
options, answer = build_option_set("nursing_process", "Planning", rng=rng)

check("option count is capped", MIN_OPTIONS <= len(options) <= MAX_OPTIONS,
      len(options))
check("answer is one of the options", answer in options, (answer, options))
check("answer is the requested category", answer.split(") ", 1)[1] == "Planning", answer)
check("options are lettered from A", options[0].startswith("A) "), options[0])
check("letters are sequential",
      [o[0] for o in options] == ["A", "B", "C", "D", "E"][:len(options)],
      [o[0] for o in options])
check("no duplicate options", len(set(options)) == len(options))

stripped = [o.split(") ", 1)[1] for o in options]
check("every option is a real category",
      all(s in FRAMEWORKS["nursing_process"]["categories"] for s in stripped), stripped)
# Planning's registry partners are Diagnosis and Implementation.
check("confusable partners are included as distractors",
      "Diagnosis" in stripped and "Implementation" in stripped, stripped)

# Erikson has 8 categories — it must sample, not list them all.
e_options, e_answer = build_option_set("erikson", "Initiative vs. Guilt",
                                       rng=random.Random(9))
check("erikson samples down to at most MAX_OPTIONS", len(e_options) <= MAX_OPTIONS,
      len(e_options))
check("erikson answer still correct",
      e_answer.split(") ", 1)[1] == "Initiative vs. Guilt", e_answer)

# The answer must not always land in the same slot.
positions = set()
for seed in range(30):
    o, a = build_option_set("nursing_process", "Assessment", rng=random.Random(seed))
    positions.add(o.index(a))
check("answer position varies across seeds", len(positions) > 1, positions)

try:
    build_option_set("nursing_process", "Nonexistent")
    check("unknown category raises", False)
except ValueError:
    check("unknown category raises", True)


# ---------------------------------------------------------------------------
print("\nCOVERAGE PLANNING")
# ---------------------------------------------------------------------------

plan5 = plan_category_coverage("nursing_process", 5, rng=random.Random(1))
check("a 5-question plan covers all 5 steps",
      set(plan5) == set(FRAMEWORKS["nursing_process"]["categories"]), plan5)
check("a 5-question plan has 5 entries", len(plan5) == 5, len(plan5))

plan8 = plan_category_coverage("nursing_process", 8, rng=random.Random(2))
check("an 8-question plan still covers everything",
      set(FRAMEWORKS["nursing_process"]["categories"]).issubset(set(plan8)), plan8)
check("an 8-question plan has 8 entries", len(plan8) == 8)
# The student's actual request was "covering all steps including planning".
check("planning is never dropped", "Planning" in plan8, plan8)

plan3 = plan_category_coverage("nursing_process", 3, rng=random.Random(3))
check("a short plan is still the right length", len(plan3) == 3, plan3)
check("a short plan does not repeat itself", len(set(plan3)) == 3, plan3)

check("zero questions gives an empty plan",
      plan_category_coverage("nursing_process", 0) == [])
check("unknown framework gives an empty plan",
      plan_category_coverage("nope", 5) == [])

for fid in FRAMEWORKS:
    p = plan_category_coverage(fid, 4, rng=random.Random(7))
    cats = get_framework(fid)["categories"]
    check(f"{fid}: plan only contains real categories",
          all(c in cats for c in p), p)


# ---------------------------------------------------------------------------
print("\nVALIDATION")
# ---------------------------------------------------------------------------

good_options, good_answer = build_option_set("nursing_process", "Planning",
                                             rng=random.Random(4))
good = make_question("nursing_process", "Planning", good_options, good_answer)
ok, errs = validate_framework_question(good, "nursing_process")
check("a well-formed question validates", ok, errs)

# THE failure this design exists to prevent.
drifted = make_question("nursing_process", "Planning",
                        [o.replace("Planning", "Planning phase") for o in good_options],
                        good_answer)
ok, errs = validate_framework_question(drifted, "nursing_process")
check("a drifted option label is rejected", not ok, errs)

mismatched = make_question("nursing_process", "Planning", good_options,
                           "Z) Something else entirely")
ok, errs = validate_framework_question(mismatched, "nursing_process")
check("an answer absent from the options is rejected", not ok, errs)

wrong_target = make_question("nursing_process", "Evaluation", good_options, good_answer)
ok, errs = validate_framework_question(wrong_target, "nursing_process")
check("an answer that is not the target category is rejected", not ok, errs)

bad_type = make_question("nursing_process", "Planning", good_options, good_answer,
                         qtype="sata")
ok, _ = validate_framework_question(bad_type, "nursing_process")
check("a non-mcq questionType is rejected", not ok)

dupes = make_question("nursing_process", "Planning",
                      [good_options[0]] * len(good_options), good_options[0])
ok, _ = validate_framework_question(dupes, "nursing_process")
check("duplicate options are rejected", not ok)

too_few = make_question("nursing_process", "Planning", good_options[:2], good_options[0])
ok, _ = validate_framework_question(too_few, "nursing_process")
check("too few options is rejected", not ok)

html_blurb = make_question("nursing_process", "Planning", good_options, good_answer,
                           blurb="Setting a <strong>goal</strong> is planning.")
ok, errs = validate_framework_question(html_blurb, "nursing_process")
check("HTML in correct_blurb is rejected", not ok, errs)

rambling = make_question("nursing_process", "Planning", good_options, good_answer,
                         blurb=" ".join(["word"] * 60))
ok, _ = validate_framework_question(rambling, "nursing_process")
check("an over-long blurb is rejected", not ok)

wrong_fw = make_question("maslow", "Planning", good_options, good_answer)
ok, _ = validate_framework_question(wrong_fw, "nursing_process")
check("metadata naming a different framework is rejected", not ok)

# Stems mode: options are generated scenarios, so the category check must not fire.
stems_q = make_question(
    "data_type", "Subjective",
    ["A) The patient reports sharp pain at 7 out of 10",
     "B) Blood pressure is 148/92",
     "C) Temperature is 38.1 C",
     "D) Bowel sounds are present in all four quadrants"],
    "A) The patient reports sharp pain at 7 out of 10",
    mode="stems", blurb="Pain described by the patient is subjective data.")
ok, errs = validate_framework_question(stems_q, "data_type")
check("stems-mode question validates without category-shaped options", ok, errs)

# Non-English keeps the structural checks but drops the strict identity check.
french = make_question(
    "nursing_process", "Planning",
    ["A) Évaluation", "B) Diagnostic", "C) Planification", "D) Interventions"],
    "C) Planification", language="french",
    blurb="Fixer un objectif mesurable avec le patient relève de la planification.")
ok, errs = validate_framework_question(french, "nursing_process")
check("translated options are allowed in non-English", ok, errs)


# ---------------------------------------------------------------------------

print("\n" + "=" * 58)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("All framework prompt tests passed.")
