# -*- coding: utf-8 -*-
"""
Quiz walkthrough validation: the tutor may explain the answer key, never
contradict it.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_quiz_walkthrough.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.quiz_walkthrough import (  # noqa: E402
    build_prompt, cache_key, clean_option, output_schema, validate_walkthrough,
)

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else f"  -> {detail}"))
    if not condition:
        failures.append(name)


QUESTION = "The nurse is giving digoxin, which slows the heart. Which actions are appropriate? Select all that apply."
OPTIONS = ["A) Check the apical pulse for a full minute", "B) Hold the dose if the pulse is below 60",
           "C) Give it with a high-fiber meal", "D) Report nausea or yellow-green vision",
           "E) Encourage a low-potassium diet"]
KEY = [0, 1, 3]
GOOD = {
    "breakdown": [
        {"role": "task", "quote": "Which actions are appropriate", "meaning": "pick every safe action"},
        {"role": "problem", "quote": "digoxin, which slows the heart", "meaning": "the heart is already slowed"},
    ],
    "test_question": "Does this protect a slowed heart?",
    "method_line": "One fact, one question, every option.",
    "options": [
        {"index": 0, "verdict": True, "part": 1, "chain": ["Real heart rate", "needed before slowing it"], "hint": "What do you need before slowing a heart?"},
        {"index": 1, "verdict": True, "part": 1, "chain": ["Already slow", "slowing more is dangerous"], "hint": "The heart is already slow."},
        {"index": 2, "verdict": False, "part": -1, "chain": ["Nothing about the heart", "fiber blocks absorption"], "hint": "Does a meal protect the heart?"},
        {"index": 3, "verdict": True, "part": 1, "chain": ["Toxicity signs", "catch it early"], "hint": "What does too much digoxin look like?"},
        {"index": 4, "verdict": False, "part": 7, "chain": ["Low potassium", "raises toxicity"], "hint": "Think about potassium and digoxin."},
    ],
}

out = validate_walkthrough(copy.deepcopy(GOOD), QUESTION, OPTIONS, KEY)
check("a correct walkthrough passes", out is not None)
check("verdicts come from the key", [o["verdict"] for o in out["options"]] == [True, True, False, True, False])
check("the key fact is kept when it is quoted from the question", out["key_fact"] == "digoxin, which slows the heart")
check("breakdown parts come back in role order: problem, then task",
      [p["role"] for p in out["breakdown"]] == ["problem", "task"], out["breakdown"])
check("option parts follow the reorder", out["options"][0]["part"] == 0, out["options"][0])
check("an out-of-range part points at nothing", out["options"][4]["part"] == -1, out["options"][4])

wrong = copy.deepcopy(GOOD); wrong["options"][2]["verdict"] = True
check("ANY verdict that contradicts the key discards the whole walkthrough",
      validate_walkthrough(wrong, QUESTION, OPTIONS, KEY) is None)

invented = copy.deepcopy(GOOD); invented["breakdown"][1]["quote"] = "digoxin lowers the heart rate"
inv = validate_walkthrough(invented, QUESTION, OPTIONS, KEY)
check("a part not quoted from the question is dropped, not highlighted",
      inv is not None and [p["role"] for p in inv["breakdown"]] == ["task"], inv and inv["breakdown"])
nothing = copy.deepcopy(GOOD); nothing["breakdown"] = [{"role": "problem", "quote": "made up", "meaning": "x"}]
check("with no real part left, the walkthrough is withheld", validate_walkthrough(nothing, QUESTION, OPTIONS, KEY) is None)
twice = copy.deepcopy(GOOD); twice["breakdown"].append({"role": "problem", "quote": "Select all that apply", "meaning": "y"})
check("a second part with the same role is dropped",
      len(validate_walkthrough(twice, QUESTION, OPTIONS, KEY)["breakdown"]) == 2)

missing = copy.deepcopy(GOOD); missing["options"] = missing["options"][:4]
check("a missing option is rejected", validate_walkthrough(missing, QUESTION, OPTIONS, KEY) is None)

dup = copy.deepcopy(GOOD); dup["options"][4]["index"] = 3
check("a duplicated option is rejected", validate_walkthrough(dup, QUESTION, OPTIONS, KEY) is None)

thin = copy.deepcopy(GOOD); thin["options"][0]["chain"] = ["Yes"]
check("a one-link chain shows no reasoning and is rejected", validate_walkthrough(thin, QUESTION, OPTIONS, KEY) is None)

shuffled = copy.deepcopy(GOOD); shuffled["options"].reverse()
s = validate_walkthrough(shuffled, QUESTION, OPTIONS, KEY)
check("options come back in question order whatever the model's order", [o["index"] for o in s["options"]] == [0, 1, 2, 3, 4])

long = copy.deepcopy(GOOD); long["options"][0]["chain"] = ["x" * 200, "y", "z", "w"]
l = validate_walkthrough(long, QUESTION, OPTIONS, KEY)
check("chains are clipped to three short links", len(l["options"][0]["chain"]) == 3 and len(l["options"][0]["chain"][0]) <= 64)

dash = copy.deepcopy(GOOD); dash["method_line"] = "One fact — every option"
filler = copy.deepcopy(GOOD); filler["options"][0]["chain"] = ["Real heart rate", "needed before slowing it", "Answer: yes, true"]
fl = validate_walkthrough(filler, QUESTION, OPTIONS, KEY)
check("a step that only restates the verdict is dropped", fl["options"][0]["chain"] == ["Real heart rate", "needed before slowing it"], fl["options"][0]["chain"])
echo = copy.deepcopy(GOOD); echo["options"][1]["chain"] = ["Checks breathing now", "Checking breathing answers the test question", "Yes"]
check("steps that only echo the test question leave too little reasoning, so the walkthrough is withheld",
      validate_walkthrough(echo, QUESTION, OPTIONS, KEY) is None)
fits = copy.deepcopy(GOOD); fits["options"][0]["chain"] = ["Checks breathing now", "Looks at chest rise", "Assessing only, no bagging, so it fits"]
check("a step that only says it fits the test is dropped", validate_walkthrough(fits, QUESTION, OPTIONS, KEY)["options"][0]["chain"] == ["Checks breathing now", "Looks at chest rise"])
notfit = copy.deepcopy(GOOD); notfit["options"][1]["chain"] = ["Already breathing", "Bagging overrides it", "Forcing air in does not fit the test"]
check("a step that says it does not fit the test is dropped", validate_walkthrough(notfit, QUESTION, OPTIONS, KEY)["options"][1]["chain"] == ["Already breathing", "Bagging overrides it"])
check("em dashes never reach the student", "—" not in validate_walkthrough(dash, QUESTION, OPTIONS, KEY)["method_line"])

check("junk is rejected", validate_walkthrough(None, QUESTION, OPTIONS, KEY) is None
      and validate_walkthrough({"options": "x"}, QUESTION, OPTIONS, KEY) is None)

prompt = build_prompt(QUESTION, OPTIONS, KEY, "en")
check("the prompt hands the model the key", "A, B, D" in prompt)
check("option letter prefixes are not doubled", "A) A)" not in prompt and clean_option("C) Give") == "Give")
check("the cache keys on the answer key too", cache_key(QUESTION, OPTIONS, KEY, "en") != cache_key(QUESTION, OPTIONS, [0, 1], "en"))
check("the schema forbids extra fields", output_schema(5)["additionalProperties"] is False)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("All checks passed.")
