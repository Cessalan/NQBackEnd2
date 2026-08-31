# -*- coding: utf-8 -*-
"""
Narration validation tests.

The knowledge map's sentences make factual claims about a student's
performance — "you've got that one", "you've got a real base". Those claims
are decided by deterministic code; /study/narrate only rewrites the WORDING
with a cheap model.

validate_narration is the wall between those two facts. If it lets a bad
rewrite through, the product tells a student something untrue about herself
in a warm voice, which is the worst failure mode available to us. So the
rejections matter more than the acceptances, and most of this file is
rejections.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_narration_validation.py
"""
import io
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load just the validator, without importing main.py (which builds LLM
# clients at import time and needs API keys).
SRC = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
start = SRC.index("MAX_NARRATION_LINES =")
end = SRC.index('@app.post("/study/narrate")')

mod = types.ModuleType("narration")
mod.re = re
exec(compile("import re\n" + SRC[start:end], "narration", "exec"), mod.__dict__)
validate = mod.validate_narration


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL  %s %s" % (name, detail))


ORIGINALS = [
    u"Okay — that tells me a lot.",
    u"You're in better shape than you might think. These are already yours:",
    u"Your exam is 12 days away — we've got time.",
    u"Here's the game plan. We start with Fluid Balance.",
]
TERMS = [u"Fluid Balance"]


print("\n=== accepts an honest rewrite ===")

good = [
    u"Right, that tells me plenty.",
    u"Better shape than you'd think. These are already yours:",
    u"You've got 12 days — that's real time.",
    u"So here's the plan. Fluid Balance first.",
]
check("a faithful paraphrase passes", validate(ORIGINALS, good, TERMS))


print("\n=== rejects invented numbers ===")

# The map deliberately shows no percentages. A model that adds one is
# reporting a score we never computed.
bad_num = list(good)
bad_num[1] = u"You're about 80% of the way there. These are already yours:"
check("a percentage that was never in the source is rejected",
      not validate(ORIGINALS, bad_num, TERMS))

bad_num2 = list(good)
bad_num2[2] = u"You've got 12 days — about 3 hours a day should do it."
check("invented advice carrying a new number is rejected",
      not validate(ORIGINALS, bad_num2, TERMS))

# Dropping a number is fine; only INVENTING one is dangerous.
ok_fewer = list(good)
ok_fewer[2] = u"You've got a couple of weeks — that's real time."
check("dropping a number is allowed", validate(ORIGINALS, ok_fewer, TERMS))


print("\n=== rejects a vanished topic name ===")

# "Fluid Balance" generalised into "your weak area" stops the map being
# about her, which is the entire point of the screen.
bad_topic = list(good)
bad_topic[3] = u"So here's the plan. Your weakest area first."
check("generalising a topic name away is rejected",
      not validate(ORIGINALS, bad_topic, TERMS))

renamed = list(good)
renamed[3] = u"So here's the plan. Fluids and Electrolytes first."
check("renaming a topic is rejected", not validate(ORIGINALS, renamed, TERMS))

# Case changes are cosmetic, not a rename.
recased = list(good)
recased[3] = u"So here's the plan. FLUID BALANCE first."
check("a case change is still the same topic", validate(ORIGINALS, recased, TERMS))


print("\n=== rejects malformed responses ===")

check("wrong line count is rejected", not validate(ORIGINALS, good[:3], TERMS))
check("extra lines are rejected", not validate(ORIGINALS, good + [u"Bonus!"], TERMS))
check("a non-list is rejected", not validate(ORIGINALS, u"just a string", TERMS))
check("None is rejected", not validate(ORIGINALS, None, TERMS))

blank = list(good)
blank[0] = u"   "
check("a blank line is rejected", not validate(ORIGINALS, blank, TERMS))

nonstring = list(good)
nonstring[0] = 42
check("a non-string line is rejected", not validate(ORIGINALS, nonstring, TERMS))

runaway = list(good)
runaway[0] = u"x" * 400
check("a runaway line is rejected", not validate(ORIGINALS, runaway, TERMS))


print("\n=== all-or-nothing ===")

# One bad line rejects the whole batch. Mixing rewritten and template lines
# gives a message that changes voice halfway through, which reads worse than
# either version alone.
one_bad = list(good)
one_bad[2] = u"You're 95% ready already."
check("one bad line rejects the entire batch",
      not validate(ORIGINALS, one_bad, TERMS))


print("\n=== degenerate inputs ===")

check("no protected terms still validates", validate(ORIGINALS, good, []))
check("None protected terms still validates", validate(ORIGINALS, good, None))
check("empty originals with empty rewrite passes", validate([], [], []))


print("\n" + "=" * 62)
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All narration-validation checks passed.")
