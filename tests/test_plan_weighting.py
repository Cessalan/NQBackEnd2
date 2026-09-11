# -*- coding: utf-8 -*-
"""
Plan-weighting tests: time sets the budget, the diagnostic sets the order.

These cover the rule that decides how long a study plan is and what order it
runs in — the number the whole activation thesis rests on. It is deterministic
Python precisely so it can be asserted here instead of being re-negotiated by
a model on every generation.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_plan_weighting.py

main.py builds LLM clients at import time and needs API keys, so we load the
weighting functions out of the source file rather than importing the module.
"""
import io
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── load just the weighting block, without importing main.py ────────────────
SRC = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()

start = SRC.index("PLAN_BUDGETS = {")
end = SRC.index("def _attach_status_and_exam_nodes")
block = SRC[start:end]

# _plan_archetype lives above the block and is the one thing it calls out to.
arch = re.search(
    r"^SPRINT_MAX_DAYS.*?^def _plan_archetype\(days_to_exam\):.*?(?=^def |\Z)",
    SRC,
    re.S | re.M,
)
assert arch, "could not locate _plan_archetype"

mod = types.ModuleType("weighting")
exec(compile(arch.group(0) + "\n\n" + block, "weighting", "exec"), mod.__dict__)

weight = mod._weight_path_by_diagnostic
summarize = mod._summarize_tiers
PLAN_BUDGETS = mod.PLAN_BUDGETS


# ── harness ─────────────────────────────────────────────────────────────────
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL  %s %s" % (name, detail))


def node(t, label):
    return {"id": "%s_%s" % (t, label[:4]), "type": t, "label": label, "tags": []}


TOPICS = ["Cardiac", "Renal", "Endocrine"]

FULL_UNIT = ["lesson", "quiz", "audio", "flashcard", "quiz"]


def generated_path(topics=TOPICS):
    """What the LLM hands us: a full unit per topic."""
    out = []
    for t in topics:
        for kind in FULL_UNIT:
            out.append(node(kind, "%s - %s" % (t, kind)))
    return out


def types_of(nodes):
    return [n["type"] for n in nodes]


def topics_in_order(nodes):
    seen = []
    for n in nodes:
        base = n["label"].split(" - ")[0]
        if base not in seen:
            seen.append(base)
    return seen


print("\n=== pass-through (no diagnostic) ===")

path = generated_path()
out = weight(path, None, TOPICS, 14)
check("absent diagnostic returns the path untouched", out is path)
check("empty diagnostic returns the path untouched", weight(path, {}, TOPICS, 14) is path)


print("\n=== ordering: worst first, solid last ===")

diag = {"Cardiac": 95, "Renal": 20, "Endocrine": 65}
out = weight(generated_path(), diag, TOPICS, 14)
order = topics_in_order(out)
check("weakest topic opens the plan", order[0] == "Renal", order)
check("solid topic is last", order[-1] == "Cardiac", order)
check("shaky sits in the middle", order[1] == "Endocrine", order)

# two gaps: the worse one opens
diag2 = {"Cardiac": 10, "Renal": 30, "Endocrine": 90}
order2 = topics_in_order(weight(generated_path(), diag2, TOPICS, 14))
check("of two gaps the weaker opens", order2[0] == "Cardiac", order2)


print("\n=== unit shapes per tier ===")

out = weight(generated_path(), {"Cardiac": 95, "Renal": 20, "Endocrine": 65}, TOPICS, 14)
by_topic = {}
for n in out:
    by_topic.setdefault(n["label"].split(" - ")[0], []).append(n["type"])

check("gap gets the full 5-node unit", by_topic["Renal"] == FULL_UNIT, by_topic.get("Renal"))
check("shaky gets lesson + quiz", by_topic["Endocrine"] == ["lesson", "quiz"], by_topic.get("Endocrine"))
check("solid gets a flashcard + quiz refresh",
      by_topic["Cardiac"] == ["flashcard", "quiz"], by_topic.get("Cardiac"))


print("\n=== nothing is ever dropped for being known ===")

out = weight(generated_path(), {"Cardiac": 100, "Renal": 100, "Endocrine": 100}, TOPICS, 14)
check("a student solid on everything still gets every topic",
      set(topics_in_order(out)) == set(TOPICS), topics_in_order(out))
check("...and still opens on a lesson (invariant 3)",
      types_of(out)[0] == "lesson", types_of(out)[:3])


print("\n=== invariant 3: the first node is always a lesson ===")

for label, d in [
    ("all gaps", {"Cardiac": 0, "Renal": 0, "Endocrine": 0}),
    ("all solid", {"Cardiac": 90, "Renal": 90, "Endocrine": 90}),
    ("mixed", {"Cardiac": 90, "Renal": 10, "Endocrine": 60}),
    ("unknown topics", {"Something Else": 50}),
]:
    first = types_of(weight(generated_path(), d, TOPICS, 14))[0]
    check("first node is a lesson (%s)" % label, first == "lesson", first)


print("\n=== budget from time to exam ===")

diag = {"Cardiac": 95, "Renal": 20, "Endocrine": 65}
for days, archetype in [(1, "sprint"), (5, "focus"), (30, "master")]:
    out = weight(generated_path(), diag, TOPICS, days)
    budget = PLAN_BUDGETS[archetype]
    check("%s plan (%d days) fits its %d-node budget" % (archetype, days, budget),
          len(out) <= budget, "got %d" % len(out))

sprint = weight(generated_path(), diag, TOPICS, 1)
master = weight(generated_path(), diag, TOPICS, 30)
check("a crammer gets a shorter plan than someone with weeks",
      len(sprint) < len(master), "%d vs %d" % (len(sprint), len(master)))

# Sprint collapses the tail rather than dropping it.
sprint_solid = [n for n in sprint if n.get("_tier") == "solid"]
check("sprint keeps a single merged review node", len(sprint_solid) == 1,
      "%d tail nodes" % len(sprint_solid))


print("\n=== budget pressure never truncates a gap unit ===")

many = ["T%d" % i for i in range(6)]
diag_many = dict((t, 5) for t in many)  # every topic a gap
out = weight(generated_path(many), diag_many, many, 1)
groups = {}
for n in out:
    groups.setdefault(n["label"].split(" - ")[0], []).append(n["type"])
partial = [t for t, kinds in groups.items()
           if kinds != FULL_UNIT and not t.startswith("T0")]
check("under a tight budget, whole units are dropped, never truncated",
      all(groups[t] == FULL_UNIT for t in groups), groups)
check("...and the budget is still respected", len(out) <= PLAN_BUDGETS["sprint"],
      "got %d" % len(out))


print("\n=== tier summary for the preview ===")

out = weight(generated_path(), {"Cardiac": 95, "Renal": 20, "Endocrine": 65}, TOPICS, 14)
tiers = summarize(out, TOPICS)
check("gap topic reported", tiers["gap"] == ["Renal"], tiers)
check("solid topic reported for the tail copy", tiers["solid"] == ["Cardiac"], tiers)
check("shaky topic reported", tiers["shaky"] == ["Endocrine"], tiers)


print("\n=== unmatched diagnostic topics degrade to 'untested' ===")

out = weight(generated_path(), {"Totally Unrelated": 90}, TOPICS, 14)
tiers = summarize(out, TOPICS)
check("topics with no diagnostic score are treated as untested, not solid",
      len(tiers["solid"]) == 0, tiers)
check("...and are still taught", len(out) > 0)


print("\n=== fuzzy topic matching ===")

out = weight(generated_path(), {"cardiac  ": 95}, TOPICS, 14)
tiers = summarize(out, TOPICS)
check("case and whitespace differences still match", tiers["solid"] == ["Cardiac"], tiers)

out = weight(generated_path(), {"Cardiac Pharmacology": 95}, TOPICS, 14)
tiers = summarize(out, TOPICS)
check("a longer diagnostic label still matches its topic",
      tiers["solid"] == ["Cardiac"], tiers)


print("\n=== the generator mislabels its nodes (regression) ===")

# What production hands us 17.6% of the time: a path whose node labels name
# nothing in the curriculum. Real case, chat 17Z4fNVsl3NbuPjNoWrb — the student
# was quizzed on "Caring in Nursing" and every node came back as an objective.
OBJECTIVE_PATH = [
    node(kind, obj)
    for obj in [
        "Describe nursing interventions to promote patient safety.",
        "Identify measures to promote health across the lifespan.",
    ]
    for kind in FULL_UNIT
]

out = weight(OBJECTIVE_PATH, {"Cardiac": 0}, TOPICS, 14)
tiers = summarize(out, TOPICS)
check("a mislabelled path still tiers the tested topic as a gap",
      tiers["gap"] == ["Cardiac"], tiers)
check("...and the plan opens on it, not on whatever the model listed first",
      topics_in_order(out)[0] == "Cardiac", topics_in_order(out)[:3])
check("...and it still opens on a lesson",
      types_of(out)[0] == "lesson", types_of(out)[:3])

out = weight(OBJECTIVE_PATH, {"Renal": 10}, TOPICS, 20)
check("every curriculum topic is represented",
      set(TOPICS).issubset(set(topics_in_order(out))), topics_in_order(out))
check("the weak topic leads, not the off-curriculum content",
      topics_in_order(out)[0] == "Renal", topics_in_order(out)[:4])


print("\n=== a correctly labelled path is unaffected ===")

out = weight(generated_path(), {"Cardiac": 0, "Renal": 55, "Endocrine": 95}, TOPICS, 14)
check("gap first, solid last, as before",
      topics_in_order(out)[0] == "Cardiac" and topics_in_order(out)[-1] == "Endocrine",
      topics_in_order(out))
check("generated nodes are reused, not replaced by synthesised ones",
      any(not n["id"].startswith("synth_") for n in out),
      [n["id"] for n in out[:4]])


print("\n=== the curriculum keeps the diagnostic's vocabulary ===")

# /study/plan re-extracts topics with a second LLM call. If that rename loses
# the topic the student was just quizzed on, her score tiers as `untested` and
# never reaches the front of the plan.
restore = mod._restore_diagnostic_topics

check("no diagnostic leaves the list alone",
      restore(["A", "B"], None) == ["A", "B"])
check("an empty diagnostic leaves the list alone",
      restore(["A", "B"], {}) == ["A", "B"])
check("a matched key is not duplicated",
      restore(["Cardiac Care", "Renal"], {"Cardiac": 30}) == ["Cardiac Care", "Renal"])
check("a renamed-away topic is restored",
      "Caring in Nursing" in restore(
          ["Describe nursing interventions to promote patient safety."],
          {"Caring in Nursing": 20}))
check("the input list is not mutated",
      (lambda base: (restore(base, {"X": 1}), base == ["A"])[1])(["A"]))
check("the curriculum stays bounded",
      len(restore(["T%d" % i for i in range(6)],
                  {"New A": 1, "New B": 2})) == mod.MAX_CURRICULUM_TOPICS)


print("\n=== course priorities and diagnostic handoff ===")
course_order = ["Cardiac", "Renal", "Endocrine"]
calibrated = weight(generated_path(), {"Cardiac": 100, "Renal": 0}, course_order, 14)
recommendation = mod._actual_recommended_start(calibrated, course_order, {"Cardiac": 100, "Renal": 0})
check("the recommendation names the calibrated first topic", recommendation["topic"] == "Renal")
check("the recommendation attributes the diagnostic", recommendation["basis"] == "diagnostic")
check("an empty path makes no recommendation", mod._actual_recommended_start([], course_order) is None)
tied = weight(generated_path(), {"Renal": 0, "Cardiac": 0}, course_order, 14)
check("course priority breaks equal diagnostic results", topics_in_order(tied)[0] == "Cardiac")

print("\n" + "=" * 62)
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All plan-weighting checks passed.")
