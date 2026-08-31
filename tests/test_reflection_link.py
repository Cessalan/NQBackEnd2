# -*- coding: utf-8 -*-
"""
Reflection-note link tests: when may a lesson claim it covered her weak spot?

`_matched_struggles` decides whether the note on an unscored node (lesson,
audio, concept map) is allowed to say "this covered the thing you keep getting
wrong". A false positive there is the worst sentence this product can produce:
it asserts the tutor watched her, about a lesson that had nothing to do with
her misses, and once a student catches that she discounts every other claim we
make. So the bar is a significant word visible in both places, and the correct
answer when nothing matches is an empty list — no link, no claim.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_reflection_link.py

main.py builds LLM clients at import time and needs API keys, so we load the
matching block out of the source file rather than importing the module.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── load just the matching block, without importing main.py ─────────────────
SRC = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()

start = SRC.index("_LINK_STOPWORDS = {")
end = SRC.index("async def _build_reflection_note")
block = SRC[start:end]

ns = {"re": re}
exec(compile(block, "main.py:link", "exec"), ns)  # noqa: S102
matched = ns["_matched_struggles"]
tokens = ns["_link_tokens"]

# ── and the note length guard, which lives just above ───────────────────────
tstart = SRC.index("STUDY_NOTE_MAX_CHARS = ")
tend = SRC.index("async def _build_study_note")
tns = {"re": re}
exec(compile(SRC[tstart:tend], "main.py:trim", "exec"), tns)  # noqa: S102
trim = tns["_trim_note"]

failures = []


def check(name, got, want):
    if got != want:
        failures.append("%s\n    expected: %r\n    got:      %r" % (name, want, got))


# ── the link that should fire ───────────────────────────────────────────────
# She has been missing afterload. The lesson is about afterload. This is the
# entire reason the feature exists.
check(
    "names the struggle the lesson actually covers",
    matched(
        "Preload and Afterload",
        ["Afterload rises with vasoconstriction", "Preload is venous return"],
        ["preload vs afterload", "insulin onset times"],
    ),
    ["preload vs afterload"],
)

check(
    "matches on the covered key points, not only the topic label",
    matched(
        "Cardiac Review",
        ["Digoxin toxicity presents with visual changes"],
        ["digoxin toxicity signs"],
    ),
    ["digoxin toxicity signs"],
)

# ── the links that must NOT fire ────────────────────────────────────────────
check(
    "unrelated struggle produces no link",
    matched(
        "Insulin Types",
        ["Rapid-acting peaks at one hour"],
        ["preload vs afterload"],
    ),
    [],
)

# The single most dangerous false positive: every nursing topic shares these
# words, so matching on them would claim a link on essentially every node.
check(
    "shared filler words are not evidence of a link",
    matched(
        "Nursing Care of the Patient",
        ["Clinical assessment basics"],
        ["patient positioning after surgery"],
    ),
    [],
)

check(
    "short words are ignored",
    # "ph" and "abg" are under the 4-char floor; nothing else overlaps.
    matched("ABG and pH", ["pH ranges"], ["abg interpretation"]),
    [],
)

check(
    "no struggles at all is an empty link, never an error",
    matched("Afterload", ["Afterload"], []),
    [],
)

check(
    "an empty node label cannot match anything",
    matched("", [], ["preload vs afterload"]),
    [],
)

# ── shape guarantees the caller depends on ──────────────────────────────────
check(
    "at most two labels, so the evidence row stays readable",
    matched(
        "Cardiac",
        ["cardiac output", "cardiac cycle"],
        ["cardiac output basics", "cardiac cycle timing", "cardiac enzymes"],
    ),
    ["cardiac output basics", "cardiac cycle timing"],
)

check(
    "matching is case-insensitive",
    matched("AFTERLOAD", [], ["Afterload vs Preload"]),
    ["Afterload vs Preload"],
)

check(
    "stopwords are excluded from the token set",
    tokens("Nursing care of the patient") & {"nursing", "care", "patient"},
    set(),
)


# ── note length ─────────────────────────────────────────────────────────────
# The card is read in about four seconds. The prompt asks for one or two
# sentences and the model mostly obliges, but "mostly" is not a limit — and
# the third sentence is reliably the one that restates the first.
check(
    "a third sentence is dropped",
    trim("You mixed up jaw-thrust and head-tilt. It's jaw-thrust in trauma. Keep practising!"),
    "You mixed up jaw-thrust and head-tilt. It's jaw-thrust in trauma.",
)

# The scored screen's takeaway renders as ONE line under a "Focus on" label,
# so it asks for a budget of one and gets it.
check(
    "the takeaway budget of one is honoured",
    trim("Why high-flow oxygen is used. It clears CO2.", max_sentences=1),
    "Why high-flow oxygen is used.",
)

check("two sentences pass through untouched",
      trim("You missed the first step. It's the jaw-thrust."),
      "You missed the first step. It's the jaw-thrust.")

check("one sentence passes through untouched",
      trim("It's jaw-thrust, not head-tilt."),
      "It's jaw-thrust, not head-tilt.")

check("a sentence with no terminator is left alone rather than emptied",
      trim("It's jaw-thrust, not head-tilt"),
      "It's jaw-thrust, not head-tilt")

check("question and exclamation marks end a sentence too",
      trim("Why jaw-thrust? Because the neck may be hurt. Remember that."),
      "Why jaw-thrust? Because the neck may be hurt.")

check("empty input survives", trim(""), "")
check("None survives", trim(None), "")


# ── report ──────────────────────────────────────────────────────────────────
print("")
if failures:
    print("FAILED (%d)" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)

print("reflection link: all checks passed")
