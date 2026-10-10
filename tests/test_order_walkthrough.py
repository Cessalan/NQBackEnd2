# -*- coding: utf-8 -*-
"""
Order walkthrough validation: the tutor may explain the stored order, never
contradict it, and says nothing when it cannot defend it.

The fixtures are a real missed question from production (2026-10-10 read,
anonymous): PVCs on telemetry, which a student ordered Assess, EKG, oxygen,
notify.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_order_walkthrough.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.order_walkthrough import (  # noqa: E402
    LADDERS, RULES, build_prompt, cache_key, output_schema, validate_order_walkthrough,
)

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else f"  -> {detail}"))
    if not condition:
        failures.append(name)


CHART = ("0815: 67-year-old male on the telemetry unit for heart failure monitoring. This morning, the patient "
         "reported feeling sudden palpitations and mild dizziness. The telemetry monitor shows frequent premature "
         "ventricular complexes (PVCs) with an irregular rhythm. No chest pain or shortness of breath reported.")
QUESTION = ("Place the following nursing actions in the correct order of priority for this patient experiencing "
            "early (premature) ventricular beats detected on EKG.")
ITEMS = ["Assess the patient's level of consciousness and vital signs",
         "Administer supplemental oxygen as ordered",
         "Notify the healthcare provider with assessment findings",
         "Obtain a 12-lead EKG to document the rhythm"]
GOOD = {
    "key_supported": True,
    "ladder": "process",
    "breakdown": [
        {"role": "task", "quote": "in the correct order of priority", "meaning": "what comes first, not what is correct"},
        {"role": "problem", "quote": "frequent premature ventricular complexes (PVCs)", "meaning": "new irritable heartbeats"},
        {"role": "risk", "quote": "mild dizziness", "meaning": "his heart may not be pumping enough"},
    ],
    "first_thinking": ["Before you do anything for him, do you know he is stable?",
                       "Not yet. He is dizzy and his heart rate jumped.", "So assess first."],
    "items": [
        {"position": 1, "rung": 0, "rule": "assess_first", "label": "Is he stable?",
         "why_here": "Assessing comes first: every other step depends on what you find.", "nudge": "Do you know yet how he is doing?"},
        {"position": 2, "rung": 1, "rule": "treat_before_record", "label": "Eases the heart's work",
         "why_here": "Oxygen comes before the call: it eases the heart's work right now.", "nudge": "Is there something you can do for him first?"},
        {"position": 3, "rung": 2, "rule": "sequence", "label": "Gets orders",
         "why_here": "The call comes next, with your findings in hand.", "nudge": "What will you tell the provider if you have not acted yet?"},
        {"position": 4, "rung": 3, "rule": "treat_before_record", "label": "Records the rhythm",
         "why_here": "The 12-lead EKG records the rhythm, so it can wait until he is treated.", "nudge": "Does the EKG help him right now, or the chart?"},
    ],
    "method_line": "Climb the ladder: most urgent need first.",
}


def run(raw):
    return validate_order_walkthrough(raw, QUESTION, CHART, ITEMS)


def mutate(fn):
    raw = copy.deepcopy(GOOD)
    fn(raw)
    return run(raw)


ok = run(GOOD)
check("a good walkthrough survives", ok is not None)
check("items come back in key order", [i["position"] for i in ok["items"]] == [1, 2, 3, 4])
check("breakdown is shown in role order", [p["role"] for p in ok["breakdown"]] == ["problem", "risk", "task"])
check("ladder kept", ok["ladder"] == "process")

check("an undefended key shows nothing", mutate(lambda r: r.update(key_supported=False)) is None)
check("a missing key_supported shows nothing", mutate(lambda r: r.pop("key_supported")) is None)
check("an unknown ladder shows nothing", mutate(lambda r: r.update(ladder="vibes")) is None)
check("a rung off the ladder shows nothing", mutate(lambda r: r["items"][0].update(rung=9)) is None)


def climbs_back(r):
    # The EKG (position 4) placed on the assess rung: later but MORE urgent.
    r["items"][3]["rung"] = 0


check("a ladder that contradicts the key shows nothing", mutate(climbs_back) is None)
check("same rung for neighbours is allowed (a tie the rule explains)",
      mutate(lambda r: r["items"][2].update(rung=1)) is not None)
check("a missing item shows nothing", mutate(lambda r: r["items"].pop()) is None)
check("a duplicated position shows nothing", mutate(lambda r: r["items"][1].update(position=1)) is None)
check("an empty why_here shows nothing", mutate(lambda r: r["items"][1].update(why_here=" ")) is None)
check("an unknown rule falls back to sequence",
      mutate(lambda r: r["items"][1].update(rule="gut_feeling"))["items"][1]["rule"] == "sequence")

paraphrased = mutate(lambda r: r["breakdown"][2].update(quote="he feels dizzy"))
check("a paraphrased quote is dropped, not highlighted",
      paraphrased is not None and [p["role"] for p in paraphrased["breakdown"]] == ["problem", "task"])
check("no literal quote at all shows nothing",
      mutate(lambda r: [p.update(quote="not in the chart") for p in r["breakdown"]]) is None)
check("a quote may come from the question", any(p["role"] == "task" for p in ok["breakdown"]))

check("one thinking line is not enough to demonstrate",
      mutate(lambda r: r.update(first_thinking=["So assess first."])) is None)
check("filler thinking lines are stripped",
      mutate(lambda r: r.update(first_thinking=["Yes.", "So assess first."])) is None)

dashed = mutate(lambda r: r["items"][0].update(why_here="Assess first — everything depends on it."))
check("em dashes never reach the student", "—" not in dashed["items"][0]["why_here"])

check("cache key changes with the order",
      cache_key(QUESTION, CHART, ITEMS, "en") != cache_key(QUESTION, CHART, list(reversed(ITEMS)), "en"))
check("cache key changes with language", cache_key(QUESTION, CHART, ITEMS, "en") != cache_key(QUESTION, CHART, ITEMS, "fr"))

prompt = build_prompt(QUESTION, CHART, "Heart rate 78 then 106 irregular", ITEMS, "fr")
check("prompt numbers the key in order", "1. Assess the patient" in prompt and "4. Obtain a 12-lead EKG" in prompt)
check("prompt carries vitals as context", "Heart rate 78" in prompt)
check("prompt asks for French", "French" in prompt)
check("schema enums match the ladders and rules",
      output_schema()["properties"]["ladder"]["enum"] == list(LADDERS)
      and output_schema()["properties"]["items"]["items"]["properties"]["rule"]["enum"] == list(RULES))

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
