# -*- coding: utf-8 -*-
"""
Ordering questions must order nursing ACTIONS (tools/casestudy_prompts.py).

The failing case is real (2026-10-10): a nephritic-syndrome question whose
items were explanations ("Recognize that...", "Explain that the findings
suggest..."), so its key was arbitrary and the "Who first?" walkthrough
refused to defend it.

    venv/Scripts/python.exe tests/test_casestudy_items.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.casestudy_prompts import ordering_item_problems  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else f"  -> {detail}"))
    if not condition:
        failures.append(name)


q = lambda *texts: {"options": [{"id": f"item{i}", "text": t} for i, t in enumerate(texts)]}

nephritic = q(
    "Assess urine output and check for worsening fluid retention or respiratory symptoms.",
    "Recognize that glomerular inflammation can damage the filtration barrier, allowing red blood cells into urine.",
    "Explain that the findings suggest a nephritic pattern, not the heavy protein-loss pattern of nephrotic syndrome.",
    "Explain that the elevated serum creatinine indicates reduced filtration function, not a syndrome by itself.",
)
check("the real nephritic question is caught on its 'Recognize that' item", ordering_item_problems(nephritic) == [
    nephritic["options"][1]["text"]], ordering_item_problems(nephritic))
check("a live teaching step ('Explain that bladder training uses...') passes", ordering_item_problems(q(
    "Explain that bladder training uses planned voiding and gradual interval increases")) == [])

real_actions = q(
    "Explain the procedure and ask the patient to remain still and relax arms and legs",
    "Assess the patient's level of consciousness and vital signs",
    "Administer supplemental oxygen as ordered",
    "Notify the healthcare provider with assessment findings",
    "Use teach-back technique to confirm Mr. Lee understands the instructions",
    "Acknowledge the patient's emotions and offer reassurance of support",
)
check("real nursing actions pass, including 'explain the procedure' and teach-back",
      ordering_item_problems(real_actions) == [], ordering_item_problems(real_actions))

check("'explain to the patient that' is patient teaching, an action", ordering_item_problems(q(
    "Explain to the patient that the rash is expected")) == [])
check("French knowledge statements are caught", ordering_item_problems(q(
    "Reconnaître que l'inflammation glomérulaire endommage la barrière")) != [])
check("plain strings and missing options are handled", ordering_item_problems({"options": ["Recognize that X"]}) != []
      and ordering_item_problems({}) == [] and ordering_item_problems(None) == [])

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
