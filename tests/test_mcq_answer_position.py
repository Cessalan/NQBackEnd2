# -*- coding: utf-8 -*-
"""
MCQ answer position — the key must follow the content, never the dice roll.

The generator used to tell the model which letter had to be correct. With
options in a natural order that let it re-key a question to fit: a Pro student
(2026-09-22) got the same perineal-care question nine times, identical options,
keyed B four times and A five times. Now the model writes the answer as A and
tools/mcq_answer_position.py moves the option. These checks pin that moving an
option never changes WHICH text is correct.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_mcq_answer_position.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mcq_answer_position import place_correct_option  # noqa: E402

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


RETRACT = "Retract the foreskin by gently pushing it towards the body."


def perineal_question():
    """Her question, written the way the model is now asked to: key at A."""
    return {
        "question": "A nurse is teaching a patient with an uncircumcised penis about perineal hygiene. Which instruction should the nurse include?",
        "options": [
            "A) " + RETRACT,
            "B) Start at the urinary meatus and wash the glans with soap.",
            "C) Use a circular motion and wash away from the urinary meatus.",
            "D) Wash the shaft of the penis using upward strokes.",
        ],
        "answer": "A) " + RETRACT,
        "metadata": {"correctAnswerIndex": 0},
    }


def keyed_text(q):
    index = ord(q["answer"][0]) - ord("A")
    return q["options"][index][3:]


print("\nplace_correct_option")

for letter in "ABCD":
    q = place_correct_option(perineal_question(), letter, rng=random.Random(1))
    check(f"target {letter}: correct option lands on {letter}", q["answer"].startswith(letter + ")"), q["answer"])
    check(f"target {letter}: the keyed text is still the retraction step", keyed_text(q) == RETRACT, keyed_text(q))
    check(f"target {letter}: metadata index agrees", q["metadata"]["correctAnswerIndex"] == "ABCD".index(letter))

rng = random.Random(7)
seen_positions, texts_ok = set(), True
for _ in range(200):
    q = place_correct_option(perineal_question(), None, rng=rng)
    seen_positions.add(q["answer"][0])
    texts_ok = texts_ok and keyed_text(q) == RETRACT and sorted(o[3:] for o in q["options"]) == sorted(o[3:] for o in perineal_question()["options"])
check("no target: every position is used", seen_positions == set("ABCD"), seen_positions)
check("no target: 200 reruns never change the correct text or lose an option", texts_ok)

q = perineal_question()
q["answer"] = "C) Use a circular motion and wash away from the urinary meatus."
q = place_correct_option(q, "A", rng=random.Random(3))
check("honours the model's key even when it ignored the A instruction",
      keyed_text(q) == "Use a circular motion and wash away from the urinary meatus.", keyed_text(q))

q = perineal_question()
q["answer"] = RETRACT  # text only, no letter
q = place_correct_option(q, "D", rng=random.Random(3))
check("resolves a letterless answer by its text", q["answer"] == "D) " + RETRACT, q["answer"])

odd = {"options": ["A) one", "B) two", "C) three"], "answer": "A) one"}
check("leaves a non-four-option question untouched", place_correct_option(dict(odd), "C") == odd)

unknown = perineal_question()
unknown["answer"] = "Something that is not an option"
before = [o for o in unknown["options"]]
place_correct_option(unknown, "B")
check("leaves an unresolvable key untouched rather than guess", unknown["options"] == before)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("All checks passed.")
