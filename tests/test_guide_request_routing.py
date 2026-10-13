# -*- coding: utf-8 -*-
"""
A request for a study guide is never answered with a quiz
(NursingTutor._is_guide_request + the GUIDE GUARD in process_message).

Why: chat XbPrJdZgXEcRmlGNhgfY said "prep guide" and then "complete the prep
guide" and got two quizzes full of case studies (owner, 2026-10-11). Only
"study guide" was a known phrase, and the intent analyzer, seeing quizzes just
before, read the message as "practise from the Prep Guide". A message that
names questions as well as a guide is still the analyzer's to decide.

    venv/Scripts/python.exe tests/test_guide_request_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.orchestrator import NursingTutor  # noqa: E402

failures = []

GUIDE = [
    "prep guide",
    "complete the prep guide",
    "create study guide for mobility fundamental of nursing",
    "make me a study guide",
    "exam guide for cardio",
    "review guide on fall prevention",
    "study sheet please",
    "fais-moi un guide d'étude",
]
NOT_GUIDE = [
    "quiz me on the study guide",
    "give me 10 questions from my prep guide",
    "case studies from the study guide",
    "flashcards from the prep guide",
    "give me a study plan for my exam",
    "more",
    "explain passive range of motion",
]

for text in GUIDE:
    ok = NursingTutor._is_guide_request(text)
    print(("PASS " if ok else "FAIL ") + f"routes to study sheet: {text!r}")
    if not ok:
        failures.append(text)
for text in NOT_GUIDE:
    ok = not NursingTutor._is_guide_request(text)
    print(("PASS " if ok else "FAIL ") + f"left to the analyzer: {text!r}")
    if not ok:
        failures.append(text)

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
