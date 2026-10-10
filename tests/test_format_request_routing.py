# -*- coding: utf-8 -*-
"""
A short message that asks for a question format is routed straight to the
quiz tool (NursingTutor._is_format_request).

Why: "give me case study" matched no fast pattern, went to the routing model,
and came back as a written scenario with discussion questions instead of an
interactive case-study card (owner, 2026-10-10). Questions ABOUT a format, and
a student's own case study, must still get a written answer.

    venv/Scripts/python.exe tests/test_format_request_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.orchestrator import NursingTutor  # noqa: E402

failures = []

QUIZ = [
    "give me case study",
    "give me case studies on heart failure",
    "case study",
    "case studies please",
    "quiz me on prioritization",
    "prioritization",
    "who to see first questions",
    "SATA please",
    "give me 10 select all that apply on diabetes",
    "give me ngn questions",
    "more case studies",
    "donne-moi une étude de cas",
]
NOT_QUIZ = [
    "how do I answer case study questions?",
    "what is a case study",
    "explain prioritization",
    "no case studies please",
    "I have a case study assignment due tomorrow about a patient with COPD, can you help me understand it",
    "I have a case study assignment, can you make it easier to understand",
    "here is my case study, give me feedback on my answers",
    "priority nursing diagnoses for heart failure",
    "give me a study plan for my exam",
]

for text in QUIZ:
    ok = NursingTutor._is_format_request(text)
    print(("PASS " if ok else "FAIL ") + f"routes to quiz: {text!r}")
    if not ok:
        failures.append(text)
for text in NOT_QUIZ:
    ok = not NursingTutor._is_format_request(text)
    print(("PASS " if ok else "FAIL ") + f"stays a written answer: {text!r}")
    if not ok:
        failures.append(text)

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
