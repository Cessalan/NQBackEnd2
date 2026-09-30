# -*- coding: utf-8 -*-
"""
The orchestrator's per-quiz decision, replayed over the study's real sequences.

tests/test_practice_profile.py pins the rules; this pins how the chat path
applies them to one quiz's tool arguments: which topic, which formats, which
source. It calls NursingTutor._apply_practice_profile on a stub, so no model,
network or Firebase is involved.

    venv/Scripts/python.exe tests/test_practice_profile_orchestrator.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.orchestrator import NursingTutor  # noqa: E402
from services import practice_profile as pp  # noqa: E402

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def tutor(profile=None, uploads=False, recent=None):
    session = SimpleNamespace(practice_profile=profile or {}, vectorstore=object() if uploads else None,
                              documents=[{"filename": "Cardiac 2.pdf"}] if uploads else [], file_insights={})
    return SimpleNamespace(session=session, _analyzer_practice={}, _recent_user_messages=recent or [],
                           _practice_dirty=False)


def turn(t, message, router_args, analyzer=None, continuation=False):
    """One chat turn: merge the message into the profile, then decide the quiz."""
    t._analyzer_practice = analyzer or {}
    analyzer_changes = t._analyzer_practice if t._analyzer_practice.get("scope_action") == "change" else \
        {k: v for k, v in t._analyzer_practice.items() if k != "scope"}
    t.session.practice_profile, _ = pp.merge(t.session.practice_profile, pp.parse_changes(message),
                                             analyzer_changes=analyzer_changes)
    return NursingTutor._apply_practice_profile(t, dict(router_args), message, continuation)


print("\nThe subscriber who asked for no ordering questions (VgYahnSmDDpkxh9jUswm)")
t = tutor(uploads=True)
cardiac = "cardiac problems including heart failure, CAD, MI, pericarditis, EKG rhythms"
args = turn(t, "ATI and NCLEX style practice questions on " + cardiac,
            {"topic": cardiac, "question_types": ["mcq", "sata", "casestudy"]},
            analyzer={"scope_action": "change", "scope": cardiac})
check("first request sets the scope", t.session.practice_profile["scope"] == cardiac)
check("source recorded as uploads", t.session.practice_profile["source"]["kind"] == "uploads")

args = turn(t, "i dont want the in order questions , also you keep focussing so heavily of pharmacology stop ! make more practice questions please",
            {"topic": cardiac, "question_types": ["mcq", "sata"]},
            analyzer={"scope_action": "keep", "emphasis": "less pharmacology", "excluded_formats": ["casestudy"]})
check("the correction: no ordering", "casestudy" not in args["question_types"], args)
check("emphasis remembered", t.session.practice_profile["emphasis"] == "less pharmacology")

args = turn(t, "more questions", {"topic": "Cardiac Tamponade Management, Heart Failure Management", "question_types": None},
            continuation=True)
check("'more questions': no ordering (was reintroduced)", "casestudy" not in args["question_types"], args)
check("'more questions' continues the saved scope, not a narrowed topic", args["topic"] == cardiac, args["topic"])

args = turn(t, "Create a short targeted practice on: Heart Failure Management", {"topic": "Heart Failure Management", "question_types": ["mcq"]},
            analyzer={"scope_action": "focus"})
check("targeted button: topic is the focus", args["topic"] == "Heart Failure Management")
check("...and the chat scope is untouched", t.session.practice_profile["scope"] == cardiac)

args = turn(t, "make me 20 questions nclex style and ATI style i have an exam tommorow , act like a nursing professor",
            {"topic": cardiac, "question_types": ["mcq", "sata", "casestudy"]}, analyzer={"scope_action": "keep"})
check("next day, '20 NCLEX-style': still no ordering", "casestudy" not in args["question_types"], args)
check("...and 20 is now the saved total", t.session.practice_profile["requestedTotal"] == 20)

args = turn(t, "more", {"topic": "x", "question_types": None}, continuation=True)
check("a later 'more' asks for the saved 20", args.get("num_questions") == 20, args)
check("the router's user_prompt is her words, not the rewrite", args["user_prompt"] == "more")

args = turn(t, "ok you can include ordering questions again", {"topic": cardiac, "question_types": ["mcq", "sata", "casestudy"]},
            analyzer={"scope_action": "keep"})
check("she can turn ordering back on", "casestudy" in args["question_types"], args)


print("\nThe pasted mental-health guide (jt7L6sxXwR01LP2ge1l3)")
guide = ("here: **Exam I Study Guide - NURS 3900 (30 questions/30points)**\n\n"
         "**Physiologic effects of stress response (acute and chronic)**\n\n**General Adaptation Syndrome**\n"
         "- alarm, resistance, exhaustion\n**Ethical and legal issues in mental health**\n- involuntary admission\n"
         "**Therapeutic relationship**\n- orientation, working, termination\n**Antipsychotics**\n- EPS, NMS\n"
         + "- further detail on each topic in the guide for the exam\n" * 20)
t = tutor()
args = turn(t, guide, {"topic": "Exam I Study Guide - NURS 3900", "question_types": ["mcq", "sata"]},
            analyzer={"scope_action": "change", "scope": "Exam I study guide"})
profile = t.session.practice_profile
check("the paste becomes the source", profile["source"]["kind"] == "pasted" and "General Adaptation Syndrome" in profile["source"]["pastedText"])
check("its headings become the source topics", "Therapeutic relationship" in profile["sourceTopics"], profile["sourceTopics"])
check("the analyzer's scope wins over the router's title-only topic", profile["scope"] == "Exam I study guide", profile["scope"])

args = turn(t, "more", {"topic": "Exam I Study Guide - NURS 3900", "question_types": None}, continuation=True)
check("'more' keeps the pasted source", t.session.practice_profile["source"]["kind"] == "pasted")

t = tutor(recent=[{"id": "m1", "content": guide}, {"id": "m2", "content": "quiz me on this"}])
turn(t, "quiz me on this", {"topic": "study guide", "question_types": None}, analyzer={"scope_action": "change"})
check("'quiz me on this' after a paste finds the paste", t.session.practice_profile["source"]["pastedMessageId"] == "m1")



print("\nThe 50-question request (Ub1BTAhBhCFVyz2CgEXI)")
t = tutor(uploads=True)
turn(t, "generate me 50 questions based on the powerpoint, the questions should be multiple choice and select all that apply.",
     {"topic": "PowerPoint content from uploaded files", "question_types": ["mcq", "sata"]},
     analyzer={"scope_action": "change", "scope": "the uploaded PowerPoint"})
turn(t, "Create a short targeted practice on: Serotonin Syndrome Actions", {"topic": "Serotonin Syndrome Actions"},
     analyzer={"scope_action": "focus"})
args = turn(t, "generate me 50  questions based on the notes, the questions should be multiple choice and select all that apply. ",
            {"topic": "Serotonin Syndrome Actions", "question_types": ["mcq", "sata"]},
            analyzer={"scope_action": "change", "scope": "all of the uploaded notes"})
check("the button's topic did not become the scope", t.session.practice_profile["scope"] == "all of the uploaded notes",
      t.session.practice_profile["scope"])
check("50 is kept", t.session.practice_profile["requestedTotal"] == 50)
check("...and the quiz is generated on the notes, not the drill topic", args["topic"] == "all of the uploaded notes", args["topic"])

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("All checks passed.")
