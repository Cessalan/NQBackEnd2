import unittest
# -*- coding: utf-8 -*-
"""
Practice profile — what a student asked for must survive the next batch.

Replays the real messages from the 2026-09-29 session study:

  * VgYahnSmDDpkxh9jUswm (Pro): "i dont want the in order questions" was obeyed
    once, then "more questions" and "20 NCLEX-style" brought ordering back.
  * jt7L6sxXwR01LP2ge1l3: a pasted mental-health guide became a quiz on
    potassium and heparin, because only the guide's title reached generation.
  * Ub1BTAhBhCFVyz2CgEXI: "50 questions based on the notes" inherited the topic
    of the targeted-practice button pressed just before.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_practice_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.practice_profile import (  # noqa: E402
    effective_formats, focus_topic, looks_like_pasted_material, merge, normalize,
    outline_topics, parse_changes, seed_from_quiz_settings,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


print("\nParsing what a message changes")

c = parse_changes("i dont want the in order questions , also you keep focussing so heavily of pharmacology stop ! make more practice questions please")
check("'dont want the in order questions' excludes the ordering format", c.get("exclude") == ["casestudy"], c)
check("...and adds nothing", "include" not in c, c)

c = parse_changes("stop giving questions where i have to place things in order i dont like those , make new questions ")
check("'place things in order' after 'stop' excludes ordering", c.get("exclude") == ["casestudy"], c)

c = parse_changes("no ordering questions")
check("'no ordering questions' EXCLUDES ordering (old check added it)", c.get("exclude") == ["casestudy"] and "include" not in c, c)

check("'more questions' changes nothing", parse_changes("more questions") == {})
check("'quiz me nclex and ati tyle' changes nothing", parse_changes("quiz me nclex and ati tyle") == {})

c = parse_changes("make me 20 questions nclex style and ATI style i have an exam tommorow , act like a nursing professor")
check("'make me 20 questions nclex style' sets total 20 and no format", c == {"requested_total": 20}, c)

c = parse_changes("Give me 40 questions from these slides, MCQ and SATA only, no ordering questions.")
check("the intended request: total 40", c.get("requested_total") == 40, c)
check("...MCQ + SATA as the complete list", c.get("only") == ["mcq", "sata"], c)
check("...ordering excluded", c.get("exclude") == ["casestudy"], c)

c = parse_changes("you can include ordering questions again")
check("explicit re-enable includes ordering", c.get("include") == ["casestudy"] and "exclude" not in c, c)

c = parse_changes("please make me NCLEX and ATI style , including case studies , NGN-CLEX on cardiac problems")
check("'including case studies' is an explicit request", "casestudy" in c.get("include", []), c)

check("'my exam has 50 questions' is not a requested total", "requested_total" not in parse_changes("my exam has 50 questions on cardiac"))
check("'generate me 50  questions' is", parse_changes("generate me 50  questions based on the notes").get("requested_total") == 50)
check("'make it 40' is", parse_changes("make it 40").get("requested_total") == 40)
check("'10 more' is", parse_changes("10 more please").get("requested_total") == 10)

check("'harder questions' sets hard", parse_changes("give me harder questions").get("difficulty") == "hard")
check("'I find SATA difficult' does not set difficulty", "difficulty" not in parse_changes("I find SATA difficult"))
check("'not too hard questions' does not set hard", "difficulty" not in parse_changes("not too hard questions please"))

guide = "**Exam I Study Guide - NURS 3900**\n" + "\n".join(
    f"**Topic {i}**\n- select all that apply to the stress response in patient {i} and more notes here" for i in range(30))
check("format words inside pasted notes are not instructions", parse_changes(guide) == {}, parse_changes(guide))


print("\nMerging into the saved profile")

profile, changed = merge({}, parse_changes("i dont want the in order questions"))
check("exclusion is saved", profile["excludedFormats"] == ["casestudy"] and "excludedFormats" in changed, profile)

for message in ["more questions", "more", "Create a short targeted practice on: Heart Failure Management",
                "make me 20 questions nclex style and ATI style i have an exam tommorow", "continue"]:
    profile, _ = merge(profile, parse_changes(message))
check("exclusion survives more / button / NCLEX-style / continue", profile["excludedFormats"] == ["casestudy"], profile)
check("...and the total from 'make me 20' was kept", profile["requestedTotal"] == 20, profile)

for guess in (None, ["mcq", "sata", "casestudy"], ["casestudy"]):
    types = effective_formats(profile, guess, "more questions")
    check(f"no ordering in a batch when the model guessed {guess}", "casestudy" not in types and types, types)

check("next-day resume: same profile, same answer", "casestudy" not in effective_formats(normalize(profile), None, "quiz me nclex and ati tyle"))

profile, _ = merge(profile, {}, analyzer_changes={"excluded_formats": [], "scope": None})
check("an analyzer that hears nothing changes nothing", profile["excludedFormats"] == ["casestudy"])

profile2, _ = merge(profile, parse_changes("you can include ordering questions again"))
check("student re-enables ordering explicitly", profile2["excludedFormats"] == [] and "casestudy" in effective_formats(profile2, ["mcq", "casestudy"]))
check("with nothing said and no guess, ordering is not a default", effective_formats({}) == ["mcq", "sata"])

check("the analyzer cannot re-enable a format", merge(profile, {}, analyzer_changes={"excluded_formats": ["sata"]})[0]["excludedFormats"] == ["casestudy", "sata"])

profile3, _ = merge({}, parse_changes("Give me 40 questions from these slides, MCQ and SATA only, no ordering questions."))
check("'only' saves an allow-list", profile3["formats"] == ["mcq", "sata"] and profile3["requestedTotal"] == 40, profile3)
check("allow-list beats the model's guess", effective_formats(profile3, ["casestudy", "mcq"]) == ["mcq", "sata"])
check("a one-off 'give me SATA' narrows nothing it wasn't asked to", effective_formats(profile3, None, "give me SATA") == ["mcq", "sata"])

all_off, _ = merge({}, {"exclude": ["mcq", "sata", "casestudy", "true_false", "matrix", "unfoldingcase"]})
check("excluding everything still yields a format", effective_formats(all_off) == ["mcq"])

_, changed = merge(profile, {})
check("no changes reports no changed fields", changed == [])

profile4, _ = merge({}, {}, analyzer_changes={"scope": "wound care", "emphasis": "less pharmacology"})
check("analyzer sets scope and emphasis", profile4["scope"] == "wound care" and profile4["emphasis"] == "less pharmacology")


print("\nScope, sources and seeding")

check("targeted-practice button is a focus", focus_topic("Create a short targeted practice on: Serotonin Syndrome Actions") == "Serotonin Syndrome Actions")
check("French button too", focus_topic("Crée une courte pratique ciblée sur : Insuffisance cardiaque") == "Insuffisance cardiaque")
check("a normal request is not a focus", focus_topic("generate me 50 questions based on the notes") is None)

c220 = """**Exam I Study Guide - NURS 3900 (30 questions/30points)**

**Physiologic effects of stress response (acute and chronic)**

**General Adaptation Syndrome**
- alarm, resistance, exhaustion

**Ethical and legal issues in mental health**
- involuntary admission, least restrictive environment

**Therapeutic relationship**
- phases: orientation, working, termination

**Groups**
**Antipsychotics**
- EPS, NMS, tardive dyskinesia
**Behavioural therapies**
""" + ("- more detail about each area of the guide for the exam.\n" * 20)
check("the pasted guide counts as material", looks_like_pasted_material(c220))
check("a request does not", not looks_like_pasted_material("give me 40 questions from these slides"))
topics = outline_topics(c220)
for expected in ["General Adaptation Syndrome", "Therapeutic relationship", "Antipsychotics", "Behavioural therapies"]:
    check(f"outline finds '{expected}'", expected in topics, topics)
check("outline skips the exam header with its point count", not any("30 questions" in t for t in topics), topics)
check("outline never returns bullet text", not any(t.startswith("alarm") for t in topics), topics)

seeded = seed_from_quiz_settings({"requested_total": 50, "question_types": ["mcq", "sata", "casestudy"], "difficulty": "hard"})
check("seed keeps total and difficulty", seeded["requestedTotal"] == 50 and seeded["difficulty"] == "hard")
check("seed does NOT freeze an old guessed format list", seeded["formats"] is None and seeded["excludedFormats"] == [])

check("normalize drops junk", normalize({"formats": ["ordering", "weird"], "requestedTotal": True})["formats"] == ["casestudy"])

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("All checks passed.")


class SourceTopicGroupTests(unittest.TestCase):
    def test_groups_survive_normalize_and_drop_bad_rows(self):
        from services import practice_profile as pp
        profile = pp.normalize({'sourceTopicGroups': [
            {'title': 'Examen primaire', 'subtopics': ['C — Circulation', '', 'A — Airway']},
            {'title': '', 'subtopics': ['orphan']}, 'not a dict']})
        self.assertEqual(profile['sourceTopicGroups'],
                         [{'title': 'Examen primaire', 'subtopics': ['C — Circulation', 'A — Airway']}])
        self.assertEqual(pp.normalize({})['sourceTopicGroups'], [])
