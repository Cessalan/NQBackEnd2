# -*- coding: utf-8 -*-
"""
Student emphasis: what she told us is on the exam must reach her plan.

Built from the real pastes found 2026-10-01 in the 900 most recent chats:

  * XenYEsZyTNUMD0udItjP (Pro, exam that day): "MUST know the bone & muscle
    landmarks. This will be on the exam." and "Landmarks (MUST know)" beside
    near-misses like "MUST be administered again" and "Providers may NOT know".
  * kKm2qh9cxaQgCfkSbCGK: "Here are the drugs that will be on the exam."
  * VvErW3Hwa7ZYwBDxF8UB: a study guide of "**Know the components of ...**" lines.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_student_emphasis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.student_emphasis import (  # noqa: E402
    PASTED_SHARE_CHARS, apply_to_topics, emphasis_direction, exam_cues, extraction_material,
    fallback_topics, gather, is_empty, prompt_block,
)

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else f"  -> {detail}"))
    if not condition:
        failures.append(name)


# ── Cues: the real flags, not the near-misses ───────────────────────────────
injections = (
    "Injections\n"
    "Providers may NOT know that you are a student if you are in isolation garb.\n"
    "If NO wheal forms, the injection was too deep & MUST be administered again (TB test).\n"
    "Be sure it's the right drug for route ordered.\n"
    "MUST know the bone & muscle landmarks. This will be on the exam.\n"
    "Ventrogluteal\n"
    "Landmarks (MUST know)\n"
)
quotes = [c["quote"] for c in exam_cues(injections)]
check("finds 'MUST know the bone & muscle landmarks'",
      any("bone & muscle landmarks" in q for q in quotes), quotes)
check("finds 'Landmarks (MUST know)'", any(q.startswith("Landmarks") for q in quotes), quotes)
check("'This will be on the exam.' carries its subject from the line before",
      any("landmarks" in q.lower() and "on the exam" in q.lower() for q in quotes), quotes)
check("ignores 'MUST be administered again'", not any("administered" in q for q in quotes), quotes)
check("ignores 'Providers may NOT know'", not any("Providers" in q for q in quotes), quotes)
check("ignores 'Be sure it's the right drug'", not any("right drug" in q for q in quotes), quotes)

drugs = "can you help me study for my pharmacology exam tomorrow. Here are the drugs that will be on the exam. furosemide, digoxin"
check("keeps 'drugs that will be on the exam' for the prompt",
      any("will be on the exam" in c["quote"] for c in exam_cues(drugs)))

guide = ("**Know the components of the health history and where to document each**\n"
         "what are expected and unexpected findings?\n"
         "know the types of thermometers and when they might be used\n"
         "be able to describe nurse self-care and why it is important.\n"
         "The patient's knowledge of their medications was expected to be good.\n")
gq = [c["quote"] for c in exam_cues(guide)]
check("study-guide 'Know the ...' lines are cues", any("health history" in q for q in gq), gq)
check("'be able to' objective lines are cues", any("self-care" in q for q in gq), gq)
check("a question line without a cue is not", not any("expected and unexpected" in q for q in gq), gq)
check("'knowledge' mid-sentence is not an objective line", not any("knowledge" in q for q in gq), gq)

many = "\n".join([f"Know the stage {i} details" for i in range(12)] + ["Sepsis bundles are high-yield."])
mq = exam_cues(many)
check("cues are capped", len(mq) <= 8, len(mq))
check("an explicit flag outranks routine objective lines under the cap",
      any("high-yield" in c["quote"] for c in mq), [c["quote"] for c in mq])

# The first dry run on production missed XenYEsZyTNUMD0udItjP's flag: her
# paste was 27k characters and the cap was applied before the scan.
huge = ("Medication administration notes, routine detail line.\n" * 400) + injections
check("a flag past the 12k mark of a long paste is still found",
      any("landmarks" in c["quote"].lower() for c in gather([huge], {})["cues"]))
check("the pasted text the prompts read stays capped", len(gather([huge], {})["pasted_text"]) <= 12000)

runon = ("Antibodies attach to specific antigens and the immune system responds " * 6
         + "You MUST know the complement cascade steps for the test " + "and more filler text follows here " * 6)
rq = exam_cues(runon)
check("a flag inside a huge run-on line is quoted around the flag",
      rq and "complement cascade" in rq[0]["quote"] and len(rq[0]["quote"]) <= 200, rq)
for prose in ["This effect is usually transient, but the patient and partner need to know that it is normal.",
              "What does the provider need to know about the allergy?",
              "The nurse will be tested for HIV at regular intervals."]:
    check(f"textbook prose is not a flag: {prose[:40]}", exam_cues(prose) == [], exam_cues(prose))
check("a student's own 'we need to know' still is",
      exam_cues("we need to know the causes, signs and symptoms of each disorder"))
joined_q = [c["quote"] for c in exam_cues("MUST know the bone & muscle landmarks. This will be on the exam.")]
check("a flag and its bare follow-up become one quote, not two", len(joined_q) == 1, joined_q)
check("explicit flags from any source sort before objective lines",
      gather(["Know the stages of labor in detail please\n" * 1, "hello"], {}, "Sepsis is high-yield.")["cues"][0]["strength"] == "explicit")

# ── Emphasis direction: 'less' must never promote ───────────────────────────
check("'less pharmacology' is a down-steer", emphasis_direction("less pharmacology") == ("less", "pharmacology"))
check("'more EKG rhythms' is an up-steer", emphasis_direction("more EKG rhythms") == ("more", "EKG rhythms"))
check("'stop focusing on pharm' is a down-steer", (emphasis_direction("stop focusing on pharmacology") or ("",))[0] == "less")
check("an empty steer is nothing", emphasis_direction("") is None and emphasis_direction("more") is None)

# ── Applying to topics ──────────────────────────────────────────────────────
topics = ["Medication Administration Rights", "Intradermal Testing", "Bone and Muscle Landmarks", "Insulin Types", "Wound Care", "Isolation Precautions"]
signals = gather([injections], {})
ordered, flags = apply_to_topics(topics, signals, limit=5)
check("a flagged topic ranked sixth... or third is promoted to first", ordered[0] == "Bone and Muscle Landmarks", ordered)
check("the cut still keeps five", len(ordered) == 5, ordered)
check("the flag carries her quote as verified evidence",
      flags and flags[0]["confidence"] == "verified" and "landmarks" in flags[0]["quote"].lower(), flags)

ordered_none, flags_none = apply_to_topics(topics, gather([], {}), limit=5)
check("no signals: exactly the old first five, untouched", ordered_none == topics[:5] and flags_none == [], ordered_none)
check("no signals: no limit means the whole list back", apply_to_topics(topics, gather())[0] == topics)

late = ["Wound Care", "Insulin Types", "Fluid Balance", "Delegation", "Pain", "Bone and Muscle Landmarks"]
check("a flagged topic past the five-topic cut is rescued, not lost",
      apply_to_topics(late, signals, limit=5)[0][0] == "Bone and Muscle Landmarks")

pharm = ["Pharmacology of Cardiac Drugs", "Heart Failure", "Fluid Balance"]
less = apply_to_topics(pharm, gather([], {"emphasis": "less pharmacology"}))[0]
check("'less pharmacology' moves pharmacology to the back, never the front", less[-1] == "Pharmacology of Cardiac Drugs", less)
more = apply_to_topics(["Fluid Balance", "EKG Interpretation"], gather([], {"emphasis": "more EKG rhythms"}))
check("'more EKG rhythms' promotes EKG with chat_emphasis as its source",
      more[0][0] == "EKG Interpretation" and more[1][0]["source"] == "chat_emphasis", more)

generic = apply_to_topics(["Nursing Care Overview", "Asthma"], gather(["Asthma will be on the exam."], {}))
check("a topic made of generic words never matches by accident", generic[0] == ["Asthma", "Nursing Care Overview"], generic)

# ── Where the material comes from ───────────────────────────────────────────
long_paste = "Unit 4 Study Guide\n" + "\n".join(f"## Topic {i}\n- detail line about topic {i} for the exam" for i in range(60))
s = gather(["give me questions", long_paste, "thanks"], {})
check("a pasted message in the chat is read even with no practiceProfile", "Unit 4 Study Guide" in s["pasted_text"])
check("short requests are not pasted material", "give me questions" not in s["pasted_text"])
saved = gather([], {"source": {"pastedText": "Saved notes from the profile\n" * 50}})
check("the profile's saved paste is used when the message is gone", "Saved notes" in saved["pasted_text"])
check("a short chat message with a flag counts", gather(["pharm and cardiac will be on the final"], {})["cues"])
check("an upload's flags count, labelled as uploaded notes",
      gather([], {}, "Know the cranial nerves and their functions")["cues"][0]["source"] == "uploaded_notes")
check("an empty chat is empty", is_empty(gather([], {})))

check("fallback topics are the paste's own headings", fallback_topics(s)[:2] == ["Unit 4 Study Guide", "Topic 0"], fallback_topics(s))
check("no paste, no fallback topics", fallback_topics(gather()) == [])

docs = "D" * 20000
mixed = extraction_material(docs[:15000], {"pasted_text": "P" * 9000})
check("with uploads, pasted notes get a bounded share",
      mixed.endswith("P" * PASTED_SHARE_CHARS) and "P" * (PASTED_SHARE_CHARS + 1) not in mixed
      and len(mixed) <= 8000 and mixed.startswith("D" * 4000), len(mixed))
check("with no uploads, the paste is the whole material", extraction_material("", {"pasted_text": "P" * 9000}) == "P" * 8000)
check("with no paste, the material is exactly what it was", extraction_material(docs[:15000], gather()) == docs[:8000])

block = prompt_block(signals)
check("the prompt block quotes her verbatim", "MUST know the bone & muscle landmarks" in block, block)
check("no signals, no prompt block", prompt_block(gather()) == "")

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("All checks passed.")
