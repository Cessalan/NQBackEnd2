# -*- coding: utf-8 -*-
"""
Answer-discipline tests: question anchoring, correction detection, sticky brevity.

The fixture is a REAL transcript — chat m3xpjm9XlPpQgCWdHGWB, 16 Aug 2026. A
two-day-old account pasted an assessment question, corrected the answer seven
times, never got it, and left. A day earlier the same student had burned a whole
session on the identical question. These tests replay her exact messages.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_answer_discipline.py

NursingTutor.__init__ builds LLM clients and needs an API key, so the tests use
object.__new__ and attach a bare session — the logic under test only reads
message_history and the class-level patterns.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.session import PersistentSessionContext  # noqa: E402
from services.orchestrator import NursingTutor  # noqa: E402


# ── the real transcript, in order ───────────────────────────────────────────
ASSESSMENT = (
    "Ms. Haddad is now presenting with vomiting, abdominal distention, and no stoma "
    "output for 12 hours. **Background** Loop ileostomy formed for ulcerative colitis, "
    "initial recovery uneventful. Has received IV fluids and regular analgesia. "
    "**Assessment** Complains of nausea, and worsening abdominal pain rated 8/10. "
    "Surgical team suspects a post-operative ileus. Dry lips, nausea, unable to tolerate "
    "oral fluids. Stoma pale and inactive. Increasing discomfort and signs of emotional "
    "distress, appears confused at times, unsure of where she is. **Recommendation** "
    "Plan: Insert nasogastric tube (NGT) for decompression and maintain nil by mouth."
)
TRANSCRIPT = [
    ASSESSMENT,
    "NO , THE BP IS BELOW 90",
    "Here, just referring to the vital signs, they are asking, so we don't need to write "
    "it down too much elaborately. But they said specifically, based on this vital signs, "
    "we call the med call because of the BP is in purple color, after that what kind of "
    "documentation we do, where do we do?",
    "BP is already recorded there. So that's why we call the med call. Again do we need "
    "to record the BP on the vital signs chart?",
    "but there are some some criteria under medical medical review that's also we need to "
    "do that isn't it what do you reckon",
    "No, no, no, I'm just asking do I need to do that one as well because just you know "
    "verify the question one more time please.",
    "No, no, no. What I am asking, this is my assessment question. So based on referring "
    "to the vital signs chart, which level of response is required? Describe where I "
    "would document my response and what I would document. Be specific. This is the "
    "answer I am asking.",
]


def tutor(user_msgs, include_current_in_history=True):
    """A NursingTutor with history but no LLM clients."""
    t = object.__new__(NursingTutor)
    t.session = PersistentSessionContext("test-chat")
    msgs = user_msgs if include_current_in_history else user_msgs[:-1]
    t.session.message_history = [{"role": "user", "content": m} for m in msgs]
    return t


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILURES.append(name)


print("\n=== question anchoring ===")

t = tutor(TRANSCRIPT)
anchor = t._format_task_anchor(TRANSCRIPT[-1])
check("pins the assessment question, not the last correction",
      "which level of response is required" in anchor.lower(), anchor[:120])
check("counts the pushback", "pushed back" in anchor, anchor[:160])
check("tells the model its earlier answers failed",
      "did NOT land" in anchor or "did not land" in anchor.lower())

# Turn 2: one correction so far ("NO , THE BP IS BELOW 90").
t2 = tutor(TRANSCRIPT[:2])
a2 = t2._format_task_anchor(TRANSCRIPT[1])
check("single correction is framed as a correction, not a new topic",
      "corrects your previous answer" in a2, a2[:160])
check("still anchored on the case study at turn 2",
      "Ms. Haddad" in a2, a2[:120])

# Turn 1: the paste itself. Nothing to push back on yet.
t1 = tutor(TRANSCRIPT[:1])
a1 = t1._format_task_anchor(TRANSCRIPT[0])
check("no pushback language on the first turn",
      "pushed back" not in a1 and "corrects your previous" not in a1)

# The frontend persists the user message before calling us, so the current
# message is often ALREADY the last history entry. It must not double-count.
dupe = tutor(TRANSCRIPT, include_current_in_history=True)
solo = tutor(TRANSCRIPT, include_current_in_history=False)
check("current message is not double-counted",
      dupe._format_task_anchor(TRANSCRIPT[-1]) == solo._format_task_anchor(TRANSCRIPT[-1]))

empty = tutor([])
check("no history is handled", "Nothing asked yet" in empty._format_task_anchor(""))
check("chit-chat pins nothing",
      "No specific question" in tutor(["hey"])._format_task_anchor("hey"))


print("\n=== sticky brevity ===")

# "we don't need to write it down too much elaborately" — turn 3.
check("detects the real phrasing from the transcript",
      tutor(TRANSCRIPT[:3])._wants_it_short(TRANSCRIPT[2]))

check("stays on for later turns (the bug: it reverted immediately)",
      tutor(TRANSCRIPT)._wants_it_short(TRANSCRIPT[-1]))

check("off before anyone asks", not tutor(TRANSCRIPT[:2])._wants_it_short(TRANSCRIPT[1]))

# Yesterday's session, same student, same question.
check("catches 'just to give me the answer'",
      tutor(["no you don't say you know you would write down this and that just to give "
             "me the answer"])._wants_it_short(""))

for phrase in ["keep it short please", "be brief", "can you be more concise",
               "just the answer", "sois bref"]:
    check("brevity: %r" % phrase, tutor([phrase])._wants_it_short(""))

# ── REAL messages, straight out of the database ─────────────────────────────
# These are the non-circular cases: students wrote them long before these
# regexes existed. The first version of BREVITY_MARKERS scored ~20% precision
# against the corpus and INVERTED on the last two — a student asking for a
# longer answer was being locked into short mode for the rest of the session.
CORPUS_BREVITY = [
    "summarize if for me in a but shorter",
    "please make my questions not as long, and make this a 30 question quiz",
    "briefly please. simple",
]
CORPUS_NOT_BREVITY = [
    # asks for MORE
    "Can you explain a little bit more about a diabetic ketoacidosis based on this scenario",
    # asks for answers to be withheld — the loose pattern matched "give me the answer"
    "give me some novice input and output nursing questions for a beginner. "
    "do not give me the answers till the end",
    # a request TO answer, not a request to be brief
    "can you give me the answer to the question i send you and the options to pick from",
    # an exam format, not an instruction
    "SECTION B: SHORT ANSWER QUESTIONS (SAQS) (20 MARKS) Answer all the questions",
    # explicitly wants it LONG
    "can you give me a quiz over all of the concepts. dont be afraid to make it long",
]
for phrase in CORPUS_BREVITY:
    check("corpus brevity: %r" % phrase[:40], tutor([phrase])._wants_it_short(""))
for phrase in CORPUS_NOT_BREVITY:
    check("corpus NOT brevity: %r" % phrase[:40], not tutor([phrase])._wants_it_short(""))

# The inverse must NOT trigger it — "elaborately" alone means the opposite.
for phrase in ["explain elaborately please", "give me a long detailed answer",
               "go into more detail"]:
    check("not brevity: %r" % phrase, not tutor([phrase])._wants_it_short(""))


print("\n=== correction detection ===")

corrections = ["NO , THE BP IS BELOW 90",
               "No, no, no. What I am asking, this is my assessment question.",
               "but there are some criteria under medical review",
               "Again do we need to record the BP?",
               "actually i meant the other chart",
               "wait, that's not what i asked"]
for c in corrections:
    check("correction: %r" % c[:38], bool(NursingTutor.CORRECTION_MARKERS.match(c)))

not_corrections = ["Ms. Haddad is now presenting with vomiting",
                   "what is a MET call",
                   "teach me respiratory failure"]
for c in not_corrections:
    check("not a correction: %r" % c[:38], not NursingTutor.CORRECTION_MARKERS.match(c))

# Real corrections pulled from the corpus — these are the messages the anchor
# has to recognise in the wild, not just in the one transcript it was built on.
for c in ["no, i asked how employers right and responsibily helps nurse",
          "wait, were not gonna go though all the organ systems?",
          "But you didn't answer the question, what is occurring to Kanchana's diabetes",
          "NOT 34. STEP 34 FROM THE DOCUMENT I UPLOADED",
          "i said comceptual hard",
          "No I need them about adverse effects, etc not this"]:
    check("corpus correction: %r" % c[:38], bool(NursingTutor.CORRECTION_MARKERS.match(c)))

# KNOWN GAP, asserted so it can't regress silently: "no" is a filler word in
# Slovak/Czech ("no tak..." = "well then..."), so these read as corrections.
# 2 of 38 corpus hits. Consequence is mild — the model is told to re-answer the
# pinned question — so it is documented rather than special-cased.
check("known Slavic false positive is still present (documented, not fixed)",
      bool(NursingTutor.CORRECTION_MARKERS.match("no tak uz mi to netreba ked mam po pisomke:)")))


print("\n" + "=" * 62)
if FAILURES:
    print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("All answer-discipline checks passed.")
