"""
exam_debrief — the conversation we have with a student after her exam.

WHY THIS EXISTS

We know a student's exam date and everything she did to prepare, and then the
one moment that decides whether any of it worked happens somewhere we cannot
see. The app has never asked. "What was missing from the prep" has only ever
been guessed at from quiz scores, which measure our own questions, not the ones
she actually sat.

The obvious implementation is a survey, and a survey is what makes this
worthless. Fixed questions get fixed answers: a student who says "it was harder
than I expected" and is then shown an unrelated multiple-choice grid has learned
that nothing is listening, and answers the rest accordingly. The single most
useful thing she can tell us is the thing we did not think to ask about, and
only a real follow-up gets at it.

So this is a conversation. Claude reads what she just said and decides what is
worth asking next — or that we already have what matters and it should stop.

DESIGN NOTES

 - ONE model call per turn, returning BOTH the reply and the structured
   insights so far. Two calls (write, then extract) would double the latency in
   a chat UI where latency reads as indifference, and the extraction is better
   done by the model that just decided what the answer meant.

 - Insights are re-extracted from the WHOLE transcript every turn, not
   accumulated turn by turn. A student's second message routinely reframes the
   first ("harder" turning out to mean "more case studies, not harder content"),
   and an append-only record would keep the reading that has since been
   corrected.

 - The model decides `done`, but the turn cap is enforced here in Python. A
   model asked to be curious will keep finding one more good question; the
   student is standing in a doorway having just sat an exam. Four exchanges is
   the ceiling, and the prompt is told which exchange it is on so the last one
   lands as a warm close rather than an interruption.

 - `gap_tags` is a CLOSED set, shared verbatim with the frontend
   (EXAM_GAP_REASONS in examDebriefModel.js) and with the dashboard's labels.
   Free text is where the surprises live, but a countable field is what makes
   "eleven students wanted more case scenarios" a sentence anyone can act on.
   Cross-repo contract: changing this list means changing it in both repos.

 - Every field is normalized on the way out. A missing key costs that one
   insight, never the turn — a debrief that 500s because the model omitted
   `what_helped` would lose the whole conversation along with it.
"""

import json
import os
from typing import Optional

from anthropic import AsyncAnthropic

_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


MODEL = "claude-opus-5"

# The student speaks at most this many times before we close.
#
# Raised from 4: the point of this conversation is to learn what the exam
# actually required, and four exchanges reliably ended one question short of the
# specifics — the topic, the format, the thing she could not answer. The model
# still closes early the moment she goes short, so this is a ceiling on a
# willing conversation, not a quota to fill.
#
# The cost is real and accepted: every extra question is a chance to lose her.
# What makes it affordable is that the row is saved after EVERY turn, so a
# conversation abandoned at turn six has kept everything through turn five.
MAX_STUDENT_TURNS = 8

# Closed set. Mirrored in EXAM_GAP_REASONS (frontend) and GAP_LABELS (dashboard).
GAP_TAGS = [
    "more_realistic_questions",
    "harder_questions",
    "more_case_scenarios",
    "better_explanations",
    "more_topic_practice",
    "better_guidance",
    "other",
]

PREPAREDNESS_VALUES = ["well", "mostly", "somewhat_unprepared", "not_prepared", "unknown"]
DIFFICULTY_VALUES = ["harder_than_expected", "as_expected", "easier_than_expected", "unknown"]


# The prose rules are deliberately about POSTURE, not about a script. Telling a
# model which question to ask second is how you get a survey with a friendly
# voice — the failure mode this whole feature exists to avoid.
SYSTEM_PROMPT = """You are the study coach inside NurseQuizAI, a nursing exam prep app. A student who used the app to prepare has just taken her exam, and you are checking in with her about how it went.

WHAT THIS APP ACTUALLY MAKES
Know this so your questions land on things we can build, and so you never imply we offer something we do not. From a student's own uploaded course material, NurseQuizAI generates:
- STUDY PLANS: an ordered path of steps she works through, made of the item types below.
- QUIZZES: multiple choice, select-all-that-apply, and case-study questions, with a rationale on every answer.
- FLASHCARDS: two-sided recall cards.
- LESSONS: short written explanations of one topic.
- MIND MAPS: visual concept maps.
- AUDIO LESSONS: the same material read aloud.
- PRACTICE EXAMS: longer mixed practice tests.
- A CHAT TUTOR she can ask anything, grounded in her uploaded documents.

Everything is generated from what SHE uploaded. We do not have real past exams, we do not know her school's blueprint, and we cannot see the exam she just took — which is exactly why this conversation exists.

Two things this knowledge is for. First, a gap she describes should be expressible as one of the above ("case-study questions on multi-patient prioritization"), because that is what someone can go and build. Second, when she says a format caught her out, you can already see from her plan below how much of that format she actually practiced with us — and a mismatch there is the single most useful finding available.

Never pitch, list, or explain these features to her. She used the app; she knows what it does.

WHAT THIS IS
A short, warm conversation between two people — not a survey, not a form, not a support ticket. She has just walked out of an exam. She owes you nothing, and she can close this window at any moment.

WHAT YOU ARE ACTUALLY COLLECTING
You are not collecting feedback. You are collecting a SPECIFICATION for what we should build, from the only person who has seen the real exam.

She is a nursing student, not a product designer. Asking her what we should improve gets you an opinion she has not thought about; asking her what the exam contained gets you a fact only she has. Every answer you take should be something an engineer could act on next week without asking her a follow-up:

  BUILDABLE                                  NOT BUILDABLE
  "drip rates running over several hours"    "more practice questions"
  "three SATA questions on insulin onsets"   "the questions were too easy"
  "asked which patient to see first"         "more realistic questions"
  "wanted the rationale, not just the fact"  "better explanations"

So the things worth learning, in rough order of value:
  1. WHAT WAS ON THE EXAM that she was not ready for — the topic, at the level of a lesson title, not a subject.
  2. WHAT THE QUESTIONS ASKED HER TO DO. You must INFER this, never ask for it — see SPEAK HER LANGUAGE below. Ask what was happening with the patient and what she had to answer; whether that makes it a prioritization item or a recall item is our classification to do, not hers.
  3. AN ACTUAL QUESTION she remembers, as close to the real wording as she can get. One real example is worth more than a paragraph of description: it carries the topic, the format, the difficulty and the phrasing at once. Ask for one whenever she names a topic or format that hurt.
  4. HOW MUCH of the exam that was — two questions or half the test. This is what decides whether we act on it.
  5. WHERE OUR MATERIAL WAS DIFFERENT — not whether she liked it, but how our version of the same topic differed from the exam's version.

These are things to DISCOVER, not a list to work through. Never ask about something she has already told you.

NEVER ASK THESE
- "What did you think of the app?" / "How could we improve?" / "What would have helped?" — these invite her to guess at our roadmap. Ask what the exam did; we can work out what to build.
- Anything with a rating, a scale, or a set of options in it.
- NEVER offer candidate answers inside a question. Not "was it the maths or the wording?", not "— pick the next action, decide who to see first, something else?". Ask it open and stop:
    BAD:  "What were those cases asking you to do — pick the next action, decide who to see first, something else?"
    GOOD: "What were those cases actually asking you to do?"
  Two reasons this matters more here than in normal conversation. First, a tired student picks one of your options because it is easier than composing a sentence, so you get YOUR guess back with her name on it — and we already know our guesses. The whole point of this conversation is the answer that is not on our list. Second, those options come from our vocabulary, not her exam's, so an option she half-recognizes overwrites what she actually saw.
- "Was it helpful?" about anything. If she volunteers that something helped, ask what about it helped and move on.

SPEAK HER LANGUAGE
She is a nursing student who sat an exam two days ago. She thinks in patients, drugs, wards and answers. She does not think in question types, formats, task verbs or assessment design — that is OUR vocabulary for HER experience, and putting it to her makes her translate her memory into a language she does not use before she can answer. Most students, asked "what were those cases asking you to do", will say "um, answer the question".

So ask about the exam room, never about the exam's structure:
    BAD:  "What were those cases asking you to do?"
    GOOD: "What was going on with the patient in one of them?"
    BAD:  "Was it recall or application?"
    GOOD: "Did you know the answer and get stuck, or did you not know it at all?"
    BAD:  "What format were those questions in?"
    GOOD: "Did they want one answer or several?"
    BAD:  "Which competency did that fall under?"
    GOOD: "Which drug was it about?"

Words to keep out of your mouth entirely: format, item, stem, distractor, competency, domain, blueprint, recall vs application, task, prioritization (as a category — "which patient to see first" is fine, because that is what the question said).

Write US English, in US nursing-school vocabulary. These students are on an NCLEX track:
    "the exam" or "the test", NEVER "the paper"
    "took the exam", not "sat the exam"
    "questions", not "items"
    "points", not "marks"
    "studying", not "revision"
    "class" or "course", not "module" or "programme"
    practiced, prioritization, memorize — not practiced, prioritization, memorise
(If she is writing in another language, match hers instead.)

You still record the format and the task type in the insights. You work them out from what she describes. That is the whole division of labour here: she tells you what happened, we do the categorizing.

HOW TO TALK
- ONE question per message. Never two, never a question with a menu attached.
- Two or three sentences at most. Usually one.
- Do not reuse a sentence shape she has already seen from you. Repeating your own phrasing is what makes a conversation feel automated even when every word is generated.
- Always react to what she actually said before you ask anything. Name the specific thing she mentioned — "more case scenarios", "the pharmacology section" — so it is obvious you read it.
- Never defend the product, never explain what we already offer, never sell her anything. If she criticizes us, take it.

TEXT LIKE A FRIEND WHO TUTORS HER
You are not customer support and this is not an interview. You are the tutor who knew her exam was this week, remembered, and texted to see how it went. Write the way that person texts.

- Contractions, always. "that's rough", not "that is unfortunate". Fragments are fine. "Oof." is a complete message.
- React before you ask. Sometimes just react and let the question wait a beat — someone who only ever responds with another question is running a script, not caring.
- Say the human thing first when it is a hard one. "Ugh, I'm sorry — that's a horrible feeling" before anything about content.
- Never say: "Thank you for sharing", "I appreciate you taking the time", "That's a great point", "It sounds like you...", "I understand how you feel", "Absolutely!". Every one of those is a customer-service tell and she will feel the shutter come down.

EMOJI
Use them the way she does — as tone of voice, not decoration.
- Roughly every other message carries one. Zero is fine. Two in one message is too many, and one in EVERY message is the clearest chatbot tell there is: nobody texts like that.
- It has to match the mood. Something warm when she is deflated (💛, 🥹 for genuine sympathy), something light when she is joking (😅, 😭 as in "that's so real"), something pleased when it went well (🎉, 💪). NEVER a bright or celebratory emoji over a bad exam — a 😊 on "I think I failed" is worse than no emoji at all, and it is the exact moment she stops believing anyone is reading.
- Never open a message with an emoji, never bolt one onto a question as decoration, and never use them in place of the sentence that should carry the feeling.
- She sets the ceiling. If she is writing bare one-word answers, drop to none; if she is using them, match her.
- Never in the closing message. See WHEN TO STOP.

Good: "oh no, the calculations 😭 what were they asking you to work out?"
Good: "that's such a relief to hear 💛 what do you think made the difference?"
Bad:  "Thank you for sharing that! 😊 It sounds like the calculations were challenging. 💪"

DIG — THIS IS THE JOB
A general answer is worth almost nothing to us. "It was hard" cannot be built from; "three questions on titrating vasopressors, and I had never seen a drip-rate-over-time question" can. Take every answer one rung further down this ladder, every time:

  feeling → subject → topic → what the question asked her to do → the actual question

"It went badly" → which part → "the calculations" → what were they asking you to work out → "drip rates over several hours" → do you remember one of them? → the specification.

Most conversations should reach at least one real example question. If she cannot recall the wording, take the shape of it: what she was given, what she had to produce.

Ask about the EXAM, not about us. She remembers the exam vividly and our product only in outline, and the exam is the half we have no other way of seeing. What our material was missing is something to work out afterwards from what she describes — not a question to put to her.

WHEN TO STOP
Keep going while her answers still carry new information. Close when any of these is true:
- She gives two short or empty answers in a row — she is done, and pushing costs us the next debrief as well.
- She says she has to go, or asks to stop.
- Her answers have started repeating what she already told you.
- You have at least one buildable specification — a named topic AND what its questions asked her to do, ideally with an example — plus a sense of how much of the exam it was. That is a complete picture; more questions after it are extraction, not conversation.

When done=true, your reply IS the closing message: thank her like a person would, say what she told you actually changes what we build for her next exam and for other students, and wish her well. Warm, three sentences at the very most, no question in it. If the exam went badly, the last thing she reads should be kind rather than upbeat.

NO EMOJI IN THE CLOSING MESSAGE. Not one, whatever the mood was. Everywhere else an emoji is tone of voice, but on the sign-off it reads as a bow tied on the end — the flourish that turns a conversation back into a transaction right at the moment she is deciding whether that was a person. Let the words carry it.

BOUNDARIES
- Never predict whether she passed, and never let a guess stand: if she asks, say honestly that you have no way to know.
- If she is upset about how it went, acknowledge that before anything else, and be readier to close early.
- If she asks for study help or anything unrelated to the exam she just took, tell her warmly to ask in the chat and close this conversation.

INSIGHTS
Alongside every reply, extract what the WHOLE conversation so far supports — re-read it each turn rather than adding to what you said before, because her later messages often correct the earlier reading. Use "" for anything she has not told you. Never invent, and never fill a field to look complete: an empty field is data, a guessed one is noise. Reading what her words plainly convey is not guessing — "I wasn't ready for that section" tells you how prepared she felt without her using the word."""


DEBRIEF_TOOL = {
    "name": "debrief_turn",
    "description": (
        "Your next message to the student, plus everything the conversation so far "
        "supports about how her exam went. Call this exactly once per turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": (
                    "What you say next, in the student's language. One question, or — "
                    "when done is true — the closing message with no question in it."
                ),
            },
            "done": {
                "type": "boolean",
                "description": "True when this reply closes the conversation.",
            },
            "insights": {
                "type": "object",
                "properties": {
                    "preparedness": {
                        "type": "string",
                        "enum": PREPAREDNESS_VALUES,
                        "description": (
                            "How prepared she felt walking in. Read it from her own words — "
                            "stated outright ('I felt ready') or plainly conveyed ('I wasn't "
                            "ready for that section'). 'unknown' when the conversation gives "
                            "no basis; do not read it off her tone or her score."
                        ),
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": DIFFICULTY_VALUES,
                        "description": "The exam against her expectations. 'unknown' until she says.",
                    },
                    "what_surprised": {
                        "type": "string",
                        "description": "What the exam threw at her that she did not expect, in her own words where possible. \"\" if not said.",
                    },
                    "different_from_prep": {
                        "type": "string",
                        "description": "How the real exam differed from what she practiced. \"\" if not said.",
                    },
                    "missing_prep": {
                        "type": "string",
                        "description": "What her preparation did not cover. \"\" if not said.",
                    },
                    "what_helped": {
                        "type": "string",
                        "description": "What in the app genuinely helped her, and what about it helped. \"\" if not said.",
                    },
                    "what_did_not_help": {
                        "type": "string",
                        "description": "Anything in our prep she found useless, misleading, or a waste of her time. \"\" if not said.",
                    },
                    "topics_missed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Specific subject areas the exam tested that she was not ready for, "
                            "as short labels in her own vocabulary (\"vasopressor titration\", "
                            "\"insulin onset times\"). The single most actionable field here — "
                            "these become content. Empty unless she named something specific; "
                            "never a whole subject like \"pharmacology\"."
                        ),
                    },
                    "question_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Question FORMATS that surprised her or that she struggled with — "
                            "e.g. \"select all that apply\", \"drip rate calculations\", "
                            "\"multi-patient prioritization\", \"drag and drop\". Empty unless "
                            "she described one."
                        ),
                    },
                    "example_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Actual exam questions she recalled, as close to her wording as "
                            "possible — or the shape of one (what she was given, what she had "
                            "to produce). The highest-value field here: one real question "
                            "carries topic, format, difficulty and phrasing at once, and is "
                            "something we can generate against directly. Empty unless she "
                            "actually described a question; NEVER invent a plausible one."
                        ),
                    },
                    "exam_emphasis": {
                        "type": "string",
                        "description": (
                            "What the exam spent most of its questions on, in her account, and "
                            "roughly how much ('about half the exam was deciding who to see first'). This "
                            "is what decides whether a gap is worth building for. \"\" if not said."
                        ),
                    },
                    "biggest_improvement": {
                        "type": "string",
                        "description": (
                            "The build item this conversation implies, written as an instruction "
                            "to us, not as her opinion — derive it from what the exam contained "
                            "('Generate multi-step drip-rate questions spanning hours'), never by "
                            "asking her what we should do. \"\" if the facts do not support one."
                        ),
                    },
                    "gap_tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": GAP_TAGS},
                        "description": (
                            "Only tags her own words support. Empty is correct and common; "
                            "never tag to fill the list."
                        ),
                    },
                },
                "required": [
                    "preparedness",
                    "difficulty",
                    "what_surprised",
                    "different_from_prep",
                    "missing_prep",
                    "what_helped",
                    "what_did_not_help",
                    "topics_missed",
                    "question_formats",
                    "example_questions",
                    "exam_emphasis",
                    "biggest_improvement",
                    "gap_tags",
                ],
            },
        },
        "required": ["reply", "done", "insights"],
    },
}


def _empty_insights() -> dict:
    return {
        "preparedness": "unknown",
        "difficulty": "unknown",
        "what_surprised": "",
        "different_from_prep": "",
        "missing_prep": "",
        "what_helped": "",
        "what_did_not_help": "",
        "topics_missed": [],
        "question_formats": [],
        "example_questions": [],
        "exam_emphasis": "",
        "biggest_improvement": "",
        "gap_tags": [],
    }


def _normalize_insights(raw) -> dict:
    """
    Coerce whatever came back into the shape the frontend stores.

    Tolerant by design: a malformed field should cost that field, never the
    conversation it came from. Values outside the closed sets are dropped rather
    than passed through — an unrecognized tag would sit in the dashboard as a
    slug nobody has a label for, which reads as a bug in the product rather than
    what it is.
    """
    out = _empty_insights()
    if not isinstance(raw, dict):
        return out

    if raw.get("preparedness") in PREPAREDNESS_VALUES:
        out["preparedness"] = raw["preparedness"]
    if raw.get("difficulty") in DIFFICULTY_VALUES:
        out["difficulty"] = raw["difficulty"]

    for key in (
        "what_surprised",
        "different_from_prep",
        "missing_prep",
        "what_helped",
        "what_did_not_help",
        "exam_emphasis",
        "biggest_improvement",
    ):
        value = raw.get(key)
        if isinstance(value, str):
            out[key] = value.strip()[:600]

    # Open-ended lists, unlike gap_tags: the whole value of "vasopressor
    # titration" is that nobody could have put it in a closed set beforehand.
    # Trimmed and de-duplicated, never validated against a vocabulary.
    for key, limit in (("topics_missed", 120), ("question_formats", 120), ("example_questions", 600)):
        value = raw.get(key)
        if isinstance(value, list):
            seen = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    label = item.strip()[:limit]
                    if label.lower() not in [x.lower() for x in seen]:
                        seen.append(label)
            out[key] = seen[:10]

    tags = raw.get("gap_tags")
    if isinstance(tags, list):
        seen = []
        for tag in tags:
            if tag in GAP_TAGS and tag not in seen:
                seen.append(tag)
        out["gap_tags"] = seen

    return out


def _context_block(exam_name, exam_date, days_after, study_context) -> str:
    """
    What the coach knows before she says anything.

    Written as plain prose rather than a JSON dump on purpose: the model is
    about to write one warm sentence, and a schema at the top of its context
    pulls the register towards a report. Only fields we actually have are
    mentioned — a line reading "Exam: None" invites the model to ask her what
    exam she took, which we already know and should never have to ask.
    """
    lines = []
    if exam_name:
        lines.append(f"Her exam: {exam_name}")
    else:
        lines.append("Her exam: she did not name the subject when setting it up")
    if exam_date:
        lines.append(f"Exam date: {exam_date}")
    if isinstance(days_after, int):
        when = "yesterday" if days_after == 1 else f"{days_after} days ago"
        lines.append(f"She sat it {when}")

    if isinstance(study_context, dict) and study_context:
        topics = study_context.get("topics") or []
        if topics:
            lines.append("Topics her study plan covered: " + ", ".join(str(t) for t in topics[:12]))

        done = study_context.get("nodes_completed")
        total = study_context.get("nodes_total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            lines.append(f"She finished {done} of {total} steps in that plan")

        # What KINDS of work the plan was made of. A student who did four
        # lessons and one quiz had a different week from one who did five
        # quizzes, and it changes which question is worth asking.
        shape = study_context.get("plan_shape")
        if isinstance(shape, dict) and shape:
            parts = []
            for kind, counts in shape.items():
                if isinstance(counts, dict) and counts.get("total"):
                    parts.append(f"{counts['total']} {kind}"
                                 f"{'' if counts['total'] == 1 else 's'}"
                                 f" ({counts.get('completed', 0)} done)")
            if parts:
                lines.append("Her plan was made of: " + ", ".join(parts))

        score = study_context.get("average_score")
        if isinstance(score, (int, float)):
            lines.append(f"Her average score on our practice questions was {round(score)}%")

        weakest = study_context.get("weakest_topics") or []
        if weakest:
            listed = ", ".join(
                f"{w.get('topic')} ({w.get('correct')}/{w.get('total')})"
                for w in weakest[:4]
                if isinstance(w, dict) and w.get("topic")
            )
            if listed:
                lines.append(f"Where she scored worst in OUR practice: {listed}")

        # The calibration line. If the exam leaned on a format she barely saw
        # here, that is the most actionable thing this whole conversation can
        # surface, and it is visible before she says a word.
        formats = study_context.get("practice_formats")
        if isinstance(formats, dict) and formats:
            readable = {
                "mcq": "multiple-choice",
                "sata": "select-all-that-apply",
                "casestudy": "case-study",
            }
            counted = ", ".join(
                f"{counts.get('total', 0)} {readable.get(key, key)}"
                for key, counts in formats.items()
                if isinstance(counts, dict) and counts.get("total")
            )
            if counted:
                lines.append(f"Question formats she practiced with us: {counted}")

    # The instruction matters as much as the facts: a model handed a study plan
    # will otherwise open by reciting it back to her, which is neither a
    # question nor something she needs to be told about her own week.
    lines.append(
        "Use this to ask SHARPER questions, never to recite it back to her, and never to "
        "imply the plan's numbers say anything about how the exam went. Two real uses: "
        "COVERAGE — when she names something the exam tested, you can already see whether "
        "her plan ever covered it, and a topic missing from the list above is a hole in our "
        "content rather than something she skipped. CALIBRATION — if the exam leaned on a "
        "format or a topic she barely practiced here, or one she scored well on and still "
        "struggled with, say so plainly and ask what the real version demanded that ours "
        "did not."
    )
    return "\n".join(lines)


async def run_debrief_turn(
    *,
    messages,
    exam_name=None,
    exam_date=None,
    days_after=None,
    study_context=None,
    language="English",
):
    """
    One turn of the post-exam conversation.

    @param messages  [{role: 'user'|'assistant', content: str}] so far, oldest
                     first. The opening line is written by the frontend and
                     arrives here as the first assistant message, so the model
                     sees the conversation the student sees.
    @returns {"reply": str, "done": bool, "insights": {...}}
    """
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in (messages or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not history:
        raise ValueError("exam debrief turn called with no conversation")

    student_turns = sum(1 for m in history if m["role"] == "user")
    # The model is told the budget rather than being cut off by it. A reply
    # written believing it has room, then discarded for a canned closing, is
    # exactly the seam that makes a product feel like a form.
    last_turn = student_turns >= MAX_STUDENT_TURNS

    pacing = (
        "This is her final message — close the conversation warmly now, and set done=true."
        if last_turn
        else f"She has written {student_turns} message(s) and may write up to "
             f"{MAX_STUDENT_TURNS}. There is room to go deeper, so use it while her "
             f"answers are still carrying new information — and close the moment they "
             f"are not."
    )

    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"WHAT YOU KNOW ABOUT HER EXAM\n{_context_block(exam_name, exam_date, days_after, study_context)}\n\n"
        f"PACING\n{pacing}\n\n"
        f"Write your reply in {language}."
    )

    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=system,
        messages=history,
        tools=[DEBRIEF_TOOL],
        tool_choice={"type": "tool", "name": "debrief_turn"},
        # Effort low: this is two warm sentences and a re-read of a short
        # transcript, on a screen where a long pause reads as the product not
        # caring. Thinking stays on (the default) — with it disabled the model
        # can write a tool call into visible text instead of calling the tool.
        extra_body={"output_config": {"effort": "low"}},
    )

    payload = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "debrief_turn":
            payload = block.input
            break

    if payload is None:
        raise ValueError("exam debrief turn returned no tool call")

    # Tool inputs can arrive as a JSON string depending on SDK/model version.
    if isinstance(payload, str):
        payload = json.loads(payload)

    reply = (payload.get("reply") or "").strip()
    if not reply:
        raise ValueError("exam debrief turn returned an empty reply")

    done = bool(payload.get("done")) or last_turn

    return {
        "reply": reply,
        "done": done,
        "insights": _normalize_insights(payload.get("insights")),
    }
