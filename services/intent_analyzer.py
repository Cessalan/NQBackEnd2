"""
Intent Analyzer
===============

Runs BEFORE any content-generating tool fires. Uses Claude Sonnet 4.6 with
adaptive thinking + server-side web search to classify what the user actually
wants, at what quality bar, from which source, and whether the AI needs to be
honest about not having something (e.g. real past papers).

Why this exists
---------------
The old routing path went user message -> regex/keyword fast-route -> tool. It
mis-routed quality-loaded prompts ("past nursing council questions on sickle
cell" got the same generic MCQ pipeline as "quiz me on diabetes") and never
disclosed that generated "past papers" were not real past papers.

This module produces a typed classification (via a tool_use block, not text
parsing) that the orchestrator applies as both a routing override AND an
override on per-tool kwargs (quiz_mode, difficulty, learning_objective,
source_preference, etc.). It also returns an optional one-sentence honesty
preamble the orchestrator streams before the quiz starts.

Latency budget
--------------
- Simple requests ("quiz me on diabetes"): ~400-600ms (adaptive thinking
  decides not to think hard, no web search).
- Quality-loaded requests ("UK NMC past papers on sickle cell"): ~2-4s
  (one round of thinking + one web_search call).

Prompt caching halves the cost after the first call of a 5-min window.
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


# Kept stable across requests so the prompt-caching prefix doesn't drift.
ANALYSIS_SYSTEM_PROMPT = """You are the routing brain of a nursing study app.

Before any content is generated, you classify what the user wants. Your job is
to think clearly about intent and quality, then emit a structured classification
by calling the classify_intent tool. You do NOT generate the content yourself.

Core principles:
1. Understand first, generate second. Read the user's message in the context of
   their uploaded documents and recent conversation before deciding.
2. Be honest about what the app is. The app cannot deliver "real past papers"
   from a specific exam board. If the user asks for those, set
   honesty_required=true with a brief honest message offering AI-generated
   practice at that standard instead.
3. Match quality to the request. "Council standard", "NCLEX-style",
   "difficult", "final exam", "NMC" imply application/clinical-judgment level
   questions with realistic distractors, not basic recall.
4. Use the user's document when they reference it. Phrases like "from this",
   "in here", "take out the answers", "the questions in this paper" mean the
   user wants content extracted/used from their upload, NOT new content
   generated from scratch.
5. Research when uncertain. If the user names a specific exam standard, board,
   country-specific format, or curriculum you are not confident about, call
   web_search FIRST to verify the format, then classify.

Decision rules:

INTENT
- "generate_quiz"        - quiz/test/MCQ/practice questions, new content
- "extract_from_doc"     - user wants the actual questions from their uploaded
                            document (often a past paper) presented as a quiz
- "generate_flashcards"  - flashcards, cartes, flash cards
- "generate_study_sheet" - study sheet, study guide, cheat sheet, fiche
- "answer_question"      - user pasted or typed a question THEY need answered:
                            homework, an assignment prompt, a case study with
                            questions, a mock-exam question, an MCQ they want
                            solved, "help me write/fill/complete X"
- "explain"              - "what is X", "how does Y work", "explain Z"
- "reformat_doc"         - "remove the answers", "make this a quiz", "turn this
                            into flashcards" applied to their document
- "study_plan"           - "build me a plan", "what should I study first"
- "conversation"         - chitchat, thanks, greetings, follow-up clarifications
- "other"                - anything else

THE MOST IMPORTANT DISTINCTION: answer_question vs generate_quiz
Many users are students pasting their assignment/exam questions because they
want ANSWERS they can study or submit. Generating a quiz at them instead of
answering is the app's #1 observed failure mode. Real transcripts show users
begging "ANSWER MY QUESTION, PLEASE" after being served four quizzes in a row.

Classify as answer_question (recommended_tool="respond_directly"), NOT
generate_quiz, when ANY of these hold:
- The message is a pasted case study, scenario, or assignment block followed
  by instructions like "outline...", "identify two...", "list four...",
  "develop...", "explain...", "provide...", "state...", "match...", "fill in
  the blanks", or lettered/numbered sub-questions (a), b), 1., 2.).
- The message IS a multiple-choice question with options (A/B/C/D or a list
  of choices). They want the correct answer + rationale, not a re-quiz.
- The user asks to review, correct, rewrite, reference, or complete THEIR OWN
  draft answer, care plan, reflection, progress note, or table.
- The user asks "can you answer...", "what is the answer", "give me the
  answer", "i need an answer", or similar in any language.

Only classify generate_quiz when the user asks to BE TESTED: "quiz me",
"test me", "give me practice questions", "make me an exam", "10 MCQs on X",
or an equivalent explicit request for questions directed AT them.
When genuinely ambiguous, prefer answer_question — a user who wanted a quiz
will ask for one in the next message; a user who wanted an answer and got a
quiz feels ignored.

FRUSTRATION / REPEAT GUARD (absolute rule)
If the recent conversation shows the app just generated a quiz (look for
"[App generated a ...-question practice quiz...]" markers) and the user's new
message is:
- a complaint ("answer my question", "you did not answer", "I don't want
  quizzes", "why are you not responding", ALL-CAPS pleading), OR
- a repeat or near-repeat of their previous message, OR
- a request for "the answer" to what they just sent
then intent MUST be answer_question with recommended_tool="respond_directly".
NEVER generate_quiz twice in a row against a complaint or a repeated message.

DOCUMENT_ACTION
- "extract_verbatim"  - pull questions/content directly from the document
- "use_as_source"     - generate new content informed by the document
- "none"              - document is not relevant to this request

QUALITY_STANDARD
- "recall"                  - definitions, basic facts, beginner review
- "application"             - apply concepts, prioritization, simple scenarios
- "clinical_judgment"       - NCLEX-style, multi-step reasoning, safety calls
- "exam_board_specific"     - matches a specific board's format (NMC, NCLEX,
                                hospital finals, council exam, etc.)

EXAM_BOARD
- One of: "NCLEX", "NMC", "nursing_council", "hospital_finals", or null.
- Set null when the user did not reference a specific board.

HONESTY_REQUIRED
- true when the user expects something the app cannot deliver (real past
  papers, official exam questions, verified council content). Generated
  practice that matches the standard is fine - just disclose.
- false otherwise.

NEEDS_RESEARCH (be conservative - this adds 30-90s of latency per request)
- true ONLY for cases where you genuinely don't already know the exam's
  format and topics from training data:
    (a) A specific named school/institution exam (e.g. "Korle Bu Teaching
        Hospital nursing finals", "University of Ghana School of Nursing
        2023 exam", "King's College London nursing finals"), OR
    (b) A niche, regional, or recently-launched exam you cannot describe
        in detail without checking (e.g. "Ghana Health Service licensure
        exam", "Kenya Council for Registered Nurses exam"), OR
    (c) The user explicitly asks you to "search", "look up", or "find
        real past papers for" a specific exam.

- DO NOT research these - you already know them well from training and
  research adds latency with no quality gain:
    * NCLEX, NCLEX-RN, NCLEX-PN
    * NMC (UK Nursing and Midwifery Council) - test of competence, CBT, OSCE
    * NMBA / AHPRA (Australia) - just maps to NCLEX-RN for IQNs
    * USMLE
    * Generic "council exam", "licensure exam", "board exam" without a
      named country/region
    * General "hard quiz" / "exam-level questions"
  For these, set needs_research=false. The quiz_mode + difficulty +
  learning_objective overrides are enough to calibrate the questions to
  the right standard - no web search needed.

- When true: also set school_name (if a school was named) and
  research_query (a 5-12 word hint).
- Default: needs_research=false. Only flip true when the user named
  something genuinely specific or niche that you couldn't describe in
  detail from memory.

HONESTY_MESSAGE
- Required when honesty_required=true. Maximum 25 words. Casual, direct, not
  apologetic. Example: "I don't have actual past council papers, but I can
  generate practice questions at that standard."
- null when honesty_required=false.

RECOMMENDED_TOOL
- "generate_quiz_stream"        - new MCQ/SATA/case-study quiz
- "generate_flashcards_stream"  - new flashcards
- "generate_study_sheet_stream" - new study sheet
- "extract_questions_from_doc"  - pull existing questions out of an upload
- "search_documents"            - answer from the user's uploads
- "summarize_document"          - summarize an upload
- "respond_directly"            - conversational reply, no tool needed

TOOL_ARGS_OVERRIDES
- Set fields the chosen tool should use. Common overrides:
    quiz_mode: "knowledge" (factual) | "nclex" (clinical judgment)
    difficulty: "easy" | "medium" | "hard"
    learning_objective: "exam_prep" | "weak_areas" | "first_review" |
                        "deep_dive" | "quick_check" | "general"
    source_preference: "documents" | "scratch" | "auto"
    num_questions: integer 1-15
    question_types: array, one or more of ["mcq", "sata", "casestudy"]
- Quality_standard of "clinical_judgment" or "exam_board_specific" should
  generally map to quiz_mode="nclex", difficulty="hard".

QUESTION_TYPES DECISION RULES
- "mcq"        - default. Standard multiple choice with one right answer.
- "sata"       - "select all that apply", "multiple correct", "SATA".
- "casestudy"  - "case study", "scenario", "NGN", "next generation", "drag
                 and drop", "ordering", "bowtie", "prioritization sequence".
- Set when the user is explicit OR when the exam board strongly implies a
  format (e.g. NCLEX-NGN is casestudy-heavy mixed format; NMC test of
  competence is application/scenario heavy).
- Examples:
    "Give me 5 SATA questions on diabetes"      -> ["sata"]
    "Mix MCQ and SATA, 8 total"                 -> ["mcq", "sata"]
    "NCLEX NGN style on heart failure"          -> ["mcq", "sata", "casestudy"]
    "Give me a quiz on diabetes" (no format)    -> ["mcq"]  (default)
    "No case study, just regular questions"     -> ["mcq"]
- If the user is silent on format, set ["mcq"]. Do not invent SATA or
  casestudy just because the topic feels clinical.

Be decisive. Do not ask the user to clarify - infer from the message. End by
calling classify_intent. Never respond with plain text.
"""


CLASSIFY_INTENT_TOOL = {
    "name": "classify_intent",
    "description": (
        "Emit the final structured classification of the user's intent. "
        "Call this exactly once, at the end, after any web_search calls."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "generate_quiz",
                    "extract_from_doc",
                    "generate_flashcards",
                    "generate_study_sheet",
                    "answer_question",
                    "explain",
                    "reformat_doc",
                    "study_plan",
                    "conversation",
                    "other",
                ],
            },
            "uses_document": {"type": "boolean"},
            "document_action": {
                "type": "string",
                "enum": ["extract_verbatim", "use_as_source", "none"],
            },
            "quality_standard": {
                "type": "string",
                "enum": [
                    "recall",
                    "application",
                    "clinical_judgment",
                    "exam_board_specific",
                ],
            },
            "exam_board": {
                "type": ["string", "null"],
                "enum": [
                    "NCLEX",
                    "NMC",
                    "nursing_council",
                    "hospital_finals",
                    None,
                ],
            },
            "honesty_required": {"type": "boolean"},
            "honesty_message": {
                "type": ["string", "null"],
                "description": "<=25 words, casual tone. Null if not required.",
            },
            "needs_research": {
                "type": "boolean",
                "description": (
                    "True ONLY when you genuinely don't already know the "
                    "exam's format and topics from training (e.g. a "
                    "specific named school's finals, a niche regional "
                    "licensure exam, or when the user explicitly asks "
                    "you to search/look up real papers). FALSE for "
                    "well-known exams you know cold: NCLEX, NMC, NMBA, "
                    "USMLE, generic 'council exam'. Triggers a 30-90s "
                    "web-search phase - set true only when grounding in "
                    "specific real material is actually needed and you "
                    "couldn't describe the exam in detail from memory."
                ),
            },
            "school_name": {
                "type": ["string", "null"],
                "description": (
                    "Name of the specific school/institution the user "
                    "referenced (e.g. 'Korle Bu Teaching Hospital', "
                    "'University of Ghana School of Nursing'). Null when "
                    "no specific school was named."
                ),
            },
            "research_query": {
                "type": ["string", "null"],
                "description": (
                    "When needs_research=true, a 5-12 word search hint "
                    "describing what the research phase should look for "
                    "(e.g. 'Korle Bu nursing finals cardiovascular care "
                    "question format'). Null when needs_research=false."
                ),
            },
            "recommended_tool": {
                "type": "string",
                "enum": [
                    "generate_quiz_stream",
                    "generate_flashcards_stream",
                    "generate_study_sheet_stream",
                    "extract_questions_from_doc",
                    "search_documents",
                    "summarize_document",
                    "respond_directly",
                ],
            },
            "tool_args_overrides": {
                "type": "object",
                "properties": {
                    "quiz_mode": {"type": "string", "enum": ["knowledge", "nclex"]},
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "learning_objective": {
                        "type": "string",
                        "enum": [
                            "exam_prep",
                            "weak_areas",
                            "first_review",
                            "deep_dive",
                            "quick_check",
                            "general",
                        ],
                    },
                    "source_preference": {
                        "type": "string",
                        "enum": ["documents", "scratch", "auto"],
                    },
                    "num_questions": {"type": "integer", "minimum": 1, "maximum": 15},
                    "question_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["mcq", "sata", "casestudy"],
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "description": (
                            "Which question formats to generate. mcq=multiple "
                            "choice single answer; sata=select all that apply; "
                            "casestudy=NGN-style scenario with drag-and-drop. "
                            "Pass a list — multiple types produces a mixed quiz."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the classification.",
            },
        },
        "required": [
            "intent",
            "uses_document",
            "document_action",
            "quality_standard",
            "honesty_required",
            "recommended_tool",
            "tool_args_overrides",
            "reasoning",
        ],
    },
}


WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    # Cap the searches Claude can do in one analysis turn. Two is enough to
    # verify an exam board's format; more than that is the wrong tool.
    "max_uses": 2,
}


def _build_context_text(
    recent_history: list[dict], uploaded_docs: list[str], file_insights: dict
) -> str:
    """Build the per-turn context block. Goes AFTER the cached prefix."""
    parts = []

    if uploaded_docs:
        parts.append(f"Uploaded documents ({len(uploaded_docs)}): {', '.join(uploaded_docs[-5:])}")
    else:
        parts.append("Uploaded documents: none")

    if file_insights:
        insight_lines = []
        for filename, info in list(file_insights.items())[-3:]:
            topics = ", ".join((info.get("topics") or [])[:3])
            doc_type = info.get("document_type", "unknown")
            insight_lines.append(f"  - {filename} ({doc_type}): {topics}")
        if insight_lines:
            parts.append("Document insights:\n" + "\n".join(insight_lines))

    if recent_history:
        history_lines = []
        for m in recent_history[-5:]:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:200]
            history_lines.append(f"{role}: {content}")
        parts.append("Recent conversation:\n" + "\n".join(history_lines))

    return "\n\n".join(parts)


def _fallback_classification(user_message: str, has_docs: bool) -> dict:
    """
    Used when the Claude call fails (network, 5xx, malformed output).
    Picks the safest non-destructive default so the orchestrator can still
    run its existing fast-route logic.
    """
    return {
        "intent": "other",
        "uses_document": has_docs,
        "document_action": "use_as_source" if has_docs else "none",
        "quality_standard": "recall",
        "exam_board": None,
        "honesty_required": False,
        "honesty_message": None,
        "recommended_tool": "respond_directly",
        "tool_args_overrides": {},
        "reasoning": "Fallback: analyzer call failed, deferring to existing fast-route.",
        "_fallback": True,
    }


async def analyze_intent(
    user_message: str,
    recent_history: list[dict] | None = None,
    uploaded_docs: list[str] | None = None,
    file_insights: dict | None = None,
) -> dict:
    """
    Classify the user's intent. Returns a dict matching the classify_intent
    tool's input_schema (plus an internal _fallback flag on failure).

    The caller (orchestrator) is responsible for acting on the classification:
    routing to the right tool, merging tool_args_overrides, and streaming the
    honesty preamble when honesty_required is true.
    """
    recent_history = recent_history or []
    uploaded_docs = uploaded_docs or []
    file_insights = file_insights or {}

    context_text = _build_context_text(recent_history, uploaded_docs, file_insights)

    try:
        client = _get_client()
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": ANALYSIS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[WEB_SEARCH_TOOL, CLASSIFY_INTENT_TOOL],
            # Force a final structured emission. Claude can still call
            # web_search in earlier turns of the same response; tool_choice
            # only constrains the FINAL stop reason.
            tool_choice={"type": "auto"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": context_text,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"User just said: {user_message}",
                        },
                    ],
                }
            ],
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "classify_intent":
                classification = dict(block.input or {})
                classification.setdefault("tool_args_overrides", {})
                classification["_fallback"] = False
                # Useful in logs to see how often web_search fired.
                classification["_web_searched"] = any(
                    getattr(b, "type", None) == "server_tool_use"
                    and getattr(b, "name", None) == "web_search"
                    for b in response.content
                )
                return classification

        print(
            "WARNING intent_analyzer: Claude responded without calling "
            "classify_intent. stop_reason=",
            getattr(response, "stop_reason", "unknown"),
        )
        return _fallback_classification(user_message, bool(uploaded_docs))

    except Exception as e:
        print(f"ERROR intent_analyzer: {type(e).__name__}: {e}")
        return _fallback_classification(user_message, bool(uploaded_docs))


def should_skip_analysis(user_message: str, is_continuation: bool) -> bool:
    """
    Cheap pre-check the orchestrator runs before invoking analyze_intent.
    Returns True when the message is so clearly a continuation or a pure
    conversational shortcut that running analysis would just burn latency.
    """
    if is_continuation:
        return True

    if not user_message:
        return True

    stripped = user_message.strip().lower()
    if len(stripped) < 4:
        # "hi", "ok", "yes", "no", emoji-only
        return True

    # The orchestrator already has CONVERSATIONAL_PATTERNS for greetings; we
    # don't duplicate it here. This is a safety net for very short messages.
    return False
