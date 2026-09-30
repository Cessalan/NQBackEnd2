"""
Quiz Generation
===============

Generates quiz questions fresh from the user's document content via LLM.

Note on the "bank" in the filename
----------------------------------
This module used to read from / write to a shared Question Bank, hence the
filename and the legacy `stream_quiz_with_bank` function name. The bank was
bypassed because its category fallback returned off-topic results and it
couldn't honour per-user constraints (quiz_mode, question style, etc.). The
file/symbol names are kept as aliases so existing callers don't break, but
the bank itself is not consulted in this user flow.

Usage:
------
    from services.quiz_with_bank import stream_quiz_questions

    async for chunk in stream_quiz_questions(
        topic="cardiac medications",
        difficulty="medium",
        num_questions=4,
        source="documents",
        session=session,
        empathetic_message="I understand you want to practice...",
        chat_id="abc123"
    ):
        yield chunk
"""

import asyncio
import random
import re
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from models.session import PersistentSessionContext
from tools.quiztools import (
    _generate_single_question,
    get_connection_manager
)

# Set up logging - make it visible in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ==========================================
# CONCEPT EXTRACTION FOR GUARANTEED UNIQUE QUESTIONS
# ==========================================

async def extract_concepts_from_content(
    content: str,
    topic: str,
    num_concepts: int,
    language: str = "english",
    quiz_mode: str = "knowledge",
    learning_objective: str = "general",
    avoid_concepts: List[str] = None
) -> List[str]:
    """
    Extract distinct, testable concepts from document content, guided by the
    student's learning objective so the most relevant concepts are selected.

    avoid_concepts: questions the student has already been asked in this quiz.
    Uniqueness is guaranteed by picking distinct concepts up front rather than
    de-duplicating questions afterwards, so extending a quiz has to exclude the
    earlier batch HERE — a second call with no memory would happily re-extract
    the same high-yield concepts and ask the same things again.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # ── Intent-aware selection instructions ───────────────────────────────
    objective_instructions = {
        "exam_prep": (
            "PRIORITY: Select the highest-yield concepts most frequently tested on NCLEX. "
            "Focus on: priority/safety topics, commonly confused medications and their side effects, "
            "critical lab values, emergency nursing interventions, and conditions with high mortality risk. "
            "Skip minor details — choose what a student MUST know to pass."
        ),
        "weak_areas": (
            "PRIORITY: Select concepts that are notoriously tricky, commonly misunderstood, or "
            "where students frequently make errors. Focus on: easily confused look-alike/sound-alike drugs, "
            "conditions with overlapping symptoms, situations that require careful priority judgment, "
            "and interventions that seem counterintuitive. Choose concepts designed to challenge and build mastery."
        ),
        "first_review": (
            "PRIORITY: Select foundational concepts in a logical learning order — definitions first, "
            "then mechanisms, then clinical presentation, then management. "
            "Ensure each concept builds on the previous one. "
            "Avoid edge cases or advanced complications — focus on core understanding."
        ),
        "deep_dive": (
            "PRIORITY: Select specific, detailed concepts that go beyond surface-level understanding. "
            "Focus on: pathophysiology mechanisms, pharmacological mechanisms of action, "
            "nuanced clinical decision-making, complications and their management, "
            "and evidence-based rationale behind nursing interventions."
        ),
        "quick_check": (
            "PRIORITY: Select the most important, high-impact concepts — the ones a student should "
            "know cold. Focus on the core essentials only, avoiding peripheral details."
        ),
        "general": (
            "Select a balanced mix of concepts: definitions, mechanisms, clinical presentation, "
            "nursing interventions, and patient education. Ensure good coverage of the topic."
        ),
    }
    selection_guidance = objective_instructions.get(learning_objective, objective_instructions["general"])

    # Capped: the exclusion list grows with every batch, and past a dozen or so
    # it costs more prompt than it buys in novelty.
    avoid_block = ""
    if avoid_concepts:
        recent = [c for c in avoid_concepts if c][-12:]
        if recent:
            joined = "\n".join(f"- {c[:160]}" for c in recent)
            avoid_block = (
                "\nALREADY ASKED — do not test these again, and avoid close "
                f"paraphrases or narrower restatements of them:\n{joined}\n"
            )

    if quiz_mode == "nclex":
        prompt = f"""You are a nursing education expert preparing a student for NCLEX.
From the following content about "{topic}", extract exactly {num_concepts} DISTINCT clinical concepts to test.

{selection_guidance}

Each concept should be:
- A specific clinical situation (e.g., "Patient with acute liver failure developing hepatic encephalopathy — nursing priority actions")
- Focused on nursing assessment, prioritization, or intervention
- Distinct enough from other concepts to produce unique questions
- Written as a testable scenario seed (not a question itself)
{avoid_block}
Content:
{content[:8000]}

Return ONLY a JSON array of {num_concepts} concept strings. No explanations.
Language: {language}
"""
    elif quiz_mode == "applied":
        # Grounding for applied questions is enforced HERE, not in the question
        # template. The extractor is already document-bound, and it is allowed to
        # return fewer concepts than asked — which is exactly the fallback we
        # want: if her material cannot support N applied concepts, the caller
        # fills the remainder with recall rather than inventing a condition.
        #
        # Same discipline as services/course_intelligence.py: a claim that cannot
        # cite its source is dropped, never softened.
        prompt = f"""You are a nursing education expert.
From the following content about "{topic}", extract up to {num_concepts} DISTINCT APPLIED concepts to test.

{selection_guidance}

An APPLIED concept pairs something the content NAMES with what the nurse does about it.
Write each one as "<subject named in the content> - what the nurse monitors/does/teaches".

Examples of the shape:
- "Systemic lupus erythematosus - routine monitoring the nurse performs"
- "Furosemide therapy - laboratory values the nurse follows"
- "Walker use at discharge - instructions the nurse gives"

HARD RULES:
- The SUBJECT (condition, medication, device, procedure or situation) MUST be named
  in the content below. Do not introduce a condition the content never mentions.
- The nursing response does NOT need to be in the content — that part may come from
  your own nursing knowledge.
- Do NOT produce concepts about ranking, prioritising, or what to do FIRST.
- If the content supports fewer than {num_concepts} such concepts, return FEWER.
  Returning an invented subject is far worse than returning a short list.
{avoid_block}
Content:
{content[:8000]}

Return ONLY a JSON array of concept strings (at most {num_concepts}). No explanations.
Language: {language}
"""

    else:
        prompt = f"""You are a nursing education expert.
From the following content about "{topic}", extract exactly {num_concepts} DISTINCT factual concepts to test.

{selection_guidance}

Each concept should be:
- A specific, testable fact or principle (e.g., "The normal range for serum ammonia in liver failure")
- Distinct enough from other concepts to produce unique questions
- Clear and focused on one idea
{avoid_block}
Content:
{content[:8000]}

Return ONLY a JSON array of {num_concepts} concept strings. No explanations.
Language: {language}
"""

    try:
        response = await llm.ainvoke(prompt)
        response_text = response.content.strip()

        # Clean up response - handle markdown code blocks
        if response_text.startswith("```"):
            # Remove markdown code block markers
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

        # Parse JSON
        import json
        concepts = json.loads(response_text)

        if isinstance(concepts, list) and len(concepts) > 0:
            logger.info(f"✅ Extracted {len(concepts)} concepts for quiz generation")
            for i, concept in enumerate(concepts[:5]):  # Log first 5
                logger.info(f"   Concept {i+1}: {concept[:60]}...")
            return concepts[:num_concepts]  # Ensure we don't exceed requested count
        else:
            logger.warning(f"⚠️ Concept extraction returned invalid format: {type(concepts)}")
            return []

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse concept extraction response: {e}")
        logger.error(f"   Response was: {response_text[:200]}...")
        return []
    except Exception as e:
        logger.error(f"❌ Concept extraction failed: {e}")
        return []


async def _extract_concepts_for_mode_plan(
    content_context: str,
    topic: str,
    mode_sequence: List[str],
    language: str,
    learning_objective: str,
    avoid_concepts: List[str] = None,
) -> tuple:
    """
    Extract one concept per planned question, honouring each slot's quiz mode.

    Returns (concepts, mode_sequence). Both are the same length, and the mode
    sequence comes back POSSIBLY ALTERED — that is the point of this function.

    The applied-mode extractor is instructed to return fewer concepts rather
    than invent a condition the student's documents never mention. When it does,
    the unfilled slots are demoted to "knowledge" instead of being dropped or
    filled with an invented subject. Demotion only ever goes applied -> recall;
    nothing is ever promoted, which mirrors the confidence discipline in
    services/course_intelligence.py.
    """
    total = len(mode_sequence)
    if total == 0:
        return [], []

    needed = {}
    for mode in mode_sequence:
        needed[mode] = needed.get(mode, 0) + 1

    pools = {}
    for mode, count in needed.items():
        extracted = await extract_concepts_from_content(
            content=content_context,
            topic=topic,
            num_concepts=count,
            language=language,
            quiz_mode=mode,
            learning_objective=learning_objective,
            avoid_concepts=avoid_concepts,
        )
        pools[mode] = list(extracted or [])
        if len(pools[mode]) < count:
            logger.info(
                f"Concept pool for '{mode}' came back short "
                f"({len(pools[mode])}/{count}) — those slots will fall back to recall"
            )

    # Top up the recall pool once if anything came up short, so demoted slots
    # still get a real concept rather than a placeholder.
    shortfall = sum(max(0, c - len(pools.get(m, []))) for m, c in needed.items())
    if shortfall > 0:
        already = list(avoid_concepts or []) + [c for pool in pools.values() for c in pool]
        extra = await extract_concepts_from_content(
            content=content_context,
            topic=topic,
            num_concepts=shortfall,
            language=language,
            quiz_mode="knowledge",
            learning_objective=learning_objective,
            avoid_concepts=already,
        )
        pools.setdefault("knowledge", []).extend(list(extra or []))

    concepts = []
    resolved_modes = []
    for slot, mode in enumerate(mode_sequence):
        pool = pools.get(mode) or []
        if pool:
            concepts.append(pool.pop(0))
            resolved_modes.append(mode)
            continue

        recall_pool = pools.get("knowledge") or []
        if recall_pool:
            concepts.append(recall_pool.pop(0))
            resolved_modes.append("knowledge")
            continue

        # Nothing left anywhere — same placeholder shape the caller used before.
        concepts.append(f"Aspect {slot + 1} of {topic}")
        resolved_modes.append("knowledge")

    return concepts, resolved_modes


_WORD = re.compile(r"[a-zA-ZÀ-ſ]{4,}")
# Words every nursing concept shares; overlap on these proves nothing.
_GENERIC_WORDS = {"nursing", "patient", "patients", "care", "management", "assessment", "interventions",
                  "intervention", "with", "from", "that", "this", "their", "about", "signs", "symptoms"}


def off_source_share(concepts, source_text):
    """Share of concepts with no distinctive word in the source text."""
    source_words = {w.lower() for w in _WORD.findall(source_text or "")}

    def on_source(concept):
        words = {w.lower() for w in _WORD.findall(str(concept))} - _GENERIC_WORDS
        return bool(words & source_words) if words else True

    concepts = [c for c in concepts if c]
    return (sum(not on_source(c) for c in concepts) / len(concepts)) if concepts else 0.0


async def stream_quiz_questions(
    topic: str,
    difficulty: str,
    num_questions: int,
    source: str,
    session: PersistentSessionContext,
    empathetic_message: str = None,
    chat_id: str = None,
    question_types: List[str] = None,
    existing_topics: List[str] = None,
    quiz_mode: str = "knowledge",
    learning_objective: str = "general",
    user_prompt: str = None,
    additional_context: str = None,
    existing_questions: List[str] = None,
    index_offset: int = 0,
    node_difficulty: int = None,
    source_text: str = None,
    guidance: str = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate quiz questions fresh from document content via LLM.
    Supports multiple question types (MCQ, SATA, etc.)

    Args:
        topic: Subject area for the quiz (e.g., "cardiac medications")
        difficulty: Question difficulty level ("easy", "medium", "hard")
        num_questions: Total number of questions to generate
        source: Source preference ("documents" or "scratch")
        session: Current session context with user info and vectorstore
        empathetic_message: Optional empathetic message to stream first
        chat_id: Chat ID for cancellation checking
        question_types: List of question types to generate ["mcq", "sata", "casestudy"]
                       Defaults to ["mcq"] if not specified
        existing_topics: User's existing topics from progress tracking. LLM will try
                        to match questions to these topics when applicable.
        quiz_mode: "knowledge" for factual recall questions (default),
                   "nclex" for clinical judgment questions,
                   "applied" for a MIX of recall and applied questions (a named
                   condition from her documents + what the nurse monitors/does).
                   "applied" is the only mode that produces a mixed batch; the
                   other two apply to every question, so existing callers are
                   unaffected.
        node_difficulty: The study plan node's 1-3 difficulty. Only consulted
                   when quiz_mode == "applied", where it sets the recall/applied
                   ratio. Missing or unrecognised falls back to the recall-leaning
                   end. See distribute_quiz_modes in tools/sata_prompts.py.
        existing_questions: Question text already in this quiz. Set when EXTENDING
                   a quiz on demand so the new batch avoids repeating the old one.
        index_offset: Position the first new question occupies in the full quiz.
                   Without it an extension batch would re-emit index 0 and the
                   client would overwrite the questions already on screen.

    Yields:
        Status updates and complete questions in the same format as
        stream_quiz_questions:
        - {"status": "empathetic_message_start", ...}
        - {"status": "empathetic_message_chunk", "chunk": "...", ...}
        - {"status": "empathetic_message_complete", ...}
        - {"status": "generating", "current": 1, "total": 4, ...}
        - {"status": "question_ready", "question": {...}, "index": 0}
        - {"status": "quiz_complete", "total_generated": 4}

    Example:
        >>> async for chunk in stream_quiz_questions(
        ...     topic="cardiac medications",
        ...     difficulty="medium",
        ...     num_questions=4,
        ...     source="scratch",
        ...     session=session,
        ...     question_types=["mcq", "sata"],  # Mixed format quiz
        ...     quiz_mode="knowledge"  # Factual recall questions
        ... ):
        ...     if chunk["status"] == "question_ready":
        ...         print(f"Got question: {chunk['question']['question'][:50]}...")
    """
    # Import SATA, Case Study, and Unfolding Case Study generators for mixed type quizzes
    from tools.sata_prompts import (
        generate_sata_question,
        distribute_question_types,
        distribute_quiz_modes,
    )
    from tools.casestudy_prompts import generate_casestudy_question
    from tools.matrix_prompts import generate_matrix_question
    from tools.unfolding_casestudy_prompts import generate_unfolding_casestudy

    # Default to MCQ if no types specified
    if question_types is None or len(question_types) == 0:
        question_types = ["mcq"]

    logger.info(f"Question types requested: {question_types}")
    logger.info(f"Quiz mode: {quiz_mode}")
    print(f"🎮 [QUIZ_WITH_BANK] Quiz mode received: {quiz_mode}")

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================

    def is_cancelled() -> bool:
        """Check if the user cancelled the quiz generation."""
        manager = get_connection_manager()
        if manager and chat_id:
            return manager.is_cancelled(chat_id)
        return False

    # ==========================================
    # PHASE 1: STREAM EMPATHETIC MESSAGE (if provided)
    # ==========================================

    if empathetic_message:
        logger.info("Starting empathetic message streaming...")

        # Check cancellation before starting
        if is_cancelled():
            logger.info("Quiz generation cancelled before empathetic message")
            return

        # Signal start of empathetic message
        yield {
            "status": "empathetic_message_start",
            "message": "Understanding your learning needs..."
        }

        # Stream the message word by word for a human-like effect
        words = empathetic_message.split()
        current_text = ""

        for i, word in enumerate(words):
            # Check cancellation
            if is_cancelled():
                logger.info("Quiz generation cancelled during empathetic message")
                return

            current_text += word + " "

            # Stream in chunks (every 4 words) for better UX
            if (i + 1) % 4 == 0 or i == len(words) - 1:
                yield {
                    "status": "empathetic_message_chunk",
                    "chunk": current_text.strip(),
                    "progress": int((i + 1) / len(words) * 100)
                }

        # Signal empathetic message complete
        yield {
            "status": "empathetic_message_complete",
            "full_message": empathetic_message
        }

        logger.info("Empathetic message streaming complete")

    # ==========================================
    # PHASE 2: GET QUESTIONS FROM BANK (instant!)
    # ==========================================

    # Determine language for bank query
    language = session.user_language or "en"
    if language.lower().startswith("fr"):
        language = "fr"
    elif language.lower().startswith("es"):
        language = "es"
    else:
        language = "en"

    # Track questions we've already used (for deduplication)
    # Get question IDs from previous quizzes to avoid repeats
    exclude_ids = []  # Could be enhanced to track question IDs across sessions

    # Try to get questions from the bank
    # For mixed-type quizzes, we query for each type separately
    # For single-type quizzes, we query for that specific type
    primary_question_type = question_types[0] if question_types else "mcq"

    # The Question Bank is bypassed by design. Bank lookups returned off-topic
    # results when the requested category had no exact match (the fallback
    # broadened the search), and bank entries couldn't honour per-call
    # constraints (quiz_mode, question style, learning_objective, etc.). Every
    # question is generated fresh from the user's document content below.
    logger.info(f"Generating all {num_questions} questions fresh via LLM (bank bypassed)")
    bank_questions = []
    from_bank_count = 0

    # Calculate how many we still need to generate
    questions_to_generate = num_questions - from_bank_count

    logger.info(
        f"Question Bank result: {from_bank_count} from bank, "
        f"{questions_to_generate} to generate"
    )

    # ==========================================
    # PHASE 3: YIELD BANK QUESTIONS (instant delivery!)
    # ==========================================

    all_questions = []
    # Starts past the questions already on screen when this is an extension, so
    # emitted indices continue the quiz instead of restarting it.
    question_index = index_offset
    quiz_total = index_offset + num_questions

    for question in bank_questions:
        # Check cancellation
        if is_cancelled():
            logger.info(f"Quiz generation cancelled at bank question {question_index + 1}")
            return

        # Yield progress update
        yield {
            "status": "generating",
            "current": question_index + 1,
            "total": quiz_total,
            "source": "bank"  # Indicates this came from the bank
        }

        # Small delay to simulate "instant" but not jarring delivery
        await asyncio.sleep(0.1)

        # Yield the question
        yield {
            "status": "question_ready",
            "question": question,
            "index": question_index,
            "source": "bank"
        }

        all_questions.append(question)
        question_index += 1

        logger.debug(f"Delivered bank question {question_index}: {question['question'][:50]}...")

    # ==========================================
    # PHASE 4: GENERATE REMAINING QUESTIONS VIA LLM
    # Using CONCEPT-FIRST approach for guaranteed unique questions
    # ==========================================

    if questions_to_generate > 0:
        logger.info(f"Generating {questions_to_generate} questions via LLM (concept-first approach)...")

        # Build content context based on source
        if source == "documents" and session.vectorstore:
            docs = session.vectorstore.similarity_search(query=topic, k=30)
            full_text = "\n\n".join([doc.page_content for doc in docs])[:12000]
            content_context = f"Document content:\n{full_text}"
        elif source_text:
            # Notes the student pasted into the chat (services/practice_profile).
            # Before this, only the router's topic reached this point: a pasted
            # mental-health study guide arrived as its title, "Exam I Study
            # Guide - NURS 3900", and came back as potassium and heparin.
            content_context = (
                "The student's own study notes (pasted into the chat). Every question "
                "must test a topic that appears in these notes:\n" + source_text[:12000]
            )
        else:
            content_context = f"""You are generating questions about: {topic}

                If this is a broad topic (like 'research design', 'pharmacology', 'cardiac care'),
                ensure you test diverse subtopics and concepts within that domain."""

        # Prepend exam research brief (when present) — this is the
        # web-search-gathered material about the specific exam the user
        # named. We put it BEFORE the document/topic content so the LLM
        # treats it as primary grounding, with documents as secondary.
        if additional_context:
            content_context = (
                "EXAM RESEARCH BRIEF (this exam's actual style and topics — "
                "ground questions in this material):\n"
                f"{additional_context}\n\n"
                "─────────────────────────────────\n\n"
                f"{content_context}"
            )
            logger.info(f"📚 Prepended exam research brief ({len(additional_context)} chars) to content context")

        # ==========================================
        # STEP 1: Extract unique concepts FIRST
        # This guarantees no duplicate questions!
        # ==========================================
        logger.info(f"🧠 Step 1: Extracting {questions_to_generate} unique concepts...")
        print(f"\n{'='*60}")
        print(f"🧠 [CONCEPT-FIRST] Extracting {questions_to_generate} concepts from content...")
        print(f"{'='*60}\n")

        # Plan the per-question mode first. Only "applied" yields a mixed batch;
        # "knowledge" and "nclex" stay one mode for the whole batch, so every
        # pre-existing caller generates exactly what it generated before.
        if quiz_mode == "applied":
            mode_sequence = distribute_quiz_modes(questions_to_generate, node_difficulty)
        else:
            mode_sequence = [quiz_mode] * questions_to_generate

        if guidance:
            content_context = f"{guidance}\n\n{content_context}"

        concepts, mode_sequence = await _extract_concepts_for_mode_plan(
            content_context=content_context,
            topic=topic,
            mode_sequence=mode_sequence,
            language=session.user_language or "english",
            learning_objective=learning_objective,
            avoid_concepts=existing_questions,
        )

        # Pasted notes: a concept sharing no distinctive word with them is
        # off-source. One stricter re-extraction; if that still drifts, keep
        # the questions (an empty quiz helps nobody) but log it so the rate
        # is visible.
        if source_text and concepts and off_source_share(concepts, source_text) > 0.5:
            logger.warning(f"off_source: {off_source_share(concepts, source_text):.0%} of concepts not in pasted notes, retrying")
            strict_context = ("Choose concepts ONLY from topics named in the student's notes below. "
                              "Do not add general nursing topics they do not mention.\n\n" + content_context)
            retry_concepts, retry_modes = await _extract_concepts_for_mode_plan(
                content_context=strict_context, topic=topic, mode_sequence=list(mode_sequence),
                language=session.user_language or "english", learning_objective=learning_objective,
                avoid_concepts=existing_questions,
            )
            if retry_concepts and off_source_share(retry_concepts, source_text) < off_source_share(concepts, source_text):
                concepts, mode_sequence = retry_concepts, retry_modes
            if off_source_share(concepts, source_text) > 0.5:
                logger.warning(f"off_source persisted for chat {chat_id}: {concepts}")

        if not concepts:
            logger.warning("⚠️ Concept extraction failed, falling back to topic-only generation")
            # Fallback: generate simple concept placeholders
            concepts = [f"Aspect {index_offset + i + 1} of {topic}" for i in range(questions_to_generate)]
            mode_sequence = ["knowledge"] * questions_to_generate

        logger.info(f"✅ Got {len(concepts)} concepts, generating one question per concept...")
        logger.info(f"Quiz mode distribution: {mode_sequence}")

        # Distribute question types for remaining questions
        remaining_type_sequence = distribute_question_types(questions_to_generate, question_types)
        logger.info(f"Question type distribution: {remaining_type_sequence}")

        # Track generated questions (no longer needed for deduplication, but kept for logging)
        generated_questions = []

        # ==========================================
        # STEP 2: Generate questions in PARALLEL
        # ==========================================
        # OPTIMIZATION: Previously questions were generated sequentially,
        # taking 1-3s per question (10-30s for 10 questions).
        # Now we generate in parallel and yield as each completes.
        # Expected improvement: ~3x faster (parallel batch of 3-4 at a time)
        # ==========================================

        logger.info(f"⚡ PARALLEL GENERATION: Starting {len(concepts)} questions in parallel...")
        print(f"\n{'='*60}")
        print(f"⚡ [PARALLEL] Generating {len(concepts)} questions concurrently...")
        print(f"{'='*60}\n")

        # Create a task for each question
        async def generate_question_task(concept_idx: int, concept: str):
            """Generate a single question - returns (index, question_data)"""
            current_question_num = question_index + concept_idx + 1
            current_question_type = remaining_type_sequence[concept_idx] if concept_idx < len(remaining_type_sequence) else "mcq"
            # Per-slot mode. Identical to `quiz_mode` for every non-applied batch.
            current_quiz_mode = mode_sequence[concept_idx] if concept_idx < len(mode_sequence) else quiz_mode

            try:
                question_data = None

                # Generate based on question type, passing the specific concept
                if current_question_type == "matrix":
                    question_data = await generate_matrix_question(
                        topic=concept, difficulty=difficulty, question_num=current_question_num,
                        language=session.user_language, content_context=content_context,
                        questions_to_avoid=[], quiz_mode=current_quiz_mode)
                elif current_question_type == "sata":
                    question_data = await generate_sata_question(
                        topic=concept,
                        difficulty=difficulty,
                        question_num=current_question_num,
                        language=session.user_language,
                        content_context=content_context,
                        questions_to_avoid=[],  # No blocking on previous - concepts are unique
                        quiz_mode=current_quiz_mode
                    )
                elif current_question_type == "casestudy":
                    question_data = await generate_casestudy_question(
                        topic=concept,
                        difficulty=difficulty,
                        question_num=current_question_num,
                        language=session.user_language,
                        content_context=content_context,
                        questions_to_avoid=[]
                    )
                elif current_question_type == "unfoldingcase" or current_question_type == "unfoldingCase":
                    question_data = await generate_unfolding_casestudy(
                        topic=concept,
                        difficulty=difficulty,
                        language=session.user_language or "english",
                        questions_to_avoid=[]
                    )
                else:
                    # Generate MCQ question (default)
                    random_target_letter = random.choice(['A', 'B', 'C', 'D'])
                    question_data = await _generate_single_question(
                        content=content_context,
                        topic=concept,
                        difficulty=difficulty,
                        question_num=current_question_num,
                        language=session.user_language,
                        questions_to_avoid=[],
                        target_letter=random_target_letter,
                        existing_topics=existing_topics,
                        quiz_mode=current_quiz_mode,
                        learning_objective=learning_objective
                    )

                    if question_data and 'questionType' not in question_data:
                        question_data['questionType'] = 'mcq'

                # Misconception-ledger key. _generate_single_question keeps a
                # short label from its own reasoning; SATA / case-study
                # generators do not, so fall back to the concept seed this
                # question was grown from. Either way every question leaves
                # here carrying a `concept`, because a question with no key is
                # invisible to progress tracking.
                if question_data and not question_data.get('concept'):
                    question_data['concept'] = str(concept)[:120]

                return (concept_idx, current_question_type, question_data)

            except Exception as e:
                logger.error(f"❌ Error generating question for concept {concept_idx}: {e}")
                return (concept_idx, current_question_type, None)

        # Signal that parallel generation is starting
        yield {
            "status": "generating",
            "current": question_index + 1,
            "total": quiz_total,
            "source": "llm",
            "parallel": True,
            "batch_size": len(concepts)
        }

        # Create all tasks
        tasks = [
            asyncio.create_task(generate_question_task(idx, concept))
            for idx, concept in enumerate(concepts)
        ]

        # Process results as they complete (fastest first)
        completed_count = 0
        for coro in asyncio.as_completed(tasks):
            # Check cancellation
            if is_cancelled():
                logger.info(f"Quiz generation cancelled - cancelling remaining tasks")
                for task in tasks:
                    task.cancel()
                return

            try:
                concept_idx, q_type, question_data = await coro
                completed_count += 1

                if question_data:
                    # Track for logging
                    generated_questions.append(question_data['question'])

                    # Yield the question immediately as it completes
                    yield {
                        "status": "question_ready",
                        "question": question_data,
                        "index": question_index,
                        "source": "llm",
                        "completed": completed_count,
                        "remaining": len(concepts) - completed_count
                    }

                    all_questions.append(question_data)
                    question_index += 1

                    logger.info(f"✅ Q{completed_count}/{len(concepts)} ({q_type}): {question_data['question'][:50]}...")
                else:
                    logger.warning(f"❌ Failed to generate question {concept_idx + 1}")

            except asyncio.CancelledError:
                logger.info("Task was cancelled")
                continue
            except Exception as e:
                logger.error(f"Error processing completed task: {e}")
                continue

        logger.info(f"⚡ PARALLEL GENERATION COMPLETE: {completed_count} questions generated")

    # ==========================================
    # PHASE 6: SIGNAL COMPLETION
    # ==========================================

    yield {
        "status": "quiz_complete",
        "total_generated": len(all_questions),
        "stats": {
            "from_bank": from_bank_count,
            "generated": questions_to_generate,
            "total": len(all_questions)
        }
    }

    logger.info(
        f"Quiz complete: {len(all_questions)} questions total "
        f"({from_bank_count} from bank, {len(all_questions) - from_bank_count} generated)"
    )


# ==========================================
# LEGACY ALIAS
# ==========================================
# Older code and the file name itself reference `stream_quiz_with_bank`. The
# canonical name is now `stream_quiz_questions` (since the bank is bypassed).
# Existing imports keep working through this alias.
stream_quiz_with_bank = stream_quiz_questions
