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
    learning_objective: str = "general"
) -> List[str]:
    """
    Extract distinct, testable concepts from document content, guided by the
    student's learning objective so the most relevant concepts are selected.
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

    if quiz_mode == "nclex":
        prompt = f"""You are a nursing education expert preparing a student for NCLEX.
From the following content about "{topic}", extract exactly {num_concepts} DISTINCT clinical concepts to test.

{selection_guidance}

Each concept should be:
- A specific clinical situation (e.g., "Patient with acute liver failure developing hepatic encephalopathy — nursing priority actions")
- Focused on nursing assessment, prioritization, or intervention
- Distinct enough from other concepts to produce unique questions
- Written as a testable scenario seed (not a question itself)

Content:
{content[:8000]}

Return ONLY a JSON array of {num_concepts} concept strings. No explanations.
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
                   "nclex" for clinical judgment questions

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
    from tools.sata_prompts import generate_sata_question, distribute_question_types
    from tools.casestudy_prompts import generate_casestudy_question
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
    question_index = 0

    for question in bank_questions:
        # Check cancellation
        if is_cancelled():
            logger.info(f"Quiz generation cancelled at bank question {question_index + 1}")
            return

        # Yield progress update
        yield {
            "status": "generating",
            "current": question_index + 1,
            "total": num_questions,
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

        concepts = await extract_concepts_from_content(
            content=content_context,
            topic=topic,
            num_concepts=questions_to_generate,
            language=session.user_language or "english",
            quiz_mode=quiz_mode,
            learning_objective=learning_objective
        )

        if not concepts:
            logger.warning("⚠️ Concept extraction failed, falling back to topic-only generation")
            # Fallback: generate simple concept placeholders
            concepts = [f"Aspect {i+1} of {topic}" for i in range(questions_to_generate)]

        logger.info(f"✅ Got {len(concepts)} concepts, generating one question per concept...")

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

            try:
                question_data = None

                # Generate based on question type, passing the specific concept
                if current_question_type == "sata":
                    question_data = await generate_sata_question(
                        topic=concept,
                        difficulty=difficulty,
                        question_num=current_question_num,
                        language=session.user_language,
                        content_context=content_context,
                        questions_to_avoid=[],  # No blocking on previous - concepts are unique
                        quiz_mode=quiz_mode
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
                        quiz_mode=quiz_mode,
                        learning_objective=learning_objective
                    )

                    if question_data and 'questionType' not in question_data:
                        question_data['questionType'] = 'mcq'

                return (concept_idx, current_question_type, question_data)

            except Exception as e:
                logger.error(f"❌ Error generating question for concept {concept_idx}: {e}")
                return (concept_idx, current_question_type, None)

        # Signal that parallel generation is starting
        yield {
            "status": "generating",
            "current": question_index + 1,
            "total": num_questions,
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
