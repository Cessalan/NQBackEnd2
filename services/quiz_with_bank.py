"""
Quiz Generation with Question Bank Integration
===============================================

This module provides an enhanced quiz question generator that uses
the Question Bank for instant delivery when possible.

The Flow:
---------
1. User requests a quiz (e.g., "Give me 4 questions on cardiac medications")
2. We first check the Question Bank for matching questions
3. Questions from the bank are delivered instantly (no LLM needed)
4. Any remaining questions are generated via LLM
5. Newly generated questions are saved to the bank in the background

Benefits:
---------
- Faster quiz delivery (bank questions = instant)
- Lower API costs (fewer LLM calls over time)
- Consistent quality (bank questions can be curated)
- The bank grows organically with each generated question

Usage:
------
    from services.quiz_with_bank import stream_quiz_with_bank

    # In orchestrator, replace stream_quiz_questions with:
    async for chunk in stream_quiz_with_bank(
        topic="cardiac medications",
        difficulty="medium",
        num_questions=4,
        source="documents",
        session=session,
        empathetic_message="I understand you want to practice...",
        chat_id="abc123"
    ):
        # Handle each chunk (same format as original stream_quiz_questions)
        yield chunk
"""

import asyncio
import random
import logging
import traceback
from typing import AsyncGenerator, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from models.session import PersistentSessionContext
from services.question_bank import question_bank
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
    quiz_mode: str = "knowledge"
) -> List[str]:
    """
    Extract distinct, testable concepts from document content.

    This is Step 1 of the concept-first approach:
    1. Extract N unique concepts from the document
    2. Generate one question per concept (guarantees no duplicates)

    Args:
        content: Document content or topic description
        topic: The main topic being studied
        num_concepts: Number of concepts to extract
        language: Language for the concepts
        quiz_mode: "knowledge" for factual concepts, "nclex" for clinical scenarios

    Returns:
        List of distinct concept strings
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7  # Some creativity for diverse concepts
    )

    # Different prompts for knowledge vs NCLEX mode
    if quiz_mode == "nclex":
        prompt = f"""You are a nursing education expert. From the following content about "{topic}",
extract exactly {num_concepts} DISTINCT clinical scenarios or nursing situations that could be tested.

Each concept should be:
- A specific clinical situation (e.g., "Patient with acute MI showing signs of cardiogenic shock")
- Focused on nursing assessment, intervention, or prioritization
- Different enough from other concepts to create unique questions
- Testable with a single question

Content:
{content[:8000]}

Return ONLY a JSON array of {num_concepts} concept strings. No explanations.
Example format: ["Concept 1 description", "Concept 2 description", ...]

Language: {language}
"""
    else:
        prompt = f"""You are an education expert. From the following content about "{topic}",
extract exactly {num_concepts} DISTINCT factual concepts that could be tested.

Each concept should be:
- A specific, testable fact or principle (e.g., "The normal range for adult heart rate")
- Different enough from other concepts to create unique questions
- Clear and focused on one idea

Content:
{content[:8000]}

Return ONLY a JSON array of {num_concepts} concept strings. No explanations.
Example format: ["Concept 1 description", "Concept 2 description", ...]

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


async def stream_quiz_with_bank(
    topic: str,
    difficulty: str,
    num_questions: int,
    source: str,
    session: PersistentSessionContext,
    empathetic_message: str = None,
    chat_id: str = None,
    question_types: List[str] = None,
    existing_topics: List[str] = None,
    quiz_mode: str = "knowledge"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate quiz questions using Question Bank first, then LLM for the rest.
    Supports multiple question types (MCQ, SATA, etc.)

    This is a drop-in replacement for stream_quiz_questions that adds
    Question Bank integration. The yielded chunks have the same format.

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
        >>> async for chunk in stream_quiz_with_bank(
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

    # IMPORTANT: Skip question bank for "knowledge" mode
    # The bank contains mostly NCLEX-style questions (clinical scenarios),
    # so we need to generate fresh questions for knowledge mode (factual questions)
    if quiz_mode == "knowledge":
        logger.info(
            f"Skipping Question Bank for quiz_mode='knowledge' - generating fresh factual questions"
        )
        print(f"🎯 [QUIZ_WITH_BANK] Skipping bank for knowledge mode - will generate all {num_questions} questions")
        bank_questions = []
        from_bank_count = 0
    else:
        
        print(f"Looking in the Bank for Questions!")
        logger.info(
            f"Checking Question Bank: topic='{topic}', lang='{language}', "
            f"diff='{difficulty}', count={num_questions}, qtype='{primary_question_type}'"
        )

        try:
            bank_questions, from_bank_count = await question_bank.get_questions(
                topic=topic,
                language=language,
                difficulty=difficulty,
                count=num_questions,
                exclude_ids=exclude_ids,
                question_type=primary_question_type  # Pass question type for filtering
            )
        except Exception as e:
            logger.error(f"Error getting questions from bank: {e}")
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
            docs = session.vectorstore.similarity_search(query=topic, k=1000)
            full_text = "\n\n".join([doc.page_content for doc in docs])[:12000]
            content_context = f"Document content:\n{full_text}"
        else:
            content_context = f"""You are generating questions about: {topic}

                If this is a broad topic (like 'research design', 'pharmacology', 'cardiac care'),
                ensure you test diverse subtopics and concepts within that domain."""

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
            quiz_mode=quiz_mode
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
        # STEP 2: Generate one question per concept
        # No retry loops needed - concepts are already unique!
        # ==========================================
        for concept_idx, concept in enumerate(concepts):
            current_question_num = question_index + 1
            current_question_type = remaining_type_sequence[concept_idx] if concept_idx < len(remaining_type_sequence) else "mcq"

            # Check cancellation before generating
            if is_cancelled():
                logger.info(f"Quiz generation cancelled at question {current_question_num}")
                return

            # Yield progress update
            yield {
                "status": "generating",
                "current": current_question_num,
                "total": num_questions,
                "source": "llm",
                "question_type": current_question_type,
                "concept": concept[:50]  # Include concept in status for debugging
            }

            logger.info(f"📝 Q{current_question_num}: Generating {current_question_type} about: {concept[:60]}...")

            question_data = None

            # Generate based on question type, passing the specific concept
            if current_question_type == "sata":
                question_data = await generate_sata_question(
                    topic=concept,  # Use concept as topic for focused generation
                    difficulty=difficulty,
                    question_num=current_question_num,
                    language=session.user_language,
                    content_context=content_context,
                    questions_to_avoid=generated_questions,
                    quiz_mode=quiz_mode
                )
            elif current_question_type == "casestudy":
                question_data = await generate_casestudy_question(
                    topic=concept,
                    difficulty=difficulty,
                    question_num=current_question_num,
                    language=session.user_language,
                    content_context=content_context,
                    questions_to_avoid=generated_questions
                )
            elif current_question_type == "unfoldingcase" or current_question_type == "unfoldingCase":
                question_data = await generate_unfolding_casestudy(
                    topic=concept,
                    difficulty=difficulty,
                    language=session.user_language or "english",
                    questions_to_avoid=generated_questions
                )
            else:
                # Generate MCQ question (default)
                random_target_letter = random.choice(['A', 'B', 'C', 'D'])

                question_data = await _generate_single_question(
                    content=content_context,
                    topic=concept,  # Use concept as topic for focused generation
                    difficulty=difficulty,
                    question_num=current_question_num,
                    language=session.user_language,
                    questions_to_avoid=generated_questions,
                    target_letter=random_target_letter,
                    existing_topics=existing_topics,
                    quiz_mode=quiz_mode
                )

                if question_data and 'questionType' not in question_data:
                    question_data['questionType'] = 'mcq'

            if question_data:
                # Track for logging
                generated_questions.append(question_data['question'])

                # Check cancellation before yielding
                if is_cancelled():
                    logger.info(f"Quiz generation cancelled after generating question {current_question_num}")
                    return

                # Yield the question
                yield {
                    "status": "question_ready",
                    "question": question_data,
                    "index": question_index,
                    "source": "llm"
                }

                all_questions.append(question_data)
                question_index += 1

                q_type = question_data.get('questionType', 'mcq')
                logger.info(f"✅ Q{current_question_num} ({q_type}): {question_data['question'][:50]}...")
            else:
                logger.warning(f"❌ Failed to generate question for concept: {concept[:50]}...")

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


async def _save_question_to_bank(
    question_data: Dict,
    topic: str,
    language: str,
    difficulty: str,
    chat_id: str = None
) -> None:
    """
    Save a generated question to the bank in the background.

    This is called as a fire-and-forget task after generating each question.
    It doesn't block the quiz delivery.

    Args:
        question_data: The generated question dictionary
        topic: The nursing topic
        language: Language code
        difficulty: Difficulty level
        chat_id: Source chat ID for tracking
    """
    print(f"💾 [QUESTION BANK] Attempting to save question for topic='{topic}'...")

    try:
        question_id = await question_bank.save_question(
            question_data=question_data,
            topic=topic,
            language=language,
            difficulty=difficulty,
            chat_id=chat_id
        )

        if question_id:
            print(f"✅ [QUESTION BANK] Successfully saved question: {question_id}")
            logger.info(f"Saved question to bank: {question_id}")
        else:
            print(f"⚠️ [QUESTION BANK] Question not saved (likely duplicate)")
            logger.debug("Question not saved (likely duplicate)")

    except Exception as e:
        # Log but don't fail - saving to bank is optional
        print(f"❌ [QUESTION BANK] Error saving question: {e}")
        print(f"❌ [QUESTION BANK] Traceback: {traceback.format_exc()}")
        logger.error(f"Error saving question to bank: {e}")


# ==========================================
# BACKWARDS COMPATIBILITY WRAPPER
# ==========================================

async def stream_quiz_questions_with_bank(
    topic: str,
    difficulty: str,
    num_questions: int,
    source: str,
    session: PersistentSessionContext,
    empathetic_message: str = None,
    chat_id: str = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Alias for stream_quiz_with_bank for backwards compatibility.

    Use stream_quiz_with_bank directly for new code.
    """
    async for chunk in stream_quiz_with_bank(
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        source=source,
        session=session,
        empathetic_message=empathetic_message,
        chat_id=chat_id
    ):
        yield chunk
