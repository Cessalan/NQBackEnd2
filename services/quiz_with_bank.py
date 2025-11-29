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

from models.session import PersistentSessionContext
from services.question_bank import question_bank
from tools.quiztools import (
    _generate_single_question,
    _extract_previous_questions,
    get_connection_manager
)

# Set up logging - make it visible in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def stream_quiz_with_bank(
    topic: str,
    difficulty: str,
    num_questions: int,
    source: str,
    session: PersistentSessionContext,
    empathetic_message: str = None,
    chat_id: str = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate quiz questions using Question Bank first, then LLM for the rest.

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
        ...     session=session
        ... ):
        ...     if chunk["status"] == "question_ready":
        ...         print(f"Got question: {chunk['question']['question'][:50]}...")
    """

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
    logger.info(
        f"Checking Question Bank: topic='{topic}', lang='{language}', "
        f"diff='{difficulty}', count={num_questions}"
    )

    try:
        bank_questions, from_bank_count = await question_bank.get_questions(
            topic=topic,
            language=language,
            difficulty=difficulty,
            count=num_questions,
            exclude_ids=exclude_ids
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
    # ==========================================

    if questions_to_generate > 0:
        logger.info(f"Generating {questions_to_generate} questions via LLM...")

        # Build content context based on source
        if source == "documents" and session.vectorstore:
            docs = session.vectorstore.similarity_search(query=topic, k=1000)
            full_text = "\n\n".join([doc.page_content for doc in docs])[:12000]
            content_context = f"Document content:\n{full_text}"
        else:
            content_context = f"""You are generating questions about: {topic}

                If this is a broad topic (like 'research design', 'pharmacology', 'cardiac care'),
                ensure you test diverse subtopics and concepts within that domain."""

        # Get previous questions for deduplication
        generated_questions = _extract_previous_questions(session=session, limit=30)

        # Add bank question texts to avoid duplicating those too
        for q in bank_questions:
            generated_questions.append(q.get("question", ""))

        # Generate remaining questions one at a time
        for i in range(questions_to_generate):
            current_question_num = question_index + 1

            # Check cancellation before generating
            if is_cancelled():
                logger.info(f"Quiz generation cancelled at question {current_question_num}")
                return

            # Yield progress update
            yield {
                "status": "generating",
                "current": current_question_num,
                "total": num_questions,
                "source": "llm"  # Indicates this is being generated
            }

            # Randomly assign correct answer position
            random_target_letter = random.choice(['A', 'B', 'C', 'D'])

            logger.debug(
                f"Generating Q{current_question_num}: target answer = {random_target_letter}"
            )

            # Generate the question via LLM
            question_data = await _generate_single_question(
                content=content_context,
                topic=topic,
                difficulty=difficulty,
                question_num=current_question_num,
                language=session.user_language,
                questions_to_avoid=generated_questions,
                target_letter=random_target_letter
            )

            if question_data:
                # Add to deduplication list
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

                logger.info(
                    f"Generated Q{current_question_num}: {question_data['question'][:50]}..."
                )

                # ==========================================
                # PHASE 5: SAVE TO BANK IN BACKGROUND
                # ==========================================
                # Use the question's assigned topic if the quiz-level topic is empty
                # This happens in game mode where topic="" means "use all documents"
                question_topic = question_data.get("topic", "") or topic
                if not question_topic:
                    question_topic = "general nursing"  # Fallback

                # Fire and forget - don't wait for this to complete
                asyncio.create_task(
                    _save_question_to_bank(
                        question_data=question_data,
                        topic=question_topic,  # Use question-specific topic
                        language=language,
                        difficulty=difficulty,
                        chat_id=chat_id
                    )
                )
            else:
                logger.warning(f"Failed to generate question {current_question_num}")

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
