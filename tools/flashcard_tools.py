"""
Flashcard Generation Tool for Streaming to Frontend

This module provides streaming flashcard generation similar to the quiz system.
Frontend expects these WebSocket status updates:
1. "flashcard_generating" - Generation started
2. "flashcard_ready" - Individual flashcard ready (streamed one by one)
3. "flashcard_complete" - All flashcards generated

Based on frontend requirements in ChatFlashcard.js
"""

from typing import Dict, Any, Optional, AsyncGenerator
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import random

# Import session management from quiztools
from tools.quiztools import (
    get_session,
    PersistentSessionContext,
    get_connection_manager
)


@tool
async def generate_flashcards_stream(
    topic: str,
    num_cards: int = 15,
    source_preference: str = "auto"
) -> Dict[str, Any]:
    """
    Generate flashcards for the student to review and memorize.

    Use this when students request flashcards or want active recall practice:
    - "Create flashcards about [topic]"
    - "Make flashcards for studying [subject]"
    - "I need flashcards to memorize [concept]"
    - "Generate flashcards from my documents"

    Args:
        topic: Subject area for flashcards (e.g., "cardiac medications", "anatomy")
        num_cards: Number of flashcards to generate (1-15, default: 15)
        source_preference: "documents" (from uploads), "scratch" (general), or "auto"

    Returns:
        Dictionary signaling flashcard streaming should begin
    """

    print("📇 FLASHCARD TOOL: Initiating streaming flashcard generation")

    try:
        session = get_session()

        # Determine source
        if source_preference == "auto":
            source = "documents" if session.documents else "scratch"
        else:
            source = source_preference

        # Validate and limit num_cards to 15 (matching frontend limit)
        num_cards = max(1, min(15, num_cards))

        # Record tool call
        session.tool_calls.append({
            "tool": "generate_flashcards_stream",
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "num_cards": num_cards,
            "source": source,
            "status": "streaming_initiated"
        })

        # Return signal to orchestrator to handle streaming
        return {
            "status": "flashcard_streaming_initiated",
            "metadata": {
                "topic": topic,
                "num_cards": num_cards,
                "source": source,
                "language": session.user_language
            },
            "message": f"Starting flashcard generation: {num_cards} cards on {topic}"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Flashcard generation failed: {str(e)}"
        }


async def stream_flashcards(
    topic: str,
    num_cards: int,
    source: str,
    session: PersistentSessionContext,
    chat_id: str = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generator that yields complete flashcards one at a time.
    Called by orchestrator after tool signals streaming intent.

    Yields status updates matching frontend expectations:
    1. "flashcard_generating" - Initial status
    2. "flashcard_ready" - Each individual card
    3. "flashcard_complete" - All cards done

    Args:
        topic: Subject area for flashcards
        num_cards: Number of flashcards to generate
        source: Source preference ("documents" or "scratch")
        session: Current session context
        chat_id: Chat ID for cancellation checking

    Yields:
        Status updates and complete flashcards
    """

    # Helper to check cancellation
    def is_cancelled():
        manager = get_connection_manager()
        if manager and chat_id:
            return manager.is_cancelled(chat_id)
        return False

    # Get content based on source
    if source == "documents" and session.vectorstore:
        print(f"📚 Using documents for flashcard context")
        docs = session.vectorstore.similarity_search(query=topic, k=1000)
        full_text = "\n\n".join([doc.page_content for doc in docs])[:15000]
        content_context = f"Document content:\n{full_text}"
    else:
        print(f"💡 Generating flashcards from general knowledge")
        content_context = f"""You are generating flashcards about: {topic}

        Cover key concepts, definitions, and important facts that students need to memorize.
        If this is a broad topic, ensure diverse coverage of subtopics."""

    # Signal generation start
    yield {
        "status": "flashcard_generating",
        "message": f"Generating {num_cards} flashcards about {topic}...",
        "current": 0,
        "total": num_cards
    }

    # Track generated flashcards for deduplication
    generated_fronts = []

    # Generate flashcards one at a time
    for card_num in range(1, num_cards + 1):

        # Check cancellation
        if is_cancelled():
            print(f"🛑 Flashcard generation cancelled at card {card_num}/{num_cards}")
            return

        # Yield progress
        yield {
            "status": "generating",
            "current": card_num,
            "total": num_cards
        }

        # Generate single flashcard
        flashcard_data = await _generate_single_flashcard(
            content=content_context,
            topic=topic,
            card_num=card_num,
            language=session.user_language,
            cards_to_avoid=generated_fronts
        )

        if flashcard_data:
            generated_fronts.append(flashcard_data['front'])

            print(f"✅ Flashcard {card_num}/{num_cards} generated - Topic: {flashcard_data.get('topic', 'N/A')}")

            # Check cancellation before yielding
            if is_cancelled():
                print(f"🛑 Flashcard generation cancelled after card {card_num} generated")
                return

            # Yield individual flashcard (matching frontend expectations)
            yield {
                "status": "flashcard_ready",
                "flashcard": flashcard_data,
                "index": card_num - 1,
                "total_so_far": card_num
            }

    # All flashcards complete
    yield {
        "status": "flashcard_complete",
        "flashcard_data": None,  # Frontend will use accumulated cards
        "total_generated": num_cards
    }


async def _generate_single_flashcard(
    content: str,
    topic: str,
    card_num: int,
    language: str,
    cards_to_avoid: list = None
) -> dict:
    """
    Generate ONE complete flashcard using LLM.
    Returns fully-formed flashcard object matching frontend expectations.

    Frontend expects:
    {
        "front": "Question or term",
        "back": "Answer or definition",
        "topic": "Category/subject",
        "hint": "Optional hint" (optional)
    }

    Args:
        content: Context from documents or general knowledge
        topic: Main topic for the flashcard
        card_num: Current card number
        language: User's language preference
        cards_to_avoid: List of previous card fronts for deduplication

    Returns:
        Flashcard dictionary matching frontend format
    """

    # Defensive defaults
    if cards_to_avoid is None:
        cards_to_avoid = []

    # Build deduplication text
    if cards_to_avoid:
        avoid_text = "\n".join([f"- {front}" for front in cards_to_avoid])
    else:
        avoid_text = "None - this is the first flashcard"

    print(f"\n{'='*60}")
    print(f"Generating Flashcard {card_num}")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    prompt = PromptTemplate(
        input_variables=["content", "topic", "card_num", "language", "cards_to_avoid"],
        template="""
You are a {language}-speaking educational flashcard generator.

Generate **EXACTLY ONE high-quality flashcard** about: {topic}

Card number: {card_num}

CRITICAL - DO NOT repeat these flashcard fronts:
{cards_to_avoid}

Context:
{content}

Requirements:
- Front: A clear, focused question or term to test (keep concise, 1-2 sentences max)
- Back: A comprehensive, accurate answer or definition (2-4 sentences)
- Topic: A specific 2-4 word category label in {language} (e.g., "Heart Anatomy", "Anatomie Cardiaque")
- Hint: Optional subtle hint without giving away the answer (1 sentence, optional)

Flashcard Design Principles:
- Front should test ONE specific concept clearly
- Back should provide complete, memorable explanation
- Use active recall - front should prompt retrieval, not recognition
- Include relevant clinical context or examples in the back
- Make it practical and applicable to real situations
- Keep language clear and accessible

🎯 TOPIC ASSIGNMENT:
- Assign a SPECIFIC topic/subject to this flashcard
- The topic should be 2-4 words maximum
- Be specific and descriptive (e.g., "Cardiac Medications" not "Medicine")
- CRITICAL: Write the topic in {language} (same language as the flashcard)
- Examples in English:
  * "Heart Anatomy"
  * "Blood Pressure"
  * "Wound Assessment"
  * "Pain Management"
- Examples in French:
  * "Anatomie Cardiaque"
  * "Pression Artérielle"
  * "Évaluation des Plaies"
  * "Gestion de la Douleur"

📤 Return ONLY valid JSON (no markdown wrapper):
{{
    "front": "Clear question or term in {language}",
    "back": "Comprehensive answer or definition in {language}",
    "topic": "Specific Topic Name",
    "hint": "Optional subtle hint (can be omitted if not needed)"
}}

Generate your flashcard in {language}:"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "content": content,
            "topic": topic,
            "card_num": card_num,
            "language": language,
            "cards_to_avoid": avoid_text
        })

        # Clean and parse
        cleaned = result.strip().strip("```json").strip("```").strip()
        parsed_flashcard = json.loads(cleaned)

        # Validate required fields
        if 'front' not in parsed_flashcard or 'back' not in parsed_flashcard:
            print(f"❌ Flashcard {card_num} missing required fields")
            return None

        # Ensure topic field exists
        if 'topic' not in parsed_flashcard or not parsed_flashcard['topic']:
            parsed_flashcard['topic'] = topic if topic else "General"
            print(f"⚠️ Topic field missing, assigned: {parsed_flashcard['topic']}")
        else:
            print(f"✅ Topic assigned: {parsed_flashcard['topic']}")

        # Hint is optional, so it's okay if it's missing
        if 'hint' not in parsed_flashcard:
            parsed_flashcard['hint'] = None

        return parsed_flashcard

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse flashcard {card_num}: {e}")
        if 'result' in locals():
            print(f"Raw output: {result[:500]}...")
        return None
    except Exception as e:
        print(f"❌ Error generating flashcard {card_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Import datetime for timestamps
from datetime import datetime
