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
    # Track existing topics to avoid duplicates/variations
    existing_topics = []

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
            cards_to_avoid=generated_fronts,
            existing_topics=existing_topics
        )

        if flashcard_data:
            generated_fronts.append(flashcard_data['front'])

            # Track the topic for future flashcards (avoid duplicates)
            card_topic = flashcard_data.get('topic', '')
            if card_topic and card_topic not in existing_topics:
                existing_topics.append(card_topic)

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
    cards_to_avoid: list = None,
    existing_topics: list = None
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
        existing_topics: List of topics already used in this flashcard set

    Returns:
        Flashcard dictionary matching frontend format
    """

    # Defensive defaults
    if cards_to_avoid is None:
        cards_to_avoid = []
    if existing_topics is None:
        existing_topics = []

    # Build deduplication text
    if cards_to_avoid:
        avoid_text = "\n".join([f"- {front}" for front in cards_to_avoid])
    else:
        avoid_text = "None - this is the first flashcard"

    # Build existing topics text for smart topic matching
    if existing_topics:
        existing_topics_text = ", ".join([f'"{t}"' for t in existing_topics])
    else:
        existing_topics_text = "None yet - you are creating the first topic"

    print(f"\n{'='*60}")
    print(f"Generating Flashcard {card_num}")
    print(f"Topic: {topic}")
    print(f"Existing topics: {existing_topics}")
    print(f"{'='*60}\n")

    prompt = PromptTemplate(
        input_variables=["content", "topic", "card_num", "language", "cards_to_avoid", "existing_topics"],
        template="""
You are creating flashcard {card_num} in {language}.

🚨 CRITICAL: USE ONLY DOCUMENT CONTENT - NO HALLUCINATION!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 ALREADY ASKED - DO NOT REPEAT THESE CONCEPTS:
{cards_to_avoid}

⚠️ YOU MUST ASK ABOUT A DIFFERENT CONCEPT!
- Find a NEW term/definition not in the list above
- Each flashcard must test a UNIQUE piece of information
- If you repeat a concept, the flashcard will be rejected

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ULTRA-SHORT FORMAT (MAX 40 WORDS!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FRONT: One clear question (max 12 words)
BACK: 2-3 bullet points, 5-8 words each

Example:
FRONT: "Qu'est-ce que la bradycardie?"
BACK: "• **Bradycardie** = FC < 60 bpm
• Cause: bloc cardiaque, médicaments"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📤 Return ONLY JSON:
{{
    "front": "Short question about NEW concept",
    "back": "• **Term** = brief definition",
    "topic": "{existing_topics}",
    "hint": null
}}"""
    )

    # Use gpt-4.1-nano for flashcard generation (cost-effective for structured JSON output)
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "content": content,
            "topic": topic,
            "card_num": card_num,
            "language": language,
            "cards_to_avoid": avoid_text,
            "existing_topics": existing_topics_text
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
