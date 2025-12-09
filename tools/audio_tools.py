# Audio Tools - Generate audio lectures and summaries
# Uses OpenAI TTS with intent-based content generation

from typing import Dict, Any
from langchain_core.tools import tool


@tool
async def generate_audio_content(
    topic: str,
    intent: str = "teach",
    duration: str = None,
    language: str = None
) -> Dict[str, Any]:
    """
    Generate audio content (lectures, summaries, explanations) based on user intent.

    INTENT DETECTION - Use this tool when user wants AUDIO content:
    - "teach me about X", "explain X to me", "tell me about X" → intent="teach"
    - "summarize X", "quick overview of X", "key points of X" → intent="summarize"
    - "deep dive into X", "everything about X", "detailed explanation" → intent="deep_dive"
    - "explain X simply", "like I'm 5", "basics of X" → intent="simplify"
    - "how am I doing", "my progress", "my stats" → intent="progress"

    Args:
        topic: The subject for the audio content
        intent: Type of audio - "teach", "summarize", "deep_dive", "simplify", "progress"
        duration: Optional duration - will show options if not provided
        language: Language for the audio - "english" or "french" (use "french" if user wrote in French or specified French)

    Returns:
        If duration not provided: Returns options for user to select
        If duration provided: Signals to start audio generation
    """
    from tools.quiztools import get_session

    try:
        session = get_session()

        # Intent configurations
        intent_config = {
            "teach": {
                "name": "Full Lesson",
                "description": "Structured lesson with examples and clinical context",
                "durations": ["2min", "5min", "10min"],
                "default_duration": "5min"
            },
            "summarize": {
                "name": "Quick Summary",
                "description": "Key points only, concise overview",
                "durations": ["1min", "2min", "3min"],
                "default_duration": "2min"
            },
            "deep_dive": {
                "name": "Deep Dive",
                "description": "Comprehensive, detailed exploration",
                "durations": ["5min", "10min", "15min"],
                "default_duration": "10min"
            },
            "simplify": {
                "name": "Simple Explanation",
                "description": "Beginner-friendly, uses analogies",
                "durations": ["2min", "3min", "5min"],
                "default_duration": "3min"
            },
            "progress": {
                "name": "Progress Report",
                "description": "Your stats, achievements, and recommendations",
                "durations": ["1min", "2min"],
                "default_duration": "2min"
            }
        }

        config = intent_config.get(intent, intent_config["teach"])

        # If no duration provided, return options for user to select
        if not duration:
            return {
                "status": "audio_options",
                "topic": topic,
                "intent": intent,
                "style_name": config["name"],
                "style_description": config["description"],
                "durations": config["durations"],
                "default_duration": config["default_duration"],
                "message": f"Ready to create a {config['name'].lower()} about {topic}. Select duration to continue."
            }

        # Duration provided - signal to start generation
        # Use explicit language param if provided, otherwise fall back to session language
        audio_language = language or session.user_language or "english"

        return {
            "status": "audio_generation_initiated",
            "metadata": {
                "topic": topic,
                "intent": intent,
                "duration": duration,
                "style_name": config["name"],
                "language": audio_language,
                "chat_id": session.chat_id,
                "has_documents": bool(session.documents)
            },
            "message": f"Generating {duration} {config['name'].lower()} about {topic}..."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Audio generation setup failed: {str(e)}"
        }


@tool
async def generate_progress_audio() -> Dict[str, Any]:
    """
    Generate an audio summary of the student's progress.

    Use this when user asks:
    - "How am I doing?"
    - "Summarize my progress"
    - "What's my status?"
    - "Tell me about my performance"

    Returns:
        Progress data and signal to generate audio
    """
    from tools.quiztools import get_session

    try:
        session = get_session()

        # Gather progress data from session/Firebase
        progress_data = await _get_progress_data(session)

        return {
            "status": "audio_generation_initiated",
            "metadata": {
                "topic": "Your Learning Progress",
                "intent": "progress",
                "duration": "2min",
                "style_name": "Progress Report",
                "language": session.user_language or "english",
                "chat_id": session.chat_id,
                "progress_data": progress_data
            },
            "message": "Generating your personalized progress report..."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Progress audio failed: {str(e)}"
        }


async def _get_progress_data(session) -> dict:
    """Gather progress data for progress report audio"""
    try:
        from firebase_admin import firestore
        db = firestore.client()

        chat_id = session.chat_id
        progress_data = {
            "level": 1,
            "total_xp": 0,
            "xp_needed": 100,
            "streak": 0,
            "daily_correct": 0,
            "daily_goal": 5,
            "recent_quizzes": "No quizzes taken yet",
            "topics_studied": "None yet"
        }

        # Try to get progress from Firebase
        # This would integrate with your ProgressContext on frontend
        # For now, extract from quiz history

        quizzes = getattr(session, 'quizzes', [])
        if quizzes:
            recent = quizzes[-5:]  # Last 5 quizzes

            quiz_summaries = []
            topics = set()
            total_correct = 0
            total_questions = 0

            for quiz in recent:
                quiz_data = quiz.get('quiz_data', {})
                questions = quiz_data.get('questions', []) if isinstance(quiz_data, dict) else []

                for q in questions:
                    total_questions += 1
                    topic = q.get('topic', '')
                    if topic:
                        topics.add(topic)
                    if q.get('userSelection', {}).get('isCorrect'):
                        total_correct += 1

            if total_questions > 0:
                accuracy = round((total_correct / total_questions) * 100)
                progress_data["recent_quizzes"] = f"{accuracy}% accuracy on last {len(recent)} quizzes ({total_correct}/{total_questions} correct)"

            if topics:
                progress_data["topics_studied"] = ", ".join(list(topics)[:5])

        return progress_data

    except Exception as e:
        print(f"Error getting progress data: {e}")
        return {
            "level": 1,
            "total_xp": 0,
            "streak": 0,
            "recent_quizzes": "Unable to load quiz history",
            "topics_studied": "Unable to load topics"
        }
