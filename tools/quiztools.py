from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool
from models.session import PersistentSessionContext
import json

# Global session access - this will be injected by NursingTutor
_CURRENT_SESSION: Optional[PersistentSessionContext] = None

def set_session_context(session: PersistentSessionContext):
    """Set the current session context for tool access"""
    global _CURRENT_SESSION
    _CURRENT_SESSION = session

def get_session() -> PersistentSessionContext:
    """Get current session context"""
    if _CURRENT_SESSION is None:
        raise RuntimeError("No session context available. This is a system error.")
    return _CURRENT_SESSION

# ============================================================================
# FIXED TOOLS WITH PROPER DECORATORS
# ============================================================================

@tool
async def search_documents(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search uploaded nursing documents for relevant information.
    
    Use this when students ask questions about their uploaded study materials,
    textbooks, or need specific information from their documents.
    
    Args:
        query: What to search for (e.g., "cardiac medications", "NCLEX tips")
        max_results: Maximum number of document chunks to return (default: 3)
    
    Returns:
        Dictionary with search results and context
    """
    try:
        session = get_session()
        
        # Load vectorstore if not already loaded
        if session.vectorstore is None and session.documents:
            session.vectorstore = await load_vectorstore_from_firebase(session)
            session.vectorstore_loaded = True
        
        if session.vectorstore:
            # Search for relevant content
            docs = session.vectorstore.similarity_search(query, k=max_results)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Cache the retrieval
            session.last_retrieval = context
            
            # Record tool call
            session.tool_calls.append({
                "tool": "search_documents",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "chunks_found": len(docs)
            })
            
            return {
                "status": "success",
                "context": context,
                "num_chunks": len(docs),
                "message": f"Found {len(docs)} relevant sections in your documents."
            }
        
        return {
            "status": "no_documents",
            "message": "No documents available for search. Ask the student to upload study materials first."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Document search failed: {str(e)}"
        }

@tool
async def generate_quiz(
    topic: str, 
    difficulty: str = "medium", 
    num_questions: int = 4,
    source_preference: str = "auto"
) -> Dict[str, Any]:
    """
    Generate a nursing quiz for the student.
    
    Use this when students request practice questions, want to test their knowledge,
    or need NCLEX-style questions on specific topics.
    
    Args:
        topic: Subject area (e.g., "pharmacology", "cardiac care", "NCLEX prep")
        difficulty: Question difficulty ("easy", "medium", "hard")
        num_questions: Number of questions to generate (1-10, default: 4)
        source_preference: "documents" (from uploads), "scratch" (general), or "auto"
    
    Returns:
        Dictionary with quiz questions, answers, and rationales
    """
    try:
        session = get_session()
        
        # Normalize difficulty
        difficulty_map = {
            "facile": "easy", "easy": "easy",
            "moyen": "medium", "medium": "medium", "normal": "medium",
            "difficile": "hard", "hard": "hard", "challenging": "hard"
        }
        normalized_difficulty = difficulty_map.get(difficulty.lower(), difficulty)
        
        # Determine source
        if source_preference == "auto":
            source = "documents" if session.documents else "scratch"
        else:
            source = source_preference
        
        # Validate num_questions
        num_questions = max(1, min(10, num_questions))
        
        # Generate quiz
        if source == "documents" and session.vectorstore:
            quiz_data = await _generate_from_documents(
                topic=topic,
                difficulty=normalized_difficulty,
                num_questions=num_questions,
                session=session
            )
        else:
            quiz_data = await _generate_from_scratch(
                topic=topic,
                difficulty=normalized_difficulty,
                num_questions=num_questions,
                session=session
            )
        
        # Cache the generated quiz
        session.last_quiz_generated = quiz_data
        
        # Record tool call
        session.tool_calls.append({
            "tool": "generate_quiz",
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": normalized_difficulty,
            "num_questions": num_questions,
            "source": source,
            "status": "generated"
        })
        
        return {
            "status": "success",
            "quiz": quiz_data,
            "metadata": {
                "topic": topic,
                "difficulty": normalized_difficulty,
                "question_count": num_questions,
                "source": source
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Quiz generation failed: {str(e)}"
        }

@tool
def check_student_progress() -> Dict[str, Any]:
    """
    Check the current student's learning session and progress.
    
    Use this to understand what the student has been working on,
    what materials they have available, and their recent activity.
    
    Returns:
        Dictionary with session information and learning progress
    """
    try:
        session = get_session()
        
        return {
            "status": "success",
            "session_info": {
                "chat_id": session.chat_id,
                "language": session.user_language,
                "has_documents": bool(session.documents),
                "document_count": len(session.documents) if session.documents else 0,
                "vectorstore_loaded": session.vectorstore_loaded,
                "message_count": len(session.message_history),
                "recent_topics": _extract_recent_topics(session),
                "recent_tool_usage": [
                    {k: v for k, v in call.items() if k != 'context'}
                    for call in session.tool_calls[-5:]
                ] if session.tool_calls else [],
                "has_cached_materials": session.last_retrieval is not None,
                "last_quiz_available": session.last_quiz_generated is not None
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Progress check failed: {str(e)}"
        }

# ============================================================================
# HELPER FUNCTIONS (keep your existing implementation)
# ============================================================================

async def _generate_from_documents(topic: str, difficulty: str, num_questions: int, session: PersistentSessionContext):
    """Generate quiz from uploaded documents - implement your existing logic"""
    # TODO: Implement your existing document-based quiz generation
    # This should use the vectorstore to find relevant content and create questions
    pass

async def _generate_from_scratch(topic: str, difficulty: str, num_questions: int, session: PersistentSessionContext):
    """Generate quiz from scratch - implement your existing logic"""
    # TODO: Implement your existing scratch quiz generation
    # This should use your LLM chain to create questions on the topic
    pass

def _extract_recent_topics(session: PersistentSessionContext) -> list:
    """Extract topics from recent messages and tool calls"""
    topics = []
    
    # Get topics from recent tool calls
    for call in session.tool_calls[-5:]:
        if call.get("tool") == "generate_quiz" and "topic" in call:
            topics.append(call["topic"])
    
    return list(set(topics))  # Remove duplicates

async def load_vectorstore_from_firebase(session: PersistentSessionContext):
    """Your existing vectorstore loading logic"""
    # TODO: Keep your existing implementation
    pass

# ============================================================================
# TOOL COLLECTION CLASS (Updated)
# ============================================================================

class NursingTools:
    """Collection of tools for the nursing tutor - Updated for LangChain integration"""
    
    def __init__(self, session: PersistentSessionContext):
        self.session = session
        # Set global session context for tools to access
        set_session_context(session)
    
    def get_tools(self):
        """Return list of tools for LangChain binding"""
        return [
            search_documents,
            generate_quiz,
            check_student_progress
        ]