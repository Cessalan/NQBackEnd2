from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class Document(BaseModel):
    filename: str
    source: str

class StatelessChatRequest(BaseModel):
    language: str = "fr"
    chat_id: str
    input: str
    chat_history: List[Message]
    documents: List[Document]

class QuizRequest(BaseModel):
    chat_id: str
    filename: Optional[str] = None
    quiz_type: str = "mcq"
    num_questions: int = 4
    language: str = "fr"

class ScratchQuizRequest(BaseModel):
    chat_id: str
    topic: str
    difficulty: str = "medium"
    num_questions: int = 4
    quiz_type: str = "mcq"
    language: str = "fr"

class SummaryRequest(BaseModel):
    chat_id: str
    filename: str
    language: str = "fr"

class DocumentsEmbedRequest(BaseModel):
    chatId: str  # Note: camelCase for frontend compatibility
    documents: List[Document]
    
    
class GenerateTitleRequest(BaseModel):
    message:str


class PlanRequest(BaseModel):
    topic: str
    chat_id: str
    num_sections: Optional[int] = 6
    
class SectionRequest(BaseModel):
    section_title: str
    topic: str
    chat_id: str
    context: str


# ============================================================================
# STUDY MODE REQUESTS
# These support the Duolingo-style study journey feature
# ============================================================================

class StudyPlanRequest(BaseModel):
    """
    Request to generate a personalized study path.

    The AI analyzes uploaded documents and creates a learning path
    with different node types: lessons, flashcards, quizzes, and audio.
    """
    chat_id: str                          # Chat ID where documents were uploaded
    upload_ids: Optional[List[str]] = []  # Optional: specific upload IDs to focus on
    language: str = "en"                  # Language for content generation


class StudyItemRequest(BaseModel):
    """
    Request to generate content for a single study node.

    Content is generated on-demand when user clicks a node,
    not all at once (saves cost, feels more dynamic).
    """
    chat_id: str                          # Chat ID for context
    node_type: str                        # "lesson" | "flashcard" | "quiz" | "audio"
    node_label: str                       # Topic/label for this node (e.g., "Cardiac Medications")
    context_tags: Optional[List[str]] = []  # Tags for better context
    asked_hashes: Optional[List[str]] = []  # Previously shown content hashes (anti-repeat)
    language: str = "en"                  # Language for content
