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


class RewriteRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


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
    userPreferences: Optional[dict] = {}  # Onboarding preferences (reviewFormat, userStage, etc)


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


class StudyAudioRequest(BaseModel):
    """Request to generate audio for a study node"""
    chat_id: str
    topic: str
    intent: str = "teach"
    duration: int = 2  # Duration in minutes
    language: str = "en"


class StudyReviewPlanRequest(BaseModel):
    """
    Request to generate a Phase 2 review study path based on performance data.
    Frontend sends performance from Firestore since backend has no Firebase auth.
    """
    chat_id: str
    language: str = "en"
    performance: dict = {}                     # Full studyPerformance doc from Firestore
    original_topics: Optional[List[str]] = []  # Topics from phase 1 for context


class StudyExamRequest(BaseModel):
    """
    Generate a mixed-format exam for a study session.
    Supports MCQ, SATA, and Case Study question types.
    """
    chat_id: str
    topic: str                                     # Topic this exam covers
    question_types: List[str] = ["mcq", "sata", "casestudy"]
    question_count: int = 10
    custom_instructions: Optional[str] = None      # Student's custom instructions
    language: str = "en"


class StudyInterpretRequest(BaseModel):
    """
    Interpret a student's free-text request during a study session.
    Returns an echo message (what the system understood) and a node definition.
    The student confirms before the node is created.
    """
    chat_id: str
    user_text: str                                # What the student typed
    current_topic: str = ""                       # Topic of the node she just completed
    current_node_type: str = ""                   # Type of the node she just completed
    language: str = "en"
    missed_items: Optional[List[str]] = []        # Specific questions/cards she got wrong
    score_percent: Optional[int] = None           # Her score on the node she just completed


class StudyMindmapRequest(BaseModel):
    """Request to generate a concept map for a study node"""
    chat_id: str
    topic: str
    depth: str = "medium"  # shallow | medium | deep
    language: str = "en"


class DiagnosticQuizRequest(BaseModel):
    """
    Request to generate 5 breadth-first diagnostic questions.
    Called before showing the study plan to establish baseline proficiency.
    One question per major topic, varying difficulty easy→medium.
    """
    chat_id: str
    upload_ids: Optional[List[str]] = []
    language: str = "en"
