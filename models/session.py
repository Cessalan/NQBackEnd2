from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class PersistentSessionContext:
    """Session state that persists across messages"""
    chat_id: str
    user_language: Optional[str] = None
    
    quiz_params: Dict[str, Any] = field(default_factory=lambda: {
        "topic": None,
        "difficulty": None,
        "num_questions": 4,
        "quiz_type": "mcq",
        "source": None
    })
    
    vectorstore: Any = None
    documents: List[Any] = field(default_factory=list)
    quizzes:List[Any] = field(default_factory=list)
    name_last_document_used:str = field(default="")
    message_history: List[tuple] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    studysheet_history: str = field(default="")
    # FIX: Changed from class-level `= []` to instance-level `field(default_factory=list)`
    # The old code shared one list across ALL sessions, causing cross-user contamination
    # in question deduplication. Now each session has its own isolated list.
    previously_generate_questions_in_quiz: List[str] = field(default_factory=list)
    file_insights: Dict[str, Dict] = field(default_factory=dict)  # {filename: {topics, concepts, doc_type}}