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
    name_last_document_used:str = field(default="")
    message_history: List[tuple] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    studysheet_history: str = field(default="")