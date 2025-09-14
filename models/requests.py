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