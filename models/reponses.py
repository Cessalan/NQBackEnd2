from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QuizResponse(BaseModel):
    status: str
    quiz: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    answer_chunk: Optional[str] = None
    message: Optional[str] = None

class GenerateTitleResponse(BaseModel):
    title: str