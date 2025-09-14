# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your models
from models.requests import StatelessChatRequest
from models.session import PersistentSessionContext

# Import your orchestrator
from services.orchestrator import NursingTutor

# Import Firebase initialization
import firebase_admin
from firebase_admin import credentials

# Initialize Firebase
cred = credentials.Certificate("FireBaseAccess.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": os.getenv("FIREBASE_BUCKET", "docai-efb03.firebasestorage.app")
})

# Create FastAPI app
app = FastAPI(
    title="Nursing Tutor AI",
    version="1.0.0",
    description="AI-powered nursing education assistant with tool calling"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://docai-efb03.web.app",
        "https://docai-efb03.firebaseapp.com",
        "https://chats.nursequizai.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session storage
ACTIVE_SESSIONS: Dict[str, NursingTutor] = {}

# ============================================================================
# MAIN CHAT ENDPOINT WITH TOOL CALLING
# ============================================================================

@app.post("/chat/stream")
async def chat_stream_response(request: StatelessChatRequest):
    """
    Main chat endpoint - handles all conversations with tool calling
    """
    
    # Get or create session for this chat
    if request.chat_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
        print(f"Created new session for chat_id: {request.chat_id}")
    
    # GET TUTOR FOR CURRENT SESSION
    nursing_tutor = ACTIVE_SESSIONS[request.chat_id]
    
    # Update session with documents if provided
    if request.documents:
        nursing_tutor.session.documents = request.documents
    
    # Process message and stream response
    return StreamingResponse(
        nursing_tutor.process_message(
            # feed the tutor the user input
            user_input=request.input,
            # feed the tutor the chat history
            chat_history=request.chat_history,
            # feed the tutor the language the user's browser
            language=request.language
        ),
        media_type="application/json"
    )

# ============================================================================
# SUPPORTING ENDPOINTS (keep your existing ones)
# ============================================================================

@app.post("/chat/embed")
async def embed_documents(request):
    """Your existing document embedding logic"""
    # Keep your existing implementation
    pass

@app.post("/chat/generate-quiz")
async def generate_quiz(request):
    """Your existing quiz generation"""
    # Keep your existing implementation
    pass

@app.post("/chat/generate-title")
async def generate_title(request):
    """Your existing title generation"""
    # Keep your existing implementation
    pass

# ============================================================================
# ADMIN/DEBUG ENDPOINTS (optional but helpful)
# ============================================================================

@app.get("/admin/sessions")
async def list_sessions():
    """View all active sessions"""
    return {
        "total_sessions": len(ACTIVE_SESSIONS),
        "sessions": [
            {
                "chat_id": chat_id,
                "language": session.session.user_language,
                "has_vectorstore": session.session.vectorstore is not None,
                "message_count": len(session.session.message_history),
                "quiz_params": session.session.quiz_params
            }
            for chat_id, session in ACTIVE_SESSIONS.items()
        ]
    }

@app.delete("/admin/sessions/{chat_id}")
async def clear_session(chat_id: str):
    """Clear a specific session"""
    if chat_id in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[chat_id]
        return {"message": f"Session {chat_id} cleared"}
    return {"message": "Session not found"}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Nursing Tutor AI",
        "sessions_active": len(ACTIVE_SESSIONS)
    }

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )