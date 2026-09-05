# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi import HTTPException, Request
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import os
from dotenv import load_dotenv

import asyncio
import json
import re
import re as _lang_re
# Load environment variables. `.env.local` (if present) OVERRIDES `.env` —
# it holds local sandbox credentials (e.g. Stripe test keys) and is excluded
# from git AND from the Docker image, so a deployed backend can never pick it
# up. Delete or empty it to go back to the real `.env` values locally.
load_dotenv()
load_dotenv(".env.local", override=True)


# ═══════════════════════════════════════════════════════════════════════════
# Session-language helpers
# ═══════════════════════════════════════════════════════════════════════════
# session.user_language is detected once per session and cached (LLM call is
# ~500-1000ms). But a hard cache caused a real failure: one pasted foreign
# snippet as the first message locked the whole session to that language, and
# "IN ENGLISH" replies were ignored. These helpers catch the two cheap,
# unambiguous signals that must beat the cache:
#   1. An explicit inline request ("in english", "en français").
#   2. A script mismatch (cached russian but the message is Latin-script).
# Only when one of those fires do we pay for re-detection.

_EXPLICIT_LANGUAGE_REQUESTS = (
    (_lang_re.compile(r'\b(in|into|to)\s+english\b|\banglais\b', _lang_re.I), 'english'),
    (_lang_re.compile(r'\ben\s+fran[çc]ais\b|\b(in|into|to)\s+french\b', _lang_re.I), 'french'),
    (_lang_re.compile(r'\ben\s+espa[ñn]ol\b|\b(in|into|to)\s+spanish\b', _lang_re.I), 'spanish'),
    (_lang_re.compile(r'\bem\s+portugu[êe]s\b|\b(in|into|to)\s+portuguese\b', _lang_re.I), 'portuguese'),
    (_lang_re.compile(r'\bна\s+русском\b|\b(in|into|to)\s+russian\b', _lang_re.I), 'russian'),
)

# Languages whose dominant script is not Latin. Anything unlisted → latin.
_NON_LATIN_LANGUAGE_SCRIPTS = {
    'russian': 'cyrillic', 'ukrainian': 'cyrillic', 'bulgarian': 'cyrillic',
    'arabic': 'arabic', 'farsi': 'arabic', 'persian': 'arabic', 'urdu': 'arabic',
    'chinese': 'cjk', 'japanese': 'cjk', 'korean': 'cjk',
    'greek': 'greek', 'hindi': 'devanagari',
}


def explicit_language_request(text: str):
    """Language explicitly asked for inline, or None."""
    for pattern, lang in _EXPLICIT_LANGUAGE_REQUESTS:
        if pattern.search(text or ""):
            return lang
    return None


def _dominant_script(text: str):
    """Rough dominant script of a message, or None when too few letters."""
    counts = {'latin': 0, 'cyrillic': 0, 'arabic': 0, 'cjk': 0, 'greek': 0, 'devanagari': 0}
    for ch in text or "":
        o = ord(ch)
        if ('a' <= ch.lower() <= 'z') or (0x00C0 <= o <= 0x024F):
            counts['latin'] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts['cyrillic'] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts['arabic'] += 1
        elif (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            counts['cjk'] += 1
        elif 0x0370 <= o <= 0x03FF:
            counts['greek'] += 1
        elif 0x0900 <= o <= 0x097F:
            counts['devanagari'] += 1
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return script if n >= 10 else None


def cached_language_conflicts(cached_language: str, text: str) -> bool:
    """True when the message's script contradicts the cached session language."""
    script = _dominant_script(text)
    if script is None:
        return False
    expected = _NON_LATIN_LANGUAGE_SCRIPTS.get((cached_language or '').lower(), 'latin')
    return script != expected

# Import your models
from models.requests import (
    StatelessChatRequest, DocumentsEmbedRequest, SummaryRequest, SectionRequest, PlanRequest,
)

# Import your orchestrator
from services.orchestrator import NursingTutor

# Import Firebase initialization
import firebase_admin
from firebase_admin import credentials

# embedding
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# to store files temporarily
import tempfile
import os
import platform
import requests

# models
from models.reponses import GenerateTitleResponse, RewriteResponse
from models.requests import GenerateTitleRequest, RewriteRequest

# document loader
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader)

from fastapi import File, UploadFile, Form
from typing import List
from uuid import uuid4
import mimetypes

from core.imageloader import OCRImageLoader
from core.pdfloader import OCRPDFLoader, is_scanned_pdf

# langchain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# firebase access
import firebase_admin
from firebase_admin import credentials,storage

from core.language import LanguageDetector

from services.vectorstore_manager import vectorstore_manager
from constants.nursing_frameworks import detect_frameworks

# Study mode imports - reuse existing streaming generators
from models.session import PersistentSessionContext
from tools.quiztools import set_session_context, get_session, load_files_for_chat, get_chat_vectorstore
from services.quiz_with_bank import stream_quiz_with_bank
from tools.flashcard_tools import stream_flashcards

import json
import hashlib

# Custom PowerPoint loader that doesn't need NLTK
class SimplePowerPointLoader:
    """Lightweight PowerPoint loader using python-pptx directly."""
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self):
        """Extract text from PowerPoint slides."""
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError:
            raise ImportError("python-pptx is required. Install: pip install python-pptx")
        
        prs = Presentation(self.file_path)
        documents = []
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            text_content = []
            
            # Extract text from all shapes in the slide
            for shape in slide.shapes:
                # Regular text shapes
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text.strip())
                
                # Handle tables - use shape_type to check safely
                if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    try:
                        table = shape.table
                        for row in table.rows:
                            row_text = " | ".join(
                                cell.text.strip() for cell in row.cells if cell.text.strip()
                            )
                            if row_text:
                                text_content.append(row_text)
                    except Exception as e:
                        print(f"⚠️ Failed to extract table: {e}")
                        continue
                
                # Handle grouped shapes (recursively extract text)
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    try:
                        for sub_shape in shape.shapes:
                            if hasattr(sub_shape, "text") and sub_shape.text.strip():
                                text_content.append(sub_shape.text.strip())
                    except Exception as e:
                        print(f"⚠️ Failed to extract grouped shape: {e}")
                        continue
            
            # Create document if there's content
            if text_content:
                page_content = "\n\n".join(text_content)
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": os.path.basename(self.file_path),
                        "page": slide_num,
                        "total_slides": len(prs.slides)
                    }
                )
                documents.append(doc)
        
        print(f"✅ Extracted text from {len(documents)} slides")
        return documents

# to load documents
def get_loader_for_file(path):
    ext = os.path.splitext(path)[-1].lower()        
    if ext == ".pdf":# pdf file support
        #Detect if PDF is scanned or text-based
        if is_scanned_pdf(path):
            print("📷 Detected scanned PDF - Using OCR")
            return OCRPDFLoader(path)
        else:
            print("📄 Detected text-based PDF - Using standard loader")
            return PyPDFLoader(path)
    elif ext == ".txt": # txt file support
        return TextLoader(path, encoding="utf-8")
    elif ext == ".csv":  #excel support
        return CSVLoader(path, encoding="utf-8")
    elif ext in [".doc", ".docx"]: # word document support
        return Docx2txtLoader(path) 
    elif ext in [".xls", ".xlsx"]: #excel support
        return UnstructuredExcelLoader(path) 
    elif ext in [".ppt", ".pptx"]: # power point support
        return SimplePowerPointLoader(path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",".heic"]:  # Add image support
        print("Extracting text from image")
        return OCRImageLoader(path)
    else:
        raise ValueError("Unsupported file type")

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("FireBaseAccess.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": os.getenv("FIREBASE_BUCKET", "docai-efb03.firebasestorage.app")
    })

# cred = credentials.Certificate("service-account-key.json")  # Path to key file
#     firebase_admin.initialize_app(cred, {
#         'storageBucket': 'docai-efb03.firebasestorage.app'
# })


# Set Google Cloud credentials to the Firebase service account file
# This enables Google Cloud TTS and other Google Cloud APIs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "FireBaseAccess.json"
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ OPENAI_API_KEY found")
    print(f"   Length: {len(api_key)}")
    print(f"   First 15 chars: {api_key[:15]}")
    print(f"   Last 4 chars: ...{api_key[-4:]}")
    print(f"   Type: {type(api_key)}")
    # Move the check outside the f-string
    has_whitespace = ' ' in api_key or '\n' in api_key or '\t' in api_key
    print(f"   Contains whitespace: {has_whitespace}")
else:
    print("❌ OPENAI_API_KEY is NOT SET")

# ============================================================================
# COST OPTIMIZATION: Lifespan context manager for startup/shutdown
# ============================================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    asyncio.create_task(periodic_cleanup())
    print("🚀 Started periodic cleanup task (30s interval, 60s timeout)")
    yield
    # Shutdown - cleanup all sessions
    print("🛑 Shutting down - cleaning up all sessions...")
    for chat_id in list(ACTIVE_SESSIONS.keys()):
        cleanup_session(chat_id)

# Create FastAPI app
app = FastAPI(
    title="Nursing Tutor AI",
    version="1.0.0",
    description="AI-powered nursing education assistant with tool calling",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://docai-efb03.web.app",
        "https://docai-efb03.firebaseapp.com",
        "https://chats.nursequizai.com",
        "https://nursequizai.com",
        "https://ragfastapi-1075876064685.europe-west1.run.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Stripe billing webhook — grants/revokes Pro (users/{uid}.usage.tier)
# ============================================================================
from services import stripe_billing
# Server-side free-tier quota check (mirrors the client gate in UsageService.js)
from services import usage_guard

@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe sends signed events here. We verify the signature (raw body required)
    and flip the user's tier in Firestore. Configure the endpoint URL + signing
    secret in the Stripe dashboard; set STRIPE_WEBHOOK_SECRET in the env.
    """
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    event, err = stripe_billing.verify_and_parse(payload, sig)
    if err:
        # 400 tells Stripe to retry (except signature/secret issues, which are
        # config problems on our side — still surfaced for visibility).
        raise HTTPException(status_code=400, detail=err)
    result = stripe_billing.handle_event(event)
    print(f"💳 Stripe webhook: {event['type']} -> {result}")
    return result

@app.post("/billing/create-portal-session")
async def create_billing_portal_session(request: Request):
    """
    Open the Stripe Customer Billing Portal for the signed-in user (where they
    can cancel or change the Pro subscription). The uid comes from a verified
    Firebase ID token — never from the body — so a user can only ever open
    their own portal. Cancellation itself is handled by Stripe; the existing
    webhook downgrades the tier when the subscription ends.
    """
    from firebase_admin import auth as firebase_auth
    authz = request.headers.get("authorization", "")
    token = authz[7:] if authz.lower().startswith("bearer ") else ""
    if not token:
        raise HTTPException(status_code=401, detail="missing_id_token")
    try:
        decoded = firebase_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="invalid_id_token")
    uid = decoded["uid"]

    # Send the user back where they came from; Origin is constrained by CORS.
    return_url = request.headers.get("origin") or "https://chats.nursequizai.com"

    url, err = stripe_billing.create_portal_session(uid, return_url)
    if err == "no_stripe_customer":
        raise HTTPException(status_code=404, detail=err)
    if err:
        raise HTTPException(status_code=500, detail=err)
    return {"url": url}

# Global session storage
ACTIVE_SESSIONS: Dict[str, NursingTutor] = {}
SESSION_LAST_ACTIVITY: Dict[str, float] = {}  # Track last activity time for each session

# ============================================================================
# COST OPTIMIZATION: Session cleanup configuration
# ============================================================================
SESSION_IDLE_TIMEOUT = 120  # 2 minutes - enough for rapid interactions
# WebSocket idle timeout. Bumped from 120s -> 300s because the
# research-grounded quiz pipeline (intent analyzer -> web research ->
# concept extraction -> 10 parallel SATA generations) can legitimately
# run 60-150s end to end. The watchdog now ALSO treats server-sent
# stream chunks as activity (see send_text loop in process_chat_message),
# so this timeout only triggers when the connection is truly silent.
CONNECTION_IDLE_TIMEOUT = 300  # 5 minutes
SESSION_MAX_AGE = 900  # 15 minutes - max session lifetime regardless of activity

import time

def cleanup_session(chat_id: str):
    """Clean up session and free memory"""
    if chat_id in ACTIVE_SESSIONS:
        try:
            session = ACTIVE_SESSIONS[chat_id]
            # Clear vectorstore to free memory
            if hasattr(session, 'session') and hasattr(session.session, 'vectorstore'):
                session.session.vectorstore = None
            del ACTIVE_SESSIONS[chat_id]
            print(f"🧹 Cleaned up session for chat_id: {chat_id}")
        except Exception as e:
            print(f"⚠️ Error cleaning up session {chat_id}: {e}")

    if chat_id in SESSION_LAST_ACTIVITY:
        del SESSION_LAST_ACTIVITY[chat_id]

def update_session_activity(chat_id: str):
    """Update last activity timestamp for a session"""
    SESSION_LAST_ACTIVITY[chat_id] = time.time()

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.cancellation_flags: Dict[str, bool] = {}  # Track if chat should be cancelled
        self.connection_times: Dict[str, float] = {}  # Track when connection was established
        self.last_activity: Dict[str, float] = {}  # Track last message time

    async def connect(self, websocket: WebSocket, chat_id: str):
        await websocket.accept()
        self.active_connections[chat_id] = websocket
        self.cancellation_flags[chat_id] = False  # Reset cancellation flag
        self.connection_times[chat_id] = time.time()
        self.last_activity[chat_id] = time.time()
        update_session_activity(chat_id)
        print(f"✅ WebSocket connected for chat_id: {chat_id}")

    def disconnect(self, chat_id: str):
        if chat_id in self.active_connections:
            del self.active_connections[chat_id]
            print(f"❌ WebSocket disconnected for chat_id: {chat_id}")
        if chat_id in self.cancellation_flags:
            del self.cancellation_flags[chat_id]
        if chat_id in self.connection_times:
            del self.connection_times[chat_id]
        if chat_id in self.last_activity:
            del self.last_activity[chat_id]

        # COST OPTIMIZATION: DON'T cleanup session immediately on disconnect
        # Keep session alive for 60s so user can reconnect without reloading vectorstore
        # The periodic_cleanup task will clean it up if truly idle
        # This saves cost by avoiding repeated Firebase/vectorstore loads
        update_session_activity(chat_id)  # Reset timer on disconnect

    def update_activity(self, chat_id: str):
        """Update last activity time for a connection"""
        self.last_activity[chat_id] = time.time()
        update_session_activity(chat_id)

    def cancel_stream(self, chat_id: str):
        """Mark a chat's stream for cancellation"""
        self.cancellation_flags[chat_id] = True
        print(f"🛑 Stream cancellation requested for chat_id: {chat_id}")

    def is_cancelled(self, chat_id: str) -> bool:
        """Check if a chat's stream has been cancelled"""
        return self.cancellation_flags.get(chat_id, False)

    def reset_cancellation(self, chat_id: str):
        """Reset cancellation flag for a chat"""
        self.cancellation_flags[chat_id] = False

    async def send_message(self, chat_id: str, message: dict):
        if chat_id in self.active_connections:
            try:
                await self.active_connections[chat_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to {chat_id}: {e}")
                self.disconnect(chat_id)

    async def cleanup_idle_connections(self):
        """Close connections that have been idle too long - called periodically"""
        current_time = time.time()
        to_close = []

        for chat_id, last_time in list(self.last_activity.items()):
            idle_seconds = current_time - last_time
            if idle_seconds > CONNECTION_IDLE_TIMEOUT:
                to_close.append(chat_id)
                print(f"⏰ Connection {chat_id} idle for {idle_seconds:.0f}s, closing...")

        for chat_id in to_close:
            try:
                ws = self.active_connections.get(chat_id)
                if ws:
                    await ws.close(1000, "Idle timeout - reconnect when needed")
            except Exception as e:
                print(f"Error closing idle connection {chat_id}: {e}")
            finally:
                self.disconnect(chat_id)

# Global connection manager
manager = ConnectionManager()

# Set the manager reference in quiztools for cancellation checks
from tools.quiztools import set_connection_manager, get_chat_context_from_db
set_connection_manager(manager)

# ============================================================================
# COST OPTIMIZATION: Background cleanup task
# ============================================================================
async def periodic_cleanup():
    """Background task to clean up idle sessions and connections"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds for 60s timeout
            current_time = time.time()

            # Clean up idle connections
            await manager.cleanup_idle_connections()

            # Clean up expired sessions (even without active WebSocket)
            sessions_to_cleanup = []
            for chat_id, last_activity in list(SESSION_LAST_ACTIVITY.items()):
                idle_time = current_time - last_activity
                if idle_time > SESSION_IDLE_TIMEOUT:
                    sessions_to_cleanup.append(chat_id)
                    print(f"🧹 Session {chat_id} expired (idle {idle_time:.0f}s)")

            for chat_id in sessions_to_cleanup:
                cleanup_session(chat_id)

            # Log stats
            active_sessions = len(ACTIVE_SESSIONS)
            active_connections = len(manager.active_connections)
            if active_sessions > 0 or active_connections > 0:
                print(f"📊 Active: {active_sessions} sessions, {active_connections} connections")

        except Exception as e:
            print(f"⚠️ Cleanup task error: {e}")

# WebSocket endpoint
@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    #connect
    await manager.connect(websocket, chat_id)
    try:
        while True:
            # Wait for incoming message from client with timeout
            # COST OPTIMIZATION: Use asyncio.wait_for to enforce connection timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=CONNECTION_IDLE_TIMEOUT
                )
            except asyncio.TimeoutError:
                print(f"⏰ WebSocket {chat_id} timed out after {CONNECTION_IDLE_TIMEOUT}s idle")
                await websocket.close(1000, "Idle timeout")
                break

            message = json.loads(data)

            # Update activity timestamp
            manager.update_activity(chat_id)

            # Handle different message types
            await handle_websocket_message(chat_id, message, websocket)

    except WebSocketDisconnect:
        manager.disconnect(chat_id)
        print(f"Client {chat_id} disconnected")
    except Exception as e:
        print(f"WebSocket error for {chat_id}: {e}")
        manager.disconnect(chat_id)
        
async def handle_websocket_message(chat_id: str, message: dict, websocket: WebSocket):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    print(f"📨 Received WebSocket message: type={message_type}, chat_id={chat_id}")

    # check if Im getting a message
    if message_type == "chat_message":
        # Handle regular chat messages (proceed to call the AI Tutor)
        await process_chat_message(chat_id, message, websocket)

    # check if Im getting a ping to get the session alive
    elif message_type == "ping":
        # Handle ping/pong for connection keepalive
        await websocket.send_text(json.dumps({"type": "pong"}))

    # check if user wants to cancel ongoing streaming
    elif message_type == "cancel_stream":
        print(f"🛑 Received cancel request for chat {chat_id}")
        manager.cancel_stream(chat_id)
        await websocket.send_text(json.dumps({
            "type": "stream_cancelled",
            "message": "Stream cancellation requested"
        }))

    # ============================================
    # GAME MODE HANDLERS
    # These handle the gamified quiz flow where users
    # collect serum by answering questions correctly
    # ============================================

    elif message_type == "game_quiz":
        # User wants to start a quiz game
        # This streams questions one at a time
        await process_game_quiz(chat_id, message, websocket)

    elif message_type == "game_deliver":
        # User finished quiz and wants to deliver serum
        # Check if they have enough to save the child
        await process_game_deliver(chat_id, message, websocket)

    elif message_type == "game_retry":
        # User didn't have enough serum, wants to try again
        # Serum persists across retries
        await process_game_retry(chat_id, message, websocket)

    # ============================================
    # MICRO-RATIONALE HANDLER
    # Generate short, encouraging feedback after quiz answer
    # Uses GPT-4.1-nano for speed, with instant fallback
    # ============================================
    elif message_type == "micro_rationale_request":
        await process_micro_rationale(chat_id, message, websocket)

    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))
               
async def process_chat_message(chat_id: str, message: dict, websocket: WebSocket):
    """
    Process chat messages through the existing NursingTutor.

    PERFORMANCE OPTIMIZATION (2024):
    ================================
    Previously, this function ran several operations SEQUENTIALLY:
      1. load_file_insights_from_firebase() - ~200-500ms
      2. load_vectorstore_from_firebase()   - ~1-3s (if not cached)
      3. get_chat_context_from_db()         - ~300-800ms
      4. LanguageDetector.detect_language() - ~500-1000ms (LLM call)

    Total sequential delay: 2-6 seconds BEFORE the AI even starts thinking!

    NOW we run operations in PARALLEL where possible:
      - For NEW sessions: insights + vectorstore + context load in parallel
      - For EXISTING sessions: only context is fetched (others already in memory)
      - Language detection moved to the END (runs while user sees "processing" status)
      - Context is passed to orchestrator to avoid DUPLICATE Firebase fetch

    Expected improvement: 50-70% faster time-to-first-response
    """
    try:
        # Extract message data
        user_input = message.get("input", "")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 0: Free-tier quota gate (server-side; the UI gate is bypassable)
        # Same policy as the client: an empty bucket blocks the send entirely.
        # code "quota_exceeded" tells the frontend to open the upgrade modal.
        # ═══════════════════════════════════════════════════════════════════
        quota = usage_guard.check_quota(chat_id)
        if not quota["allowed"]:
            print(f"🚫 Quota exceeded for chat {chat_id} — rejecting chat message")
            # Must be the top-level {type:"error"} envelope: the frontend
            # dispatcher switches on `type` and drops unknown envelopes.
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "quota_exceeded",
                "message": usage_guard.QUOTA_MESSAGE
            }))
            return

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Get or create session
        # ═══════════════════════════════════════════════════════════════════
        session_existed = chat_id in ACTIVE_SESSIONS
        print(f"🔍 ACTIVE_SESSIONS keys: {list(ACTIVE_SESSIONS.keys())}")
        print(f"🔍 Looking for chat_id: {chat_id}")
        print(f"🔍 Session exists: {session_existed}")

        if not session_existed:
            ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)
            print(f"🆕 Created new session for chat {chat_id}")
        else:
            print(f"♻️ Reusing existing session for chat {chat_id}")

        nursing_tutor = ACTIVE_SESSIONS[chat_id]

        # Debug: Check vectorstore status
        has_vectorstore = nursing_tutor.session.vectorstore is not None
        print(f"📊 Session vectorstore status: {'EXISTS in memory' if has_vectorstore else 'NOT in memory'}")
        print(f"📊 Session object id: {id(nursing_tutor.session)}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Run loading operations in PARALLEL (not sequential!)
        # ═══════════════════════════════════════════════════════════════════
        # We build a list of async tasks that need to run, then execute them
        # all at once with asyncio.gather(). This is MUCH faster than awaiting
        # each one sequentially.
        #
        # Tasks to potentially run in parallel:
        #   - load_file_insights_from_firebase() (only for new sessions)
        #   - load_vectorstore_from_firebase()   (only if not in memory)
        #   - get_chat_context_from_db()         (always needed)
        # ═══════════════════════════════════════════════════════════════════

        # Initialize variables for results
        full_context_from_db = None

        # Define async helper functions that we can run in parallel
        async def load_insights_if_needed():
            """Load file insights for new sessions only."""
            if not session_existed:
                await nursing_tutor.load_file_insights_from_firebase()
                print(f"✅ File insights loaded for {chat_id}")

        async def load_vectorstore_if_needed():
            """Load vectorstore if not already in memory."""
            if nursing_tutor.session.vectorstore is None:
                print(f"📥 Loading vectorstore from Firebase for chat {chat_id}...")
                loaded_vectorstore = await vectorstore_manager.load_combined_vectorstore_from_firebase(chat_id)
                if loaded_vectorstore:
                    nursing_tutor.session.vectorstore = loaded_vectorstore
                    print(f"✅ Vectorstore loaded successfully for {chat_id}")
                else:
                    print(f"⚠️ No vectorstore found in Firebase for {chat_id} - background upload may still be in progress")

        async def load_context():
            """Load chat context from Firebase. Returns the context dict."""
            context = await get_chat_context_from_db(chat_id)
            print(f"✅ Chat context loaded for {chat_id}")
            return context

        # ═══════════════════════════════════════════════════════════════════
        # Execute all loading tasks in PARALLEL using asyncio.gather()
        # ═══════════════════════════════════════════════════════════════════
        # asyncio.gather() runs all coroutines concurrently and waits for all
        # to complete. This means:
        #   - If insights takes 300ms, vectorstore takes 2s, context takes 500ms
        #   - Sequential would take: 300 + 2000 + 500 = 2800ms
        #   - Parallel takes: max(300, 2000, 500) = 2000ms (30% faster!)
        # ═══════════════════════════════════════════════════════════════════
        print(f"⚡ Starting parallel loading operations for {chat_id}...")
        parallel_start_time = asyncio.get_event_loop().time()

        # Run all three operations in parallel
        # Note: load_context() returns a value, others just have side effects
        results = await asyncio.gather(
            load_insights_if_needed(),
            load_vectorstore_if_needed(),
            load_context(),
            return_exceptions=True  # Don't fail all if one fails
        )

        parallel_end_time = asyncio.get_event_loop().time()
        print(f"⚡ Parallel loading completed in {(parallel_end_time - parallel_start_time)*1000:.0f}ms")

        # Extract the context result (third item in results)
        # First two tasks (insights, vectorstore) don't return values
        context_result = results[2]

        # Handle potential errors from parallel execution
        if isinstance(context_result, Exception):
            print(f"⚠️ Error loading context: {context_result}")
            full_context_from_db = {'conversation': [], 'quizzes': [], 'study_sheets': []}
        else:
            full_context_from_db = context_result

        # Check if other tasks had errors (log but don't fail)
        if isinstance(results[0], Exception):
            print(f"⚠️ Error loading insights: {results[0]}")
        if isinstance(results[1], Exception):
            print(f"⚠️ Error loading vectorstore: {results[1]}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Send "processing" status to user BEFORE language detection
        # ═══════════════════════════════════════════════════════════════════
        # We send the status update now so the user knows something is happening.
        # Language detection runs next, but user already sees feedback.
        # ═══════════════════════════════════════════════════════════════════
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "processing",
            "message": "Processing your message..."
        }))

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Get or detect language (CACHED per session)
        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZATION: Language detection requires an LLM call (~500-1000ms).
        # Instead of detecting on EVERY message, we detect ONCE per session
        # and reuse the result for all subsequent messages.
        #
        # Logic:
        #   - First message: Detect language, store on session.user_language
        #   - Subsequent messages: Reuse session.user_language (instant, 0ms)
        #
        # This saves 500-1000ms on every message after the first one!
        # ═══════════════════════════════════════════════════════════════════

        # Check if we already have a detected language for this session
        # Debug: Log current state of user_language
        print(f"🔍 DEBUG: session.user_language = {nursing_tutor.session.user_language}")

        requested_language = explicit_language_request(user_input)
        cached_language = nursing_tutor.session.user_language

        if requested_language:
            # "IN ENGLISH", "en français", ... always beats the cache.
            language = requested_language
            nursing_tutor.session.user_language = language
            print(f"🌐 Explicit language request honored: {language}")
        elif cached_language is not None and not cached_language_conflicts(cached_language, user_input):
            # Reuse cached language - skip LLM call entirely
            language = cached_language
            print(f"🌐 Using cached session language: {language} (skipped LLM detection)")
        else:
            # First message in session, or the message's script contradicts the
            # cached language (e.g. session locked to russian by one pasted
            # snippet, but the user is clearly writing Latin-script English).
            if cached_language is not None:
                print(f"🌐 Cached language '{cached_language}' conflicts with message script - re-detecting")
            chat_history = full_context_from_db.get("conversation", [])[-10:] if full_context_from_db else []
            language = await LanguageDetector.detect_language(user_input, chat_history)
            nursing_tutor.session.user_language = language  # Cache for future messages
            print(f"🌐 Detected and cached language: {language}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Process message with orchestrator
        # ═══════════════════════════════════════════════════════════════════
        # IMPORTANT: We pass the pre_fetched_context to avoid a DUPLICATE
        # Firebase query inside the orchestrator. Previously, orchestrator.py
        # was calling get_chat_context_from_db() AGAIN, wasting 300-800ms.
        # ═══════════════════════════════════════════════════════════════════
        async for chunk in nursing_tutor.process_message(
            user_input,
            language,
            pre_fetched_context=full_context_from_db  # Pass context to avoid duplicate fetch!
        ):
            # ─────────────────────────────────────────────────────────
            # Treat every server-sent chunk as connection activity, so
            # the idle-watchdog doesn't kill a connection while we're
            # actively streaming. Without this, long pipelines (e.g.
            # web-research-grounded quiz) silently get terminated mid-
            # stream even though the server is doing real work.
            # ─────────────────────────────────────────────────────────
            manager.update_activity(chat_id)

            # Check for cancellation before processing each chunk
            if manager.is_cancelled(chat_id):
                print(f"🛑 Stream cancelled for chat {chat_id}, stopping...")
                await websocket.send_text(json.dumps({
                    "type": "stream_cancelled",
                    "message": "Streaming stopped by user"
                }))
                manager.reset_cancellation(chat_id)  # Reset for next message
                return  # Stop streaming

            # Parse the existing streaming format
            try:
                chunk_data = json.loads(chunk.strip())

                # Debug: Log mindmap-related chunks
                if chunk_data.get("status") in ["mindmap_generating", "mindmap_complete"]:
                    print(f"🧠 Sending mindmap chunk: status={chunk_data.get('status')}, has_data={bool(chunk_data.get('mindmap_data'))}")

                # Forward to WebSocket with type wrapper
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "data": chunk_data
                }))

            except json.JSONDecodeError:
                # Handle non-JSON chunks
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "data": {"answer_chunk": chunk.strip()}
                }))

            # Every server-sent chunk counts as activity. Without this,
            # the idle watchdog (cleanup_idle_connections) only saw
            # client->server messages and would close the connection
            # mid-stream during long phases like exam research or
            # parallel quiz generation. Updating per-chunk keeps the
            # connection considered "alive" as long as we're actively
            # streaming events to the client.
            manager.update_activity(chat_id)
        
        # Send completion signal
        manager.reset_cancellation(chat_id)  # Reset cancellation flag
        await websocket.send_text(json.dumps({
            "type": "stream_complete",
            "message": "Response complete"
        }))

        # Update activity timestamp - connection stays open for more messages
        # Will be closed by idle timeout (5 min) or client disconnect
        manager.update_activity(chat_id)
        print(f"✅ Stream complete for {chat_id}, connection stays open for follow-up messages")

    except Exception as e:
        print(f"Error processing chat message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Processing failed: {str(e)}"
        }))


# ============================================================================
# UNUSED FOR NOW, WE GAVE UP ON THE GAMIFICATION IDEA, Code kept in case there is an opportunity for this in the future
# GAME MODE FUNCTIONS
# These functions handle the gamified quiz flow where users collect serum
# by answering NCLEX-style questions correctly to save a sick child.
#
# Flow:
#   1. User uploads file → documents get embedded (existing flow)
#   2. Frontend sends "game_quiz" → backend streams questions
#   3. User answers questions → frontend validates client-side
#   4. Frontend sends "game_deliver" with serum count
#   5. If enough serum → child saved! If not → "game_retry"
#
# Serum Math:
#   - Each correct answer = 20mL
#   - Required to save child = 100mL (default)
#   - Serum ACCUMULATES across retries (forgiving design)
# ============================================================================

async def process_game_quiz(chat_id: str, message: dict, websocket: WebSocket):
    """
    Generate and stream quiz questions for game mode.

    Each question includes the answer so the frontend can validate locally.
    This gives instant feedback without server round-trips.

    Message format from frontend:
    {
        "type": "game_quiz",
        "questionCount": 5,       # Optional, default 5
        "difficulty": "medium",   # Optional, default "medium"
        "existingTopics": []      # Optional, user's existing topics for smart matching
    }
    """
    try:
        # Import what we need
        # Use the Question Bank-integrated version for instant delivery + bank enrichment
        from services.quiz_with_bank import stream_quiz_with_bank as stream_quiz_questions
        from firebase_admin import firestore
        from uuid import uuid4

        db = firestore.client()
        chat_ref = db.collection("chats").document(chat_id)

        print(f"🎮 Starting game quiz for chat: {chat_id}")

        # ------------------------------------------
        # Step 1: Get or create the NursingTutor session
        # This gives us access to the vectorstore with user's documents
        # ------------------------------------------
        if chat_id not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)
            await ACTIVE_SESSIONS[chat_id].load_file_insights_from_firebase()

        nursing_tutor = ACTIVE_SESSIONS[chat_id]
        session = nursing_tutor.session  # PersistentSessionContext

        # ------------------------------------------
        # Step 1.5: Ensure vectorstore is loaded from Firebase
        # The upload saves vectorstore to Firebase, but a new WebSocket
        # session needs to load it back into memory.
        #
        # RACE CONDITION HANDLING: The background upload might still be
        # in progress when this runs, so we retry a few times with delays.
        # ------------------------------------------
        if session.vectorstore is None:
            print(f"📥 Loading vectorstore from Firebase for game quiz...")

            # Notify frontend we're loading documents
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "game_loading_documents",
                    "message": "Loading your documents..."
                }
            }))

            # Retry up to 5 times with increasing delays (total ~10 seconds max)
            max_retries = 5
            retry_delays = [1, 2, 2, 3, 3]  # seconds between retries
            loaded_vectorstore = None

            for attempt in range(max_retries):
                loaded_vectorstore = await vectorstore_manager.load_combined_vectorstore_from_firebase(chat_id)

                if loaded_vectorstore:
                    session.vectorstore = loaded_vectorstore
                    print(f"✅ Vectorstore loaded successfully for {chat_id} (attempt {attempt + 1})")
                    break
                else:
                    if attempt < max_retries - 1:
                        wait_time = retry_delays[attempt]
                        print(f"⏳ Vectorstore not ready yet, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"⚠️ No vectorstore found in Firebase for {chat_id} after {max_retries} attempts")

            if not loaded_vectorstore:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Documents are still processing. Please wait a moment and try again."
                }))
                return

        # ------------------------------------------
        # Step 2: Get or initialize game state from Firestore
        # ------------------------------------------
        chat_doc = chat_ref.get()

        if not chat_doc.exists:
            # This shouldn't happen - chat should exist from file upload
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Chat not found. Please upload a file first."
            }))
            return

        chat_data = chat_doc.to_dict()
        game_state = chat_data.get("gameState", {})

        # If no game state exists, initialize it
        if not game_state:
            game_state = {
                "status": "in_progress",
                "serumCollected": 0,
                "serumRequired": 100,
                "attempts": 0,
                "completedAt": None
            }
            chat_ref.update({"gameState": game_state})
            print(f"🎮 Initialized new game state for chat: {chat_id}")

        # ------------------------------------------
        # Step 3: Send game initialized message
        # Frontend uses this to set up the UI
        # ------------------------------------------
        await websocket.send_text(json.dumps({
            "type": "stream_chunk",
            "data": {
                "status": "game_initialized",
                "serumCollected": game_state.get("serumCollected", 0),
                "serumRequired": game_state.get("serumRequired", 100)
            }
        }))

        # ------------------------------------------
        # Step 4: Extract quiz parameters from message
        # ------------------------------------------
        question_count = message.get("questionCount", 5)
        difficulty = message.get("difficulty", "medium")
        existing_topics = message.get("existingTopics", [])  # User's existing topics
        quiz_id = f"quiz_{uuid4().hex[:8]}"

        print(f"🎮 Generating {question_count} {difficulty} questions...")
        print(f"📚 User's existing topics: {existing_topics}")
        print(f"📊 Session vectorstore status: {session.vectorstore is not None}")
        if session.vectorstore:
            print(f"📊 Vectorstore type: {type(session.vectorstore)}")

        # ------------------------------------------
        # Step 5: Stream questions using existing function
        # Each question includes answer + justification for client-side validation
        # ------------------------------------------
        question_index = 0

        async for chunk in stream_quiz_questions(
            topic="",                    # Empty = use all document content
            difficulty=difficulty,
            num_questions=question_count,
            source="documents",          # Generate from user's uploaded docs
            session=session,
            empathetic_message=None,     # No intro message for game mode
            chat_id=chat_id,             # For cancellation checking
            existing_topics=existing_topics  # User's existing topics for smart matching
        ):
            # Handle different chunk types from stream_quiz_questions

            if chunk.get("status") == "generating":
                # Progress update: "Generating question 2 of 5..."
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "data": {
                        "status": "game_generating",
                        "current": chunk.get("current"),
                        "total": chunk.get("total")
                    }
                }))

            elif chunk.get("status") == "question_ready":
                # A complete question is ready to send
                question = chunk.get("question")

                # Send the full question (including answer for client-side validation)
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "data": {
                        "status": "game_question_ready",
                        "question": {
                            "index": question_index,
                            "question": question.get("question"),
                            "options": question.get("options"),
                            "answer": question.get("answer"),           # For client validation
                            # Legacy full rationale (only present on old generations);
                            # new generations leave this empty and ship correctBlurb
                            # instead — the frontend fetches the full per-option
                            # rationale on demand from /quiz_rationale.
                            "justification": question.get("justification", ""),
                            "correctBlurb": question.get("correct_blurb", ""),
                            "topic": question.get("topic", "General"),
                            "serumValue": 20  # Each correct answer = 20mL
                        },
                        "quizId": quiz_id,
                        "isFirst": question_index == 0  # Frontend can start showing UI
                    }
                }))

                question_index += 1
                print(f"✅ Sent question {question_index}/{question_count}")

            elif chunk.get("status") == "quiz_complete":
                # All questions generated
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "data": {
                        "status": "game_quiz_complete",
                        "totalQuestions": chunk.get("total_generated", question_count)
                    }
                }))
                print(f"🎮 Quiz complete! Sent {question_index} questions")

                # Update activity - connection stays open for delivery/retry
                manager.update_activity(chat_id)
                print(f"✅ Game quiz complete for {chat_id}, connection stays open")

    except Exception as e:
        print(f"❌ Game quiz error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to generate quiz: {str(e)}"
        }))


async def process_game_deliver(chat_id: str, message: dict, websocket: WebSocket):
    """
    Handle serum delivery attempt.

    Frontend reports how much serum was collected (based on correct answers).
    We check if it's enough to save the child.

    Message format from frontend:
    {
        "type": "game_deliver",
        "serumCollected": 80  # Total serum from this session
    }
    """
    try:
        from firebase_admin import firestore
        db = firestore.client()

        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()

        if not chat_doc.exists:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Game not found"
            }))
            return

        # ------------------------------------------
        # Step 1: Get current game state
        # ------------------------------------------
        chat_data = chat_doc.to_dict()
        game_state = chat_data.get("gameState", {})

        # Get the serum values
        serum_from_this_quiz = message.get("serumCollected", 0)
        previous_serum = game_state.get("serumCollected", 0)
        serum_required = game_state.get("serumRequired", 100)

        # Total serum = previous attempts + this attempt
        total_serum = previous_serum + serum_from_this_quiz

        print(f"🧪 Delivery attempt: {serum_from_this_quiz}mL this quiz + {previous_serum}mL previous = {total_serum}mL total")
        print(f"🧪 Required: {serum_required}mL")

        # ------------------------------------------
        # Step 2: Update game state with new serum total
        # ------------------------------------------
        attempts = game_state.get("attempts", 0) + 1

        chat_ref.update({
            "gameState.serumCollected": total_serum,
            "gameState.attempts": attempts
        })

        # ------------------------------------------
        # Step 3: Check if we have enough serum
        # ------------------------------------------
        if total_serum >= serum_required:
            # SUCCESS! Child is saved!
            chat_ref.update({
                "gameState.status": "completed",
                "gameState.completedAt": firestore.SERVER_TIMESTAMP
            })

            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "game_child_saved",
                    "serumDelivered": total_serum,
                    "attempts": attempts,
                    "message": "The serum worked. The child is stabilizing. You saved them!"
                }
            }))

            print(f"🎉 Child saved! Total serum: {total_serum}mL in {attempts} attempt(s)")

        else:
            # NOT ENOUGH - Need to retry
            serum_needed = serum_required - total_serum

            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "game_need_more_serum",
                    "serumCollected": total_serum,
                    "serumRequired": serum_required,
                    "serumNeeded": serum_needed,
                    "attempts": attempts,
                    "message": f"The child needs {serum_needed}mL more serum. Keep going!"
                }
            }))

            print(f"⚠️ Need more serum: {serum_needed}mL more required")

    except Exception as e:
        print(f"❌ Game deliver error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Delivery failed: {str(e)}"
        }))


async def process_game_retry(chat_id: str, message: dict, websocket: WebSocket):
    """
    Handle retry request after not having enough serum.

    Key behavior: Serum PERSISTS across retries!
    This is intentional - we want to encourage persistence, not punish failure.

    Message format from frontend:
    {
        "type": "game_retry",
        "questionCount": 5,      # Optional
        "difficulty": "medium"   # Optional
    }
    """
    try:
        from firebase_admin import firestore
        db = firestore.client()

        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()

        if not chat_doc.exists:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Game not found"
            }))
            return

        # ------------------------------------------
        # Get current serum level (this persists!)
        # ------------------------------------------
        chat_data = chat_doc.to_dict()
        game_state = chat_data.get("gameState", {})
        current_serum = game_state.get("serumCollected", 0)
        serum_required = game_state.get("serumRequired", 100)
        serum_needed = serum_required - current_serum

        print(f"🔄 Retry requested. Current serum: {current_serum}mL, need {serum_needed}mL more")

        # ------------------------------------------
        # Notify frontend that retry is starting
        # ------------------------------------------
        await websocket.send_text(json.dumps({
            "type": "stream_chunk",
            "data": {
                "status": "game_retry_starting",
                "serumCollected": current_serum,
                "serumRequired": serum_required,
                "serumNeeded": serum_needed,
                "message": "Generating more questions... Your serum is safe!"
            }
        }))

        # ------------------------------------------
        # Reuse the game_quiz handler to generate new questions
        # ------------------------------------------
        await process_game_quiz(chat_id, message, websocket)

    except Exception as e:
        print(f"❌ Game retry error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Retry failed: {str(e)}"
        }))


# ============================================================================
# MICRO-RATIONALE HANDLER
# Generate short, encouraging feedback using GPT-4.1-nano
# ============================================================================

# Static fallbacks for instant response when GPT-nano is slow/fails
MICRO_RATIONALE_FALLBACKS = {
    "correct": {
        "en": {
            "encouragement": "Nice — that's a core clinical priority.",
            "rationale": "The priority is addressing life-threatening problems first."
        },
        "fr": {
            "encouragement": "Bien joué — c'est une priorité clinique essentielle.",
            "rationale": "La priorité est de traiter d'abord les problèmes vitaux."
        }
    },
    "incorrect": {
        "en": {
            "encouragement": "Good attempt — this is a common exam trap.",
            "rationale": "Focus on the most critical intervention first."
        },
        "fr": {
            "encouragement": "Bonne tentative — c'est un piège d'examen courant.",
            "rationale": "Concentrez-vous d'abord sur l'intervention la plus critique."
        }
    }
}


async def process_micro_rationale(chat_id: str, message: dict, websocket: WebSocket):
    """
    Generate short, encouraging feedback for quiz answers using GPT-4.1-nano.

    Hard constraints:
    - Encouragement: ≤ 12 words
    - Rationale: ≤ 2 sentences
    - No lists, no option-by-option breakdown
    - Must return in <300ms or use fallback

    Message format from frontend:
    {
        "type": "micro_rationale_request",
        "question": "What is the primary purpose...",
        "correct_answer": "To quickly determine...",
        "selected_answer": "To assess the patient's...",
        "is_correct": false,
        "topic": "Emergency Evaluation",
        "original_rationale": "Option C is correct because...",
        "language": "en"
    }
    """
    try:
        # Extract data from message
        question = message.get("question", "")
        correct_answer = message.get("correct_answer", "")
        selected_answer = message.get("selected_answer", "")
        is_correct = message.get("is_correct", False)
        topic = message.get("topic", "General")
        original_rationale = message.get("original_rationale", "")
        language = message.get("language", "en")

        # Ensure language is supported
        lang = language if language in ["en", "fr"] else "en"

        # Get fallback based on correctness
        fallback_key = "correct" if is_correct else "incorrect"
        fallback = MICRO_RATIONALE_FALLBACKS[fallback_key][lang]

        print(f"🎯 Micro-rationale request: {'correct' if is_correct else 'incorrect'} answer")

        try:
            # Use GPT-4.1-nano for fastest response
            llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7, request_timeout=0.5)

            # Build tone instruction
            if is_correct:
                tone = "Supportive and confirming. Reinforce why this was the right choice."
            else:
                tone = "Supportive and non-judgmental. Normalize the mistake, explain briefly."

            # Build prompt - minimal, focused
            prompt = f"""Quiz feedback generator. Keep responses SHORT and ENCOURAGING.

Question: {question[:300]}
Correct answer: {correct_answer[:150]}
User selected: {selected_answer[:150]}
User was: {"CORRECT" if is_correct else "INCORRECT"}
Topic: {topic}

{f"Original rationale to condense: {original_rationale[:400]}" if original_rationale else ""}

HARD CONSTRAINTS:
- Encouragement: ≤ 12 words, {tone}
- Rationale: ≤ 2 sentences explaining the key concept
- NO lists, NO "Option A/B/C", NO "According to..."
- Be warm, brief, human

Return ONLY valid JSON:
{{"encouragement": "...", "rationale": "..."}}"""

            # Race against timeout
            result = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=0.5  # 500ms hard limit
            )

            # Parse response
            response_text = result.content.strip()

            # Extract JSON from response
            import re
            if response_text.startswith("{"):
                parsed = json.loads(response_text)
            else:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            # Send successful response
            await websocket.send_text(json.dumps({
                "type": "micro_rationale_response",
                "data": {
                    "encouragement": parsed.get("encouragement", fallback["encouragement"])[:100],
                    "rationale": parsed.get("rationale", fallback["rationale"])[:250],
                    "source": "gpt-nano"
                }
            }))

            print(f"✅ Micro-rationale generated via GPT-nano")
            return

        except asyncio.TimeoutError:
            print("⚠️ GPT-nano timeout, using fallback")
        except Exception as e:
            print(f"⚠️ GPT-nano error: {e}, using fallback")

        # Fallback response
        await websocket.send_text(json.dumps({
            "type": "micro_rationale_response",
            "data": {
                **fallback,
                "source": "fallback"
            }
        }))

    except Exception as e:
        print(f"❌ Micro-rationale error: {e}")
        # Return generic fallback on total failure
        await websocket.send_text(json.dumps({
            "type": "micro_rationale_response",
            "data": {
                "encouragement": "Keep going — you're learning!",
                "rationale": "Each question helps build your understanding.",
                "source": "error_fallback"
            }
        }))


# ============================================================================
# SUPPORTING ENDPOINTS (keep your existing ones)
# ============================================================================
def get_temp_dir():
    """Get appropriate temp directory for the current environment"""
    if platform.system() == "Linux":
        # On Cloud Run (Linux), use /tmp which is writable
        return "/tmp"
    else:
        # On Windows/Mac, use system default
        return tempfile.gettempdir()


# ============================================================================
# POST-UPLOAD MESSAGE HELPERS
# ============================================================================
# These functions build the friendly message shown after file upload.
# They use templates instead of LLM calls for instant response + zero cost.
# ============================================================================

def build_post_upload_message(topics: list, file_count: int, language: str) -> str:
    """
    Build a friendly message after file upload using templates.

    WHY TEMPLATES INSTEAD OF LLM:
    - Instant response (no API latency)
    - Zero cost (no tokens used)
    - Predictable output
    - Still feels natural with variety built in

    Args:
        topics: List of topics extracted from uploaded files
        file_count: Number of files uploaded
        language: User's language ('english', 'french', 'fr', etc.)

    Returns:
        A friendly message string
    """
    import random

    # Determine if French
    is_french = language.lower() in ["fr", "french", "français"]

    # Format topics into readable string
    if topics:
        if len(topics) == 1:
            topics_str = topics[0]
        elif len(topics) == 2:
            topics_str = f"{topics[0]} and {topics[1]}" if not is_french else f"{topics[0]} et {topics[1]}"
        else:
            # "Topic1, Topic2, and Topic3"
            if is_french:
                topics_str = ", ".join(topics[:-1]) + f" et {topics[-1]}"
            else:
                topics_str = ", ".join(topics[:-1]) + f", and {topics[-1]}"
    else:
        topics_str = "your study material" if not is_french else "ton matériel d'étude"

    # File count text
    if file_count == 1:
        file_text = "your notes" if not is_french else "tes notes"
    else:
        file_text = f"all {file_count} files" if not is_french else f"les {file_count} fichiers"

    # Message templates - variety keeps it feeling natural
    if is_french:
        templates = [
            f"C'est bon! J'ai parcouru {file_text} et trouvé du contenu sur {topics_str}. Par où veux-tu commencer?",
            f"Parfait! J'ai analysé {file_text} — on a de la matière sur {topics_str}. Qu'est-ce qu'on attaque en premier?",
            f"J'ai tout reçu! Tes documents couvrent {topics_str}. Comment veux-tu étudier?",
        ]
    else:
        templates = [
            f"Got it! I've gone through {file_text} and found material on {topics_str}. Where would you like to start?",
            f"All set! I've looked through {file_text} — there's good content on {topics_str}. How do you want to dive in?",
            f"Nice! Your documents cover {topics_str}. Pick a way to start studying!",
        ]

    return random.choice(templates)


def get_post_upload_actions(language: str) -> list:
    """
    Get the action buttons to show after file upload.

    Four chips, ordered by measured click share. Six chips wrapped to a ragged
    4+2 second row and diluted the menu; these four cover ~93% of all recorded
    chip clicks and fit one row.

    1. Check my understanding - explain it back, get corrected (dialogue)
    2. Quiz me      - 47% of chip clicks, 228 distinct users
    3. Study sheet  - 27% of chip clicks, 111 distinct users
    4. Flashcards   - 18% of chip clicks, 103 distinct users

    Audio and concept map were REMOVED from this menu, not from the product:
    in chat they were near-dead (28 and 44 artifacts ever), but inside a study
    session they are heavily used (study_audio 1,566, study_mindmap 407). They
    were in the wrong surface, so they stay in Study Mode where they work.
    The frontend still renders both ids if any other caller emits them.

    "checkme" is deliberately FIRST. Explain-it-back-and-get-corrected is the
    interaction with the strongest observed link to conversion, but it had no
    UI surface at all — students only reached it by free-typing into the
    composer. First position is what makes it testable; re-measure click share
    after a few weeks and drop it if it can't clear ~10% from this slot.

    Args:
        language: User's language

    Returns:
        List of action dictionaries with id, label, and icon
    """
    is_french = language.lower() in ["fr", "french", "français"]

    if is_french:
        return [
            {"id": "checkme", "label": "Vérifie ma compréhension", "icon": "💬"},
            {"id": "quiz", "label": "Teste-moi sur ces sujets", "icon": "🧪"},
            {"id": "studysheet", "label": "Fais-moi un résumé", "icon": "📝"},
            {"id": "flashcards", "label": "Créer des flashcards", "icon": "📇"}
        ]
    else:
        return [
            {"id": "checkme", "label": "Check my understanding", "icon": "💬"},
            {"id": "quiz", "label": "Quiz me on these topics", "icon": "🧪"},
            {"id": "studysheet", "label": "Break it down for me", "icon": "📝"},
            {"id": "flashcards", "label": "Create flashcards to study", "icon": "📇"}
        ]


@app.post("/chat/upload-files")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    chat_id: str = Form(...),
    user_id: str = Form(...),
    language: str = Form(...),
):
    """Upload multiple files and process in parallel."""
    # Validate and normalize language parameter
    if not language or language == "undefined" or language == "null":
        language = "english"  # Default fallback

    # Normalize language code (handle 'fr-FR', 'fr-CA', etc.)
    language = language.lower().split('-')[0]  # 'fr-FR' -> 'fr'
    prompt_language = _language_for_prompt(language)
    print(f"*Upload received: user {user_id}, chat:{chat_id}, language:{language}*")
    # Read all files immediately
    file_data_list = []
    for file in files:
        try:
            file_bytes = await file.read()
            file_data_list.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "bytes": file_bytes,
                "size": len(file_bytes)
            })
            print(f"✅ Read file: {file.filename} ({len(file_bytes)} bytes)")
        except Exception as read_error:
            print(f"❌ Failed to read {file.filename}: {read_error}")
            file_data_list.append({
                "filename": file.filename,
                "error": str(read_error)
            })
    
    async def process_and_stream():
        try:
            if not file_data_list:
                yield json.dumps({
                    "type": "error",
                    "message": "No files provided"
                }) + "\n"
                return
            
            # Check for read errors
            failed_reads = [f for f in file_data_list if "error" in f]
            if failed_reads:
                for failed in failed_reads:
                    yield json.dumps({
                        "type": "file_error",
                        "filename": failed["filename"],
                        "message": f"Failed to read file: {failed['error']}"
                    }) + "\n"
            
            # Get successfully read files
            valid_files = [f for f in file_data_list if "bytes" in f]
            
            if not valid_files:
                yield json.dumps({
                    "type": "error",
                    "message": "No valid files to process"
                }) + "\n"
                return
            
            yield json.dumps({
                "type": "batch_start",
                "total_files": len(valid_files),
                "filenames": [f["filename"] for f in valid_files]
            }) + "\n"
            
            # ========================================
            # LOAD EXISTING VECTORSTORE IF NEEDED
            # ========================================
            yield json.dumps({
                "type": "loading_existing_documents",
                "message": "Reading existing documents..."
            }) + "\n"
            
            await ensure_session_with_vectorstore(chat_id)
            
            # ========================================
            # PROCESS FILES (EMBEDDING ONLY)
            # ========================================
            semaphore = asyncio.Semaphore(15)

            # Stream progress events the moment file tasks emit them (instead of
            # one burst per file at the end), and heartbeat every 10s while
            # waiting so Safari/proxies never see a silent connection and the
            # frontend stall-watchdog knows the server is alive.
            update_queue: asyncio.Queue = asyncio.Queue()

            class _QueueSink:
                """List-like sink: every append() streams immediately."""
                def append(self, item):
                    update_queue.put_nowait(item)

            live_updates = _QueueSink()

            async def process_single_file_data(file_data):
                async with semaphore:
                    return await process_file_from_bytes(
                        file_data["bytes"],
                        file_data["filename"],
                        chat_id,
                        user_id,
                        language,  # Pass browser language
                        updates=live_updates
                    )

            # Start all file processing tasks
            tasks = [asyncio.create_task(process_single_file_data(fd)) for fd in valid_files]
            gather_task = asyncio.gather(*tasks, return_exceptions=True)

            total_words = 0
            completed_files = []
            file_documents = {}  # filename -> documents
            file_bytes_map = {}  # filename -> bytes (for background upload)
            last_file_error = None  # remember why files failed (for batch error code)

            # Drain the queue until all tasks are done and nothing is pending
            while True:
                if gather_task.done() and update_queue.empty():
                    break
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield json.dumps({"type": "heartbeat"}) + "\n"
                    continue

                if not isinstance(update, dict):
                    continue

                yield json.dumps(update) + "\n"

                if update.get("type") == "file_complete":
                    total_words += update.get("word_count", 0)
                    completed_files.append(update.get("file_id"))
                elif update.get("type") in ("file_error", "embedding_error"):
                    last_file_error = update.get("message") or last_file_error

            # Collect per-file results (documents, insights, raw bytes)
            for result in gather_task.result():
                if isinstance(result, Exception):
                    print(f"❌ Task error: {result}")
                    last_file_error = str(result)
                    yield json.dumps({
                        "type": "file_error",
                        "message": str(result)
                    }) + "\n"
                    continue

                documents = result.get("documents", [])
                insights = result.get("insights")
                filename = result.get("filename", "unknown")
                file_bytes = result.get("file_bytes")

                print(f"🔍 Processing result for: {filename}")
                print(f"   Documents count: {len(documents)}")

                if insights:
                    print(f"   Insights extracted from upload: {len(insights.get('topics', []))} topics")

                # Store documents for background upload
                if documents:
                    file_documents[filename] = documents
                if file_bytes:
                    file_bytes_map[filename] = file_bytes

                # ========================================
                # STORE INSIGHTS IN SESSION
                # ========================================
                if insights and chat_id in ACTIVE_SESSIONS:
                    session = ACTIVE_SESSIONS[chat_id]
                    #update the session language using the front-end browser language when uploading
                    session.session.user_language = language
                    if not hasattr(session.session, "file_insights"):
                        session.session.file_insights = {}
                    session.session.file_insights[filename] = insights

            # If every file failed to process, surface a batch error instead of
            # a fake success — the frontend flips the loading box to its failed
            # state on this. Quota/rate-limit failures get code "capacity" so
            # the UI can tell users we're overloaded rather than blame them.
            if not completed_files:
                err_text = (last_file_error or "").lower()
                overloaded = any(token in err_text for token in
                                 ("429", "quota", "rate limit", "ratelimit", "overloaded"))
                yield json.dumps({
                    "type": "error",
                    "code": "capacity" if overloaded else "processing_failed",
                    "message": ("We have too many users right now and couldn't complete the request"
                                if overloaded else "All files failed to process")
                }) + "\n"
                return


            # ========================================
            # Generate Upload Summary (1-2 sentences)
            # ========================================
            if chat_id in ACTIVE_SESSIONS:
                session = ACTIVE_SESSIONS[chat_id]
                file_insights = getattr(session.session, "file_insights", {})
                
                if file_insights:
                    try:
                        # Aggregate all insights
                        all_topics = []
                        all_doc_types = []
                        
                        for filename, insights in file_insights.items():
                            if insights:
                                all_topics.extend(insights.get("topics", []))
                                doc_type = insights.get("document_type", "")
                                if doc_type:
                                    all_doc_types.append(doc_type)
                        
                        # Deduplicate
                        unique_topics = list(set(all_topics))
                        unique_doc_types = list(set(all_doc_types))
                        
                        # Generate summary with LLM
                        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
                        
                        
                        summary_prompt = f"""Generate a brief 1-2 sentence summary about what these uploaded documents contain.

                        Files: {len(file_insights)} document(s)
                        Topics found: {', '.join(unique_topics)}
                        Document types: {', '.join(unique_doc_types) if unique_doc_types else 'various'}

                        Write a natural, conversational summary that tells the student what content was found.
                        Examples:
                        - "I found materials about cardiac pharmacology and arrhythmia management."
                        - "Les documents sur la pharmacologie cardiaque et la gestion des arythmies."

                        IMPORTANT REQUIREMENT: The entire summary must be written in {prompt_language}.

                        Return ONLY the summary text in {prompt_language}, nothing else."""
                        
                        # Bounded wait: a hung LLM call must not stall the stream
                        summary_response = await asyncio.wait_for(
                            llm.ainvoke([{"role": "user", "content": summary_prompt}]),
                            timeout=20.0
                        )

                        upload_summary = summary_response.content.strip()

                        # Yield summary to frontend
                        yield json.dumps({
                            "type": "upload_summary",
                            "summary": upload_summary,
                            "file_count": len(valid_files),
                            "filenames": [f["filename"] for f in valid_files]
                        }) + "\n"

                        print(f"📝 Generated upload summary: {upload_summary}")

                        # Chat title isn't needed by the stream — generate and
                        # save it in the background so it doesn't delay completion
                        asyncio.create_task(generate_and_save_chat_title(
                            chat_id, unique_topics, unique_doc_types, prompt_language
                        ))

                    except Exception as summary_error:
                        print(f"⚠️ Summary generation failed: {summary_error}")
            
            # ========================================
            # DISABLED: Suggestions after upload
            # PostUploadActions now handles guiding the user
            # ========================================
            # Suggestions are only generated during normal conversation flow
            # ========================================
            # READY TO CHAT - USER CAN START IMMEDIATELY
            # ========================================
            yield json.dumps({
                "type": "ready_to_chat",
                "message": "Files processed! You can start asking questions.",
                "total_files": len(valid_files),
                "completed_files": len(completed_files),
                "total_words": total_words
            }) + "\n"
            
            # ========================================
            # BACKGROUND: UPLOAD EVERYTHING
            # ========================================
            if chat_id in ACTIVE_SESSIONS:
                session = ACTIVE_SESSIONS[chat_id]
                has_vs = session.session.vectorstore is not None
                print(f"📤 Upload complete for {chat_id}: session in ACTIVE_SESSIONS=True, vectorstore={'EXISTS' if has_vs else 'None'}")
                if session.session.vectorstore:
                    # Start background task (fire-and-forget with retry)
                    asyncio.create_task(
                        upload_everything_background(
                            chat_id=chat_id,
                            vectorstore=session.session.vectorstore,
                            file_documents=file_documents,
                            file_bytes_map=file_bytes_map
                        )
                    )
            else:
                print(f"⚠️ Upload complete but chat_id {chat_id} NOT in ACTIVE_SESSIONS!")
            
            # ========================================
            # FINAL SUMMARY (IMMEDIATE)
            # ========================================
            yield json.dumps({
                "type": "all_complete",
                "total_files": len(valid_files),
                "completed_files": len(completed_files),
                "total_words": total_words,
                "status": "success"
            }) + "\n"

            # ========================================
            # POST-UPLOAD: FRIENDLY MESSAGE + ACTIONS
            # ========================================
            #
            # WHY: After uploading, users see stats but don't know what to do next.
            # This sends a friendly AI message with action buttons to guide them.
            #
            # COST OPTIMIZATION: We reuse the topics already extracted during upload
            # (stored in file_insights) instead of making another LLM call to analyze.
            # The friendly message is built from a template - no extra API call needed.
            #
            # FLOW:
            # 1. Collect topics from all uploaded files (already extracted)
            # 2. Build a friendly message using templates (instant, no LLM)
            # 3. Send to frontend + save to Firebase
            # ========================================

            if chat_id in ACTIVE_SESSIONS:
                session = ACTIVE_SESSIONS[chat_id]
                file_insights = getattr(session.session, "file_insights", {})

                if file_insights:
                    try:
                        # -----------------------------------------
                        # STEP 1: Collect all topics and educational insights from uploaded files
                        # These were already extracted during file processing
                        # -----------------------------------------
                        all_topics = []
                        all_insights = []
                        for filename, insights in file_insights.items():
                            if insights and insights.get("topics"):
                                all_topics.extend(insights.get("topics", []))
                            if insights and insights.get("insights"):
                                all_insights.extend(insights.get("insights", []))

                        # Remove duplicates, keep max 5 topics and 3 insights for readability
                        unique_topics = list(set(all_topics))[:5]
                        # Deduplicate insights by topic
                        seen_topics = set()
                        unique_insights = []
                        for insight in all_insights:
                            topic = insight.get("topic", "")
                            if topic not in seen_topics:
                                seen_topics.add(topic)
                                unique_insights.append(insight)
                            if len(unique_insights) >= 3:
                                break

                        filenames = [f["filename"] for f in valid_files]
                        file_count = len(valid_files)

                        # -----------------------------------------
                        # STEP 2: Build friendly message from template
                        # No LLM call = instant response, zero cost
                        # -----------------------------------------
                        friendly_message = build_post_upload_message(
                            topics=unique_topics,
                            file_count=file_count,
                            language=language
                        )

                        # -----------------------------------------
                        # STEP 3: Build action buttons (localized)
                        # These appear below the message for quick actions
                        # -----------------------------------------
                        actions = get_post_upload_actions(language)

                        # -----------------------------------------
                        # STEP 4: Send to frontend
                        # Frontend will handle saving to Firebase to control timing
                        # (ensures LoadingMessageBox is saved first, then PostUploadActions)
                        # -----------------------------------------
                        yield json.dumps({
                            "type": "post_upload_message",
                            "message": friendly_message,
                            "topics": unique_topics,
                            "insights": unique_insights,  # Educational insights for orientation flow
                            "filenames": filenames,
                            "file_count": file_count,
                            "actions": actions
                        }) + "\n"

                        print(f"✅ Post-upload message sent to frontend (frontend saves to Firebase)")
                        print(f"   Topics: {unique_topics}")
                        print(f"   Educational insights: {len(unique_insights)}")

                    except Exception as post_upload_error:
                        # Non-critical - user can still chat even if this fails
                        print(f"⚠️ Post-upload message failed: {post_upload_error}")

        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(
        process_and_stream(),
        media_type="application/x-ndjson",
        headers={
            # Prevent proxies/CDNs (and Safari) from buffering the NDJSON
            # stream — without these, progress events arrive in one burst
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
    )

# ============================================================================
# HELPERS FOR FILE UPLOAD START
# ============================================================================

# Serialize FAISS index writes per chat — the index isn't safe under
# concurrent mutation now that file embeddings run truly in parallel
_VECTORSTORE_WRITE_LOCKS: Dict[str, asyncio.Lock] = {}

def _get_vectorstore_write_lock(chat_id: str) -> asyncio.Lock:
    if chat_id not in _VECTORSTORE_WRITE_LOCKS:
        _VECTORSTORE_WRITE_LOCKS[chat_id] = asyncio.Lock()
    return _VECTORSTORE_WRITE_LOCKS[chat_id]


async def generate_and_save_chat_title(chat_id: str, unique_topics: list, unique_doc_types: list, prompt_language: str):
    """Generate a chat title from upload insights and save it to Firestore.

    Runs as a background task after the upload stream completes so it never
    delays the user-facing upload flow.
    """
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

        title_prompt = f"""Generate a short, descriptive chat title in 3 to 6 words based on these uploaded study materials.


        Topics Found: {', '.join(unique_topics) if unique_topics else 'medical content'}
        Document Types: {', '.join(unique_doc_types) if unique_doc_types else 'study materials'}

        Requirements:
        - Be concise and clear
        - Focus on the main topic/subject area
        - Max 6 words
        - Make it specific to the content (e.g., "Cardiac Pharmacology Notes", "NCLEX Respiratory Review")
        - Write in {prompt_language}

        Return ONLY the title, no quotes or extra text."""

        title_response = await llm.ainvoke([
            {"role": "user", "content": title_prompt}
        ])

        chat_title = title_response.content.strip().replace('"', '').replace("'", "")

        # Update Firebase chat document with new title (sync client → thread)
        from firebase_admin import firestore
        db = firestore.client()
        await asyncio.to_thread(
            db.collection('chats').document(chat_id).update,
            {
                'title': chat_title,
                'updatedAt': firestore.SERVER_TIMESTAMP
            }
        )

        print(f"✅ Auto-generated and saved chat title: '{chat_title}'")

    except Exception as title_error:
        print(f"⚠️ Background title generation failed: {title_error}")


async def upload_everything_background(
    chat_id: str,
    vectorstore: FAISS,
    file_documents: Dict[str, List[Document]],
    file_bytes_map: Dict[str, bytes],
    max_retries: int = 3
):
    """
    Upload files and vectorstores to Firebase in the background.
    Includes automatic retry on failure (silent).
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"🔄 Background upload starting for chat {chat_id} (attempt {retry_count + 1}/{max_retries})...")
            
            # ========================================
            # UPLOAD FILES TO FIREBASE STORAGE
            # ========================================
            file_upload_tasks = []
            
            for filename, file_bytes in file_bytes_map.items():
                file_upload_tasks.append(
                    firebase_upload_task_simple(file_bytes, filename, chat_id)
                )
            
            # ========================================
            # UPLOAD VECTORSTORES
            # ========================================
            vectorstore_results = await vectorstore_manager.upload_all_vectorstores(
                chat_id=chat_id,
                combined_vectorstore=vectorstore,
                file_documents=file_documents
            )
            
            # ========================================
            # UPLOAD FILES IN PARALLEL
            # ========================================
            file_results = await asyncio.gather(*file_upload_tasks, return_exceptions=True)
            
            # Check results
            file_failures = [r for r in file_results if isinstance(r, Exception)]
            
            if vectorstore_results["combined_success"] and len(file_failures) == 0:
                print(f"✅ Background upload complete for chat {chat_id}")
                return  # Success - exit
            else:
                print(f"⚠️ Background upload had issues (attempt {retry_count + 1})")
                if not vectorstore_results["combined_success"]:
                    print(f"   - Combined vectorstore failed")
                if file_failures:
                    print(f"   - {len(file_failures)} file uploads failed")
                
                # Retry
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                    print(f"   Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        
        except Exception as e:
            print(f"❌ Background upload error (attempt {retry_count + 1}): {e}")
            import traceback
            traceback.print_exc()
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"   Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    # All retries failed - log and give up (silent failure)
    print(f"❌ Background upload failed for chat {chat_id} after {max_retries} attempts")
    print(f"   User can still chat - vectorstore is in memory")

def _language_for_prompt(lang_code: str) -> str:
    """
    Map short language codes to clearer labels for prompt conditioning.
    Keeps default as-is if unknown.
    """
    if not lang_code:
        return "English"
    code = lang_code.lower()
    mapping = {
        "en": "English",
        "english": "English",
        "fr": "French",
        "french": "French",
        "es": "Spanish",
        "spanish": "Spanish"
    }
    return mapping.get(code, lang_code)


async def extract_file_insights_from_text(
    text: str, 
    filename: str, 
    chat_id: str, 
    file_id: str, 
    updates: list,
    language: str = "english"
) -> dict:
    """
    Extract key topics and concepts from document text using random sampling.
    Runs in parallel with embedding for speed.
    
    Args:
        text: Full document text
        filename: Name of the file
        chat_id: Chat ID
        file_id: File ID for progress updates
        updates: List to append progress updates to
        language: Browser language for localized insights
    
    Returns:
        Dict with topics, concepts, and document_type
    """
    try:
        updates.append({
            "type": "insight_extraction_start",
            "file_id": file_id,
            "filename": filename
        })
        
        # ========================================
        # FRAMEWORK DETECTION — on the FULL text
        # ========================================
        # Deliberately BEFORE the sampling below. That sampling reads three
        # random 1,000-char windows, and a framework (the nursing process,
        # Maslow, ABCDE...) is typically defined once, in one place — random
        # windows miss it. This pass is pure keyword matching: no model call,
        # no added latency, and it can afford to read everything.
        #
        # Frameworks are the closed set in constants/nursing_frameworks.py.
        # An empty list is the normal result and means "this document teaches
        # no framework we can test" — never a reason to loosen the thresholds.
        detected_frameworks = detect_frameworks(text)
        if detected_frameworks:
            print(f"🧭 Frameworks in {filename}: " + ", ".join(
                f'{f["id"]}({f["confidence"]})' for f in detected_frameworks))

        # Sample random sections for fast analysis
        text_length = len(text)
        
        if text_length < 5000:
            # Small file - use all text
            sample_text = text
        else:
            # Large file - sample 3 random sections (1000 chars each)
            import random
            samples = []
            for _ in range(3):
                start_pos = random.randint(0, max(0, text_length - 1000))
                samples.append(text[start_pos:start_pos + 1000])
            sample_text = "\n\n---\n\n".join(samples)
        
        # Convert to a clearer label for the prompt (e.g., "fr" -> "French")
        prompt_language = _language_for_prompt(language)

        # Use GPT-4o-mini for fast, cheap analysis                
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        
        # Determine response language
        prompt = f"""Analyze this document and identify its CORE PURPOSE in {prompt_language}.

        Document: {filename}
        Content sample:
        {sample_text[:2500]}

        CRITICAL: Focus on the MAIN THESIS or CENTRAL QUESTION of this document.
        - What is the document trying to teach or prove?
        - What is the key relationship or concept being explored?
        - Ignore metadata (demographics, methodology details, sample sizes) - focus on the CONCLUSION or MAIN TEACHING POINT.

        Example of what we want:
        - Document about sleep and testosterone → Topic: "Sleep deprivation reduces testosterone levels"
        - Document about pressure injuries → Topic: "Staging pressure injuries and preventing them"
        - NOT: "Demographics, education levels, sample characteristics" (these are details, not the core topic)

        Keep a named framework NAMED. If the document teaches the nursing process,
        Maslow's hierarchy, ABCDE, SBAR or similar, say so by name rather than
        paraphrasing it into a description — those names are how the student's
        course and exam refer to it.

        Identify:
        1. Core topic (1-3 MAIN subjects this document is fundamentally about - the central thesis)
        2. Key concepts (5-10 important terms or findings the student needs to remember)
        3. Document type (research paper, textbook, clinical guide, lecture notes, etc.)
        4. Key insights - For each core topic, extract what the student MUST learn:
           - topic: The main subject (e.g. "Sleep and Testosterone", "ABCDE Assessment")
           - insight: The key finding or teaching point (1 sentence - what should the student remember?)
           - key_points: 2-4 specific facts, steps, or conclusions FROM the document
           - context: Where/when this knowledge applies

        IMPORTANT:
        - Topics should answer "What is this document ABOUT?" not "What variables were measured?"
        - key_points should be actionable knowledge, not methodology details

        Return ONLY valid JSON with content in {prompt_language}:
        {{
        "topics": ["Core topic 1", "Core topic 2"],
        "concepts": ["key term 1", "key term 2", ...],
        "document_type": "type",
        "insights": [
            {{
                "topic": "Main subject of document",
                "insight": "The key finding or teaching point",
                "key_points": ["Important fact 1", "Important fact 2", "Important fact 3"],
                "context": "Where this knowledge is applied"
            }}
        ]
        }}
        """
        
        response = await llm.ainvoke([
            {"role": "system", "content": f"You are an expert tutor who identifies the CORE PURPOSE and MAIN THESIS of educational documents. Focus on what the student needs to LEARN, not on research methodology or metadata. Return only valid JSON with all content in {prompt_language}."},
            {"role": "user", "content": prompt}
        ])
        
        # Parse response
        try:
            insights = json.loads(response.content.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse insights JSON for {filename}")
            default_topic = "contenu médical" if language.lower() in ["fr", "french", "français"] else "medical content"
            default_type = "document"
            insights = {
                "topics": [default_topic],
                "concepts": [],
                "document_type": default_type,
                "insights": []
            }
        
        # Attached after the parse so the fallback path keeps them too: detection
        # is deterministic and independent of whether the model returned valid JSON.
        insights["frameworks"] = detected_frameworks

        print(f"✅ Extracted insights from {filename}:")
        print(f"   Topics: {insights.get('topics', [])}")
        print(f"   Concepts: {insights.get('concepts', [])[:3]}...")
        print(f"   Educational insights: {len(insights.get('insights', []))} generated")

        # Stream insight batch to frontend
        updates.append({
            "type": "insight_batch",
            "file_id": file_id,
            "filename": filename,
            "topics": insights.get("topics", []),
            "concepts": insights.get("concepts", [])[:5],  # Limit to 5 for UX
            "document_type": insights.get("document_type", ""),
            "insights": insights.get("insights", [])[:3],  # Limit to 3 educational insights
            # Only what a consumer needs to act: which framework, and how sure we
            # are. The matched vocabulary stays server-side for debugging.
            "frameworks": [
                {"id": f["id"], "name": f["name"], "confidence": f["confidence"]}
                for f in detected_frameworks
            ]
        })
        
        return insights
        
    except Exception as e:
        print(f"⚠️ Insight extraction failed for {filename}: {e}")
        return None

async def firebase_upload_task_simple(file_bytes: bytes, filename: str, chat_id: str):
    """Simple file upload to Firebase Storage (for background task)."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"chats/{chat_id}/uploads/{filename}")
        
        blob.upload_from_string(
            file_bytes,
            content_type=get_content_type(filename)
        )
        
        blob.make_public()
        firebase_url = blob.public_url
        
        print(f"✅ Background: Uploaded {filename} to Firebase Storage")
        return firebase_url
        
    except Exception as e:
        print(f"❌ Background: Failed to upload {filename}: {e}")
        raise

async def process_file_from_bytes(file_bytes: bytes, filename: str, chat_id: str, user_id: str, language: str = "english", updates=None):
    """Process a file from bytes and return list of progress updates.

    `updates` can be any object with .append() — a plain list, or a streaming
    sink that forwards each event to the client as it happens.
    """
    if updates is None:
        updates = []
    file_id = str(uuid4())
    temp_path = None
    documents_for_vectorstore = []
    
    try:
        file_size = len(file_bytes)
        
        updates.append({
            "type": "file_start",
            "file_id": file_id,
            "filename": filename,
            "size": file_size
        })
        
        # Save to temp file
        temp_path = await save_temp_file(file_bytes, filename)
        
        updates.append({
            "type": "file_processing",
            "file_id": file_id,
            "stage": "saved_temp"
        })
        
        # ========================================
        # ONLY EMBEDDING NOW - NO FIREBASE UPLOAD
        # ========================================
        embedding_result = await embed_document_task(
            temp_path, filename, chat_id, file_id, updates, language
        )
        
        # Handle errors
        if isinstance(embedding_result, Exception):
            print(f"❌ Embedding error: {embedding_result}")
            updates.append({
                "type": "embedding_error",
                "file_id": file_id,
                "message": str(embedding_result)
            })
            embedding_result = {"word_count": 0, "chunks": 0, "documents": []}
        
        # Extract documents for vectorstore
        documents_for_vectorstore = embedding_result.get("documents", [])
        
        # 3. Extract insights
        insights = embedding_result.get("insights")
        
        # Final update - embedding complete
        updates.append({
            "type": "file_complete",
            "file_id": file_id,
            "filename": filename,
            "word_count": embedding_result.get("word_count", 0),
            "chunk_count": embedding_result.get("chunks", 0),
            "status": "success"
        })
        
        # Return both updates and documents + file_bytes for background upload
        return {
            "updates": updates,
            "documents": documents_for_vectorstore,
            "filename": filename,
            "file_bytes": file_bytes,  # ← Include for background upload
            "insights": insights 
        }
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        
        updates.append({
            "type": "file_error",
            "file_id": file_id,
            "filename": filename,
            "message": str(e)
        })
        
        return {
            "updates": updates,
            "documents": [],
            "filename": filename,
            "file_bytes": file_bytes
        }
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"✅ Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup {temp_path}: {cleanup_error}")

async def embed_document_task(temp_path: str, filename: str, chat_id: str, file_id: str, updates: list, language: str = "english"):
    """Embed document and extract insights in parallel."""
    try:
        updates.append({
            "type": "embedding_start",
            "file_id": file_id
        })
        
        # Verify file exists
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temp file not found: {temp_path}")
        
        print(f"📄 Loading document: {filename}")

        # Load document in a worker thread — PDF parsing/OCR is blocking CPU/IO
        # work that would otherwise freeze the event loop and stall the
        # progress stream for every connected client
        loader = get_loader_for_file(temp_path)
        pages = await asyncio.to_thread(loader.load)

        print(f"✅ Loaded {len(pages)} pages from {filename}")
        
        updates.append({
            "type": "embedding_progress",
            "file_id": file_id,
            "stage": "loaded_pages",
            "page_count": len(pages)
        })
        
        # Extract text
        text = "\n\n".join([p.page_content for p in pages])
        word_count = len(text.split())
        
        print(f"📝 Extracted {word_count} words from {filename}")
        
        # ========================================
        # 🆕 START INSIGHT EXTRACTION IN PARALLEL
        # ========================================
        insight_task = asyncio.create_task(
            extract_file_insights_from_text(text, filename, chat_id, file_id, updates, language)
        )
        
        # ========================================
        # CONTINUE WITH EMBEDDING (PARALLEL)
        # ========================================
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        print(f"✂️ Split into {len(chunks)} chunks")
        
        updates.append({
            "type": "embedding_progress",
            "file_id": file_id,
            "stage": "chunked",
            "chunk_count": len(chunks)
        })
        
        # Create documents
        documents = [
            Document(page_content=chunk, metadata={"source": filename})
            for chunk in chunks
        ]
        
        print(f"🔤 Creating embeddings for {len(documents)} documents...")

        # Compute embeddings with the async client: the HTTP calls to OpenAI
        # run without blocking the event loop, so progress events keep
        # streaming and multiple files embed truly in parallel
        embeddings = OpenAIEmbeddings()
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = await embeddings.aembed_documents(texts)
        text_embeddings = list(zip(texts, vectors))

        # Get session
        if chat_id not in ACTIVE_SESSIONS:
            print(f"⚠️ No session found for {chat_id}, creating...")
            ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)

        session = ACTIVE_SESSIONS[chat_id]
        print(f"📤 Upload using session object id: {id(session.session)}")

        # FAISS index mutation isn't safe under concurrent writes — serialize
        # per chat (the slow embedding work above already happened in parallel)
        async with _get_vectorstore_write_lock(chat_id):
            if session.session.vectorstore:
                print(f"➕ Adding to existing vectorstore")
                session.session.vectorstore.add_embeddings(text_embeddings, metadatas=metadatas)
            else:
                print(f"🆕 Creating new vectorstore")
                session.session.vectorstore = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)

        print(f"✅ Embedding complete for {filename}")
        print(f"📤 Vectorstore now set: {session.session.vectorstore is not None}")
        
        updates.append({
            "type": "embedding_complete",
            "file_id": file_id,
            "word_count": word_count,
            "chunks": len(documents)
        })
        
        # ========================================
        # 🆕 WAIT FOR INSIGHTS (SHOULD BE READY)
        # ========================================
        try:
            insights = await asyncio.wait_for(insight_task, timeout=20.0)
        except asyncio.TimeoutError:
            print(f"⚠️ Insight extraction timed out for {filename}")
            insights = None
        
        # Return documents + insights
        return {
            "word_count": word_count,
            "chunks": len(documents),
            "documents": documents,
            "insights": insights  # ← Include insights
        }
        
    except Exception as e:
        print(f"❌ Embedding error for {filename}: {e}")
        import traceback
        traceback.print_exc()
        raise

async def firebase_upload_task(file_bytes, filename, chat_id, file_id, updates):
    """Upload to Firebase - appends progress to updates list"""
    try:
        updates.append({
            "type": "firebase_start",
            "file_id": file_id
        })
        
        bucket = storage.bucket()
        blob = bucket.blob(f"chats/{chat_id}/uploads/{filename}")
        
        # Upload
        blob.upload_from_string(
            file_bytes,
            content_type=get_content_type(filename)
        )
        
        # Make public and get URL
        blob.make_public()
        firebase_url = blob.public_url
        
        updates.append({
            "type": "firebase_complete",
            "file_id": file_id,
            "firebase_url": firebase_url
        })
        
        return firebase_url
        
    except Exception as e:
        print(f"Firebase upload error for {filename}: {e}")
        raise
         
def get_content_type(filename: str) -> str:
    """
    Get MIME type from filename using Python's standard library.
    
    Args:
        filename: Name of the file (e.g., "notes.pdf")
    
    Returns:
        MIME type string (e.g., "application/pdf")
    """
    # Guess MIME type from filename
    content_type, _ = mimetypes.guess_type(filename)
    
    # If unknown, default to generic binary
    return content_type or 'application/octet-stream'

async def save_temp_file(file_bytes: bytes, filename: str) -> str:
    """
    Save uploaded file bytes to a temporary file.
    
    Args:
        file_bytes: The file content as bytes
        filename: Original filename (used to get extension)
    
    Returns:
        Path to the temporary file
    """
    temp_dir = get_temp_dir()
    suffix = os.path.splitext(filename)[-1]  # Get file extension (.pdf, .docx, etc.)
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as f:
        f.write(file_bytes)
        temp_path = f.name
    
    print(f"✅ Saved temp file: {temp_path}")
    return temp_path

async def ensure_session_with_vectorstore(chat_id: str):
    """
    Ensure session exists and has vectorstore loaded.
    Shows "reading documents" message when downloading from Firebase.
    """
    
    if chat_id in ACTIVE_SESSIONS:
        print(f"✅ Session already exists for {chat_id}")
        return ACTIVE_SESSIONS[chat_id]
    
    print(f"🆕 Creating new session for {chat_id}")
    
    # Create session
    ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)
    
    # Try to load existing vectorstore
    print(f"📚 Checking for existing documents...")
    vectorstore = await vectorstore_manager.load_combined_vectorstore_from_firebase(chat_id)
    
    if vectorstore:
        ACTIVE_SESSIONS[chat_id].session.vectorstore = vectorstore
        print(f"✅ Loaded existing vectorstore into session")
    else:
        print(f"📝 No existing vectorstore, will create new one")
    
    return ACTIVE_SESSIONS[chat_id]

# ============================================================================
# HELPERS FOR FILE UPLOAD END
# ============================================================================

@app.post("/chat/generate-summary")
async def generate_summary(request:SummaryRequest):
    
    from tools.quiztools import _search_vectorstore_for_summary,set_session_context
     # Get or create session for this chat
    if request.chat_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
        print(f"Created new session for chat_id: {request.chat_id}")
    
    # GET TUTOR FOR CURRENT SESSION
    nursing_tutor = ACTIVE_SESSIONS[request.chat_id]
    print(f"nursing tutor created")
        
    
    nursing_tutor.session.name_last_document_used=request.filename
    
    set_session_context(nursing_tutor.session)
    
    from tools.quiztools import _search_vectorstore_for_summary
    chunks = await _search_vectorstore_for_summary(request.filename, request.chat_id, "", "detailed")
    
    print("Got the chunks for summary through endpoint")
    
    async def json_chunk_generator():
       async for chunk in   nursing_tutor.stream_document_summary(
           relevant_chunks=chunks,
           detail_level="detailed",
           filename=request.filename,
           language=request.language):
            yield json.dumps({
                "answer_chunk": chunk
            }) + "\n"
    
    print("Streaming summary through endpoint response")
    
    return StreamingResponse(
        json_chunk_generator(),
        media_type="application/json"
    )
 
@app.post("/plan")
async def create_plan(request: PlanRequest):
    from tools.quiztools import search_documents
     
    print("building plan for study guide",request)
    
    session = ACTIVE_SESSIONS[request.chat_id]
    
    try:
        # Get context using your existing tool
        search_result = await search_documents.ainvoke({
            "query": request.topic
        })
        
        context = search_result.get("context", "")
        
        
        # STEP 1: Create a prompt asking LLM to generate a plan
        prompt = f"""
        Create {request.num_sections} sections for a study guide about {request.topic}.
        base in this context {context}
        in this language {session.session.user_language}
        Return ONLY a JSON array:
        [
        {{"id": "introduction", "title": "Introduction", "color": "blue"}},
        {{"id": "concepts", "title": "Key Concepts", "color": "green"}}
        ]
        """
        # Section-title JSON is a trivial structured task — mini handles it for ~5x less.
        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.3
        )
        
        # STEP 2: Send prompt to LLM (this generates the actual plan)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])

        # STEP 3: LLM returns a string that looks like JSON
        plan_json = response.content.strip()
        # plan_json is now a STRING: '[{"id":"intro","title":"Introduction"...}]'

        # STEP 4: Clean up markdown code blocks if LLM wrapped it
        if plan_json.startswith("```"):
            plan_json = plan_json.split("```")[1]
            if plan_json.startswith("json"):
                plan_json = plan_json[4:]
            plan_json = plan_json.strip()

        # STEP 5: Convert JSON string to Python list/dict
        sections = json.loads(plan_json)  # ← This parses the string into actual Python objects
        # sections is now a Python LIST: [{"id": "intro", "title": "Introduction"...}]
        # Generate plan...
        
        # Return BOTH sections AND context
        plan = {
            "sections": sections,
            "context": context  # ← Add this
        }
    
        print("This is the plan", plan)
        
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search")
async def search_for_context(request: dict):
    """
    Reuse your existing search_documents tool to get context
    """
    from tools.quiztools import search_documents
    
    try:
        # Call your existing search tool
        result = await search_documents.ainvoke({
            "query": request.get("query"),
            # Add any other params your search tool needs
        })
        
        # Extract context string
        context = result.get("context", "")
        
        return {"context": context}
        
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
@app.post("/generate-section")
async def generate_section(request: SectionRequest):
    """Generate content for a section using RAG context"""
    
    print("GENERATING SECTION BASED ON",request)
    
    try:
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)    
        prompt = f"""
        You are creating educational content for a study guide.

        Topic: {request.topic}
        Section: {request.section_title}

        Retrieved Context from Student's Documents:
        {request.context}

        Generate comprehensive educational content for this section using ONLY information from the documents.

        Format as HTML:
        - <h3>Subsection Title</h3>
        - <p>Explanation text</p>
        - <ul><li>Bullet points</li></ul>

        - <div class="card card-blue">
            <div class="card-title">🔑 Key Concept</div>
            <p>Important nursing information</p>
        </div>

        - <div class="card card-green">
            <div class="card-title">✅ Clinical Application</div>
            <p>How to apply this in practice</p>
        </div>

        - <div class="card card-yellow">
            <div class="card-title">⚠️ Critical Alert</div>
            <p>Warning or safety information</p>
        </div>

        - Use <strong> for emphasis
        - Use <span class="highlight">term</span> for key terms

        Guidelines:
        - Use nursing emojis (🩺 💊 🫁 ❤️ 🧠)
        - Include rationales (WHY, not just WHAT)
        - Focus on NCLEX-style critical thinking
        - Be comprehensive but concise
        - Use proper medical terminology
        
        IMPORTANT:
        Make sure all the content written are in the same language, either french or english
        It should be uniform from the header titles, to critical alert, key concepts etc

        Return ONLY the HTML content, no markdown code blocks.
        """
        
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        # Clean up
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("html"):
                content = content[4:]
            content = content.strip()
        
        return {"content": content}
        
    except Exception as e:
        print(f"Error generating section: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STUDY MODE ENDPOINTS
# Duolingo-style learning path generation (NOT USED FOR CHATINTERFACE, IT IS FOR STUDY PLAN VERY SEPARATE)
# ============================================================================

from models.requests import StudyPlanRequest, StudyItemRequest, StudyAudioRequest, StudyReviewPlanRequest, DiagnosticQuizRequest, StudyMindmapRequest, StudyInterpretRequest, StudyExamRequest, NodeDebriefRequest, NarrationRequest
from models.requests import ExamDebriefTurnRequest
from services.exam_debrief import run_debrief_turn as run_exam_debrief_turn
import hashlib

# ── Study node sizes ────────────────────────────────────────────────────────
# Both were 12. Production analysis (2026-08-03) showed:
#   - flashcards are 2.03x over-represented as the node people quit on, and
#     completing one DROPS the odds of doing the next node by 6.5pp — a
#     fatigue signature pointing at set length, not at the format itself
#   - quizzes are the best momentum node in the product (+5.1pp continuation),
#     so the plan should contain MORE of them, each one shorter
# Shorter units also mean a free user's question budget buys ~2 quizzes per
# window instead of one, so the throttle stops cutting mid-node.
# Keep STUDY_QUIZ_QUESTIONS in mind alongside FREE_LIMIT in usage_guard.py.
STUDY_QUIZ_QUESTIONS = 5
STUDY_FLASHCARD_CARDS = 5
STUDY_DIAGNOSTIC_QUESTIONS = 3   # legacy in-plan calibration node (pre-diagnostic flow)

# Pre-plan diagnostic. Six, not five: two questions each on the three topics
# most likely to be moved to the refresh tail, plus singles on the rest.
#
# Two is the minimum evidence for deciding a student is solid on something.
# One question is a coin flip on four options, and the cost of that coin
# landing wrong is not a wasted node — it is telling her she is strong on a
# topic she will meet again on the exam. The extra question costs about
# fifteen seconds against the "quick questions" promise.
DIAGNOSTIC_QUESTION_COUNT = 6
DIAGNOSTIC_DEEP_TOPICS = 3       # topics that get two questions
DIAGNOSTIC_TOPIC_LIMIT = 4       # topics covered at all

# Question formats a study-plan QUIZ node may emit (2026-08-26).
# Was ["mcq"] — which meant a student's first quiz was always the one format
# they are already good at. Measured on the paying cohort: 33/33 correct on
# plain MCQ against 9/28 on SATA + case study + prioritisation. That gap is
# what makes a student realise they are not ready, and every subscriber we can
# trace a reason for converted after meeting it.
# It used to arrive only at node 2 (or via an adaptive branch), but ~39% of
# study sessions stop at or before node 1 — so 4 in 10 students only ever saw
# multiple choice, aced it, and left believing they were fine.
# distribute_question_types keeps the mix MCQ-weighted (gentle on-ramp) while
# guaranteeing at least one SATA per node, so nobody finishes a quiz node
# without meeting the format they actually struggle with.
STUDY_QUIZ_TYPES = ["mcq", "sata"]


def _format_study_question(q: dict, fallback_topic: str) -> dict:
    """Shape one generated question for the study-mode cards.

    Mirrors the dispatch in /study/generate-exam so a quiz node can carry the
    same mixed formats an exam node already does. MCQ keeps its historical
    shape — the letter answer flattened to `correctIndex` — because
    StudyQuizCard grades on that field; SATA and case study pass through whole,
    since those components read the generator's native fields directly.

    `questionType` is now always set. It was absent before, when quiz nodes
    were MCQ-only and the frontend could safely assume the format.
    """
    q_type = q.get("questionType", "mcq")

    if q_type in ("sata", "casestudy", "unfoldingCase"):
        return {**q, "questionType": q_type, "topic": q.get("topic", fallback_topic)}

    answer = q.get("answer", "A)")
    answer_letter = answer[0] if answer else "A"
    return {
        "questionType": "mcq",
        "question": q.get("question", ""),
        "options": q.get("options", []),
        "correctIndex": ord(answer_letter) - ord("A"),
        # Legacy full rationale (empty on new generations).
        "rationale": q.get("justification", ""),
        # New one-sentence summary; full rationale fetched on demand.
        "correctBlurb": q.get("correct_blurb", ""),
        "topic": q.get("topic", fallback_topic),
    }


@app.post("/study/plan")
async def generate_study_plan(request: StudyPlanRequest):
    """
    Generate a personalized study path based on uploaded documents.

    HOW IT WORKS:
    1. Get file insights (topics, concepts) from the session
    2. Use LLM to create a logical learning sequence
    3. Return a list of nodes (lesson, flashcard, quiz, audio)

    Each node has:
    - id: unique identifier
    - type: "lesson" | "flashcard" | "quiz" | "audio"
    - label: topic name shown to user
    - tags: context tags for content generation
    - difficulty: 1-3 scale
    - status: "locked" | "available" | "completed"

    Frontend will store this in Firestore and track progress.
    """
    print(f"\n{'='*60}")
    print(f"📚 STUDY PATH GENERATION - chat_id: {request.chat_id}")
    print(f"{'='*60}")

    # Plan-creation gate. Raised BEFORE the try so the blanket except below
    # can't swallow it into a 500 (same pattern as the question gate on
    # /study/generate-item).
    plan_quota = usage_guard.check_plan_quota(request.chat_id)
    if not plan_quota["allowed"]:
        print(f"🚫 Plan quota exceeded for chat {request.chat_id} — rejecting /study/plan")
        raise HTTPException(status_code=429, detail=usage_guard.PLAN_QUOTA_MESSAGE)

    try:
        # ------------------------------------------
        # STEP 1: Get or create session & load insights
        # ------------------------------------------
        if request.chat_id not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
            await ACTIVE_SESSIONS[request.chat_id].load_file_insights_from_firebase()
            print(f"🆕 Created new session for study plan")

        session = ACTIVE_SESSIONS[request.chat_id]
        file_insights = getattr(session.session, "file_insights", {})

        # ------------------------------------------
        # STEP 2: Collect all topics and concepts
        # ------------------------------------------
        all_topics = []
        all_concepts = []

        for filename, insights in file_insights.items():
            if insights:
                all_topics.extend(insights.get("topics", []))
                all_concepts.extend(insights.get("concepts", []))

        # Remove duplicates
        unique_topics = list(set(all_topics))[:8]  # Max 8 topics for manageable path
        unique_concepts = list(set(all_concepts))[:15]

        print(f"📊 Found {len(unique_topics)} topics: {unique_topics}")
        print(f"📊 Found {len(unique_concepts)} concepts")

        # ------------------------------------------
        # STEP 3: ALWAYS extract topics from actual document content
        # ------------------------------------------
        document_content = ""
        if session.session.vectorstore:
            # Get comprehensive document content
            docs = session.session.vectorstore.similarity_search("main topics concepts definitions", k=1000)
            document_content = "\n\n".join([doc.page_content for doc in docs])[:15000]
            print(f"📄 Retrieved {len(docs)} document chunks for topic extraction")

        # Extract CORE topics from the actual document
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        prompt_language = _language_for_prompt(request.language)

        topic_extraction_prompt = f"""Analyze this document and extract the 3-5 MAIN TOPICS that the student needs to learn.

🚨 CRITICAL: Only extract topics that are ACTUALLY IN the document below.
DO NOT invent topics. DO NOT add general knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT CONTENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{document_content[:8000]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. Identify the 3-5 MAIN TOPICS/SECTIONS in this document
2. Name the topics based on the document content, but EXPRESS THEM IN {prompt_language}
3. Topics should be what the document is teaching (not generic categories)
4. Order from foundational to advanced

Return ONLY valid JSON (all strings must be in {prompt_language}):
{{
    "main_topics": ["Topic 1 in {prompt_language}", "Topic 2 in {prompt_language}", "Topic 3 in {prompt_language}"],
    "key_terms": ["term1", "term2", "term3", "term4", "term5"]
}}"""

        topic_response = await llm.ainvoke([{"role": "user", "content": topic_extraction_prompt}])

        try:
            topic_json = topic_response.content.strip()
            if topic_json.startswith("```"):
                topic_json = topic_json.split("```")[1]
                if topic_json.startswith("json"):
                    topic_json = topic_json[4:]
                topic_json = topic_json.strip()

            extracted = json.loads(topic_json)
            unique_topics = extracted.get("main_topics", unique_topics)[:5]
            key_terms = extracted.get("key_terms", [])[:10]
            print(f"✅ Extracted topics from document: {unique_topics}")
            print(f"✅ Key terms: {key_terms}")
        except Exception as e:
            print(f"⚠️ Topic extraction failed: {e}, using file insights")
            key_terms = unique_concepts[:10]

        if not unique_topics:
            unique_topics = ["Document Overview"]

        # ------------------------------------------
        # STEP 4: Generate learning path with LLM
        # ------------------------------------------
        user_prefs = request.userPreferences or {}
        review_format = user_prefs.get("reviewFormat", "Visual Concept Maps")

        # Single source of truth for the path prompt — this endpoint used to
        # carry a byte-for-byte duplicate of _build_study_path_prompt, which
        # meant every change had to be made twice or the /study/plan fallback
        # would hand the student a differently-shaped plan than /study/start.
        path_prompt = _build_study_path_prompt(
            unique_topics,
            key_terms,
            review_format,
            prompt_language,
            days_to_exam=_days_to_exam(user_prefs),
            hardest_topics=user_prefs.get("hardestTopics") or [],
        )

        response = await llm.ainvoke([{"role": "user", "content": path_prompt}])

        # Parse the response
        path_json = response.content.strip()

        # Clean markdown code blocks if present
        if path_json.startswith("```"):
            path_json = path_json.split("```")[1]
            if path_json.startswith("json"):
                path_json = path_json[4:]
            path_json = path_json.strip()

        nodes = json.loads(path_json)

        # ------------------------------------------
        # STEP 5: Weight by the diagnostic, then attach status + exam nodes
        # ------------------------------------------
        # Both steps used to be inlined here as a byte-for-byte copy of
        # _attach_status_and_exam_nodes. Two copies meant every change had to
        # be made twice, or /study/plan — the StartStudyModal fallback path —
        # quietly handed the student a differently shaped plan than
        # /study/start did.
        nodes = _weight_path_by_diagnostic(
            nodes,
            request.diagnostic,
            unique_topics,
            _days_to_exam(user_prefs),
        )
        nodes = _attach_status_and_exam_nodes(nodes, unique_topics)

        print(f"✅ Final study path with {len(nodes)} nodes (including exams)")

        # ------------------------------------------
        # STEP 6: Return the study path
        # ------------------------------------------
        return {
            "nodes": nodes,
            "topics": unique_topics,
            "total_nodes": len(nodes),
            "estimated_time_minutes": len(nodes) * 3,  # ~3 min per node
            "archetype": _plan_archetype(_days_to_exam(user_prefs)),
            "tiers": _summarize_tiers(nodes, unique_topics),
        }

    except Exception as e:
        print(f"❌ Study path generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# /study/start — fast-path that combines plan + first-node generation
# ============================================================================
# WHY: The legacy two-step flow (POST /study/plan, then POST /study/generate-item-stream)
# does three sequential round trips before the user sees any node content:
#   1. /study/plan (~5–15s with redundant topic extraction LLM call)
#   2. createStudySession Firestore write
#   3. /study/generate-item-stream for the first node (~3–10s)
#
# /study/start collapses this into one SSE stream that:
#   - skips the redundant topic-extraction LLM call (uses file_insights topics directly)
#   - emits `plan_ready` as soon as the path JSON is parsed (frontend can navigate)
#   - immediately proceeds to generate the first node's content in the same request
#   - emits `first_node_ready` when content is done (frontend uses prefetched cache)
#
# Net effect: the LLM time for plan + first-node OVERLAPS the Firestore write and
# the route transition, instead of running sequentially. Time-to-first-node drops
# significantly without changing any node-content rendering.
# ============================================================================

def _build_deadline_path_prompt(
    unique_topics, key_terms, prompt_language, archetype, days_to_exam, hardest_topics
) -> str:
    """Prompt for the SPRINT and FOCUS shapes — plans built against a deadline."""
    focus_line = (
        f"\nThe student says these are hardest for them: {hardest_topics}. "
        "Cover these FIRST and give them the most nodes.\n"
        if hardest_topics else "\n"
    )

    if archetype == "sprint":
        when = "today" if days_to_exam == 0 else ("tomorrow" if days_to_exam == 1 else f"in {days_to_exam} days")
        return f"""Create a LAST-MINUTE triage study path. The student's exam is {when}.

🚨 THERE IS NO TIME TO TEACH NEW MATERIAL. Do NOT include "lesson", "audio",
"flashcard" or "mindmap" nodes. The only job of this path is to find out what
the student does not know and drill exactly that.

CORE TOPICS from their document:
{unique_topics}

KEY TERMS: {key_terms}
{focus_line}
STRUCTURE — 6 to 8 nodes TOTAL, no more:
1. "quiz" — rapid check across the highest-yield material (label it "<Topic> - Quick Check")
2. For each of the 2-3 most important topics, in priority order:
   - "exam": a short mini-test on that topic
   - "quiz": drill on the specific ideas that mini-test covers
3. Finish with ONE "quiz" labelled "<main topic> - Final Check"

RULES:
- Prioritise breadth of coverage over depth. High-yield only.
- Every label MUST name a real topic from the document.
- ALL labels in {prompt_language}.

Return ONLY a valid JSON array:
[
  {{"id": "node_1", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quick Check", "tags": ["topic1", "assessment"], "difficulty": 1}},
  {{"id": "node_2", "type": "exam", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Mini-Test", "tags": ["topic1", "assessment"], "difficulty": 2}},
  {{"id": "node_3", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Drill", "tags": ["topic1", "assessment"], "difficulty": 2}}
]"""

    # FOCUS — 3 to 9 days out
    return f"""Create a FOCUSED review path. The student's exam is in {days_to_exam} days.

There is time to relearn weak areas, but NOT to cover everything from scratch.
Lead with assessment so the path targets real gaps rather than guessing.

CORE TOPICS from their document:
{unique_topics}

KEY TERMS: {key_terms}
{focus_line}
STRUCTURE — 10 to 14 nodes TOTAL:
1. Open with ONE "quiz" across the material ("<Topic> - Quick Check")
2. Then per topic, hardest first:
   - "lesson": a tight refresher on that topic
   - "quiz": check it landed
   - "exam": mini-test to confirm
3. Close with ONE "exam" labelled "<main topic> - Final Test"

RULES:
- Do NOT include "audio" or "mindmap" nodes — there isn't time.
- At most ONE "flashcard" node, and only for a heavy terminology topic.
- Every label MUST name a real topic from the document.
- ALL labels in {prompt_language}.

Return ONLY a valid JSON array:
[
  {{"id": "node_1", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quick Check", "tags": ["topic1", "assessment"], "difficulty": 1}},
  {{"id": "node_2", "type": "lesson", "label": "{unique_topics[0] if unique_topics else 'Topic 1'}", "tags": ["topic1"], "difficulty": 1}},
  {{"id": "node_3", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quiz", "tags": ["topic1", "assessment"], "difficulty": 2}},
  {{"id": "node_4", "type": "exam", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Mini-Test", "tags": ["topic1", "assessment"], "difficulty": 2}}
]"""


def _days_to_exam(user_prefs: dict):
    """Days until the student's exam, or None if they didn't give a date.

    PlanOnboarding sends both `examDaysAway` (computed client-side) and an ISO
    `examDate`. Prefer the raw date so a plan built today for an exam set last
    week isn't using a stale day count.
    """
    if not user_prefs:
        return None
    iso = user_prefs.get("examDate")
    if iso:
        try:
            from datetime import datetime, timezone
            exam = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            if exam.tzinfo is None:
                exam = exam.replace(tzinfo=timezone.utc)
            return max(0, (exam - datetime.now(timezone.utc)).days)
        except Exception:
            pass
    days = user_prefs.get("examDaysAway")
    return int(days) if isinstance(days, (int, float)) else None


# ── Plan archetypes by time-to-exam ─────────────────────────────────────────
# The exam date has been collected since June and never reached the generator,
# so a student sitting an exam tomorrow got the same 15-20 node plan as someone
# three weeks out — and did two lessons before running out of time.
#
# Node mix per archetype is driven by measured behaviour (2026-08-03):
#   quiz      +5.1pp continuation — the momentum node, so it leads and closes
#   exam      0.32x quit rate     — the MOST tolerated node, and underused
#   lesson    best raw completion, but -3.2pp continuation
#   flashcard -6.5pp continuation — latest slot, or dropped under time pressure
SPRINT_MAX_DAYS = 2      # can't teach new material; triage and drill
FOCUS_MAX_DAYS = 9       # shore up weak spots


def _plan_archetype(days_to_exam):
    if days_to_exam is None:
        return "master"
    if days_to_exam <= SPRINT_MAX_DAYS:
        return "sprint"
    if days_to_exam <= FOCUS_MAX_DAYS:
        return "focus"
    return "master"


def _build_study_path_prompt(
    unique_topics: List[str],
    key_terms: List[str],
    review_format: str,
    prompt_language: str,
    days_to_exam=None,
    hardest_topics=None,
) -> str:
    """Build the study-path planning prompt.

    Shape is chosen by time-to-exam first (sprint / focus / master); the user's
    review_format preference only applies to the master shape, because under
    deadline pressure the format that works is not a preference question.
    """
    archetype = _plan_archetype(days_to_exam)
    hardest_topics = [t for t in (hardest_topics or []) if t]

    if archetype in ("sprint", "focus"):
        return _build_deadline_path_prompt(
            unique_topics, key_terms, prompt_language,
            archetype, days_to_exam, hardest_topics,
        )

    if review_format == "Flashcards":
        structure_rule = "- AUDIO: Listen to an intro for that topic\n   - FLASHCARD: Key terms and definitions from that topic\n   - FLASHCARD: Advanced concepts\n   - QUIZ: Quick test"
        node_types = '- "audio": Short audio intro for the topic\n- "flashcard": Key terms\n- "quiz": Questions testing that topic'
        example_json_rows = f"""  {{"id": "node_1", "type": "audio", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Listen", "tags": ["topic1", "audio"], "difficulty": 1}},
  {{"id": "node_2", "type": "flashcard", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Basics", "tags": ["topic1", "terms"], "difficulty": 1}},
  {{"id": "node_3", "type": "flashcard", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Advanced", "tags": ["topic1", "advanced"], "difficulty": 2}},
  {{"id": "node_4", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quiz", "tags": ["topic1", "assessment"], "difficulty": 1}}"""
    elif review_format == "Audio Summaries":
        structure_rule = "- AUDIO: Listen to a summary\n   - LESSON: Read the details\n   - QUIZ: Test understanding"
        node_types = '- "audio": Listen to summary\n- "lesson": Read details\n- "quiz": Questions testing that topic'
        example_json_rows = f"""  {{"id": "node_1", "type": "audio", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Audio", "tags": ["topic1", "audio"], "difficulty": 1}},
  {{"id": "node_2", "type": "lesson", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Lesson", "tags": ["topic1", "lesson"], "difficulty": 2}},
  {{"id": "node_3", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quiz", "tags": ["topic1", "assessment"], "difficulty": 1}}"""
    elif review_format == "Practice Questions":
        structure_rule = "- QUIZ: Initial assessment\n   - LESSON: Review concepts\n   - AUDIO: Listen to a recap\n   - QUIZ: Final test"
        node_types = '- "quiz": Assessment questions\n- "lesson": Review content\n- "audio": Short audio recap'
        example_json_rows = f"""  {{"id": "node_1", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Pre-Test", "tags": ["topic1", "assessment"], "difficulty": 1}},
  {{"id": "node_2", "type": "lesson", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Review", "tags": ["topic1", "lesson"], "difficulty": 2}},
  {{"id": "node_3", "type": "audio", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Recap", "tags": ["topic1", "audio"], "difficulty": 1}},
  {{"id": "node_4", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Final Test", "tags": ["topic1", "assessment"], "difficulty": 2}}"""
    elif review_format == "Visual Concept Maps":
        structure_rule = "- LESSON: Introduce the topic (content comes from document)\n   - AUDIO: Listen to the lesson explained out loud\n   - MINDMAP: Visual concept map linking key ideas\n   - QUIZ: Test understanding of that specific topic"
        node_types = f'- "lesson": Introduction to ONE topic from the document\n- "audio": Audio explanation of that topic (listen instead of reading)\n- "mindmap": Visual concept map for that topic\n- "quiz": Questions testing that topic (will generate {STUDY_QUIZ_QUESTIONS} questions)'
        example_json_rows = f"""  {{"id": "node_1", "type": "lesson", "label": "{unique_topics[0] if unique_topics else 'Topic 1'}", "tags": ["topic1"], "difficulty": 1}},
  {{"id": "node_2", "type": "audio", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Listen", "tags": ["topic1", "audio"], "difficulty": 1}},
  {{"id": "node_3", "type": "mindmap", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Concept Map", "tags": ["topic1", "visual"], "difficulty": 1}},
  {{"id": "node_4", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quiz", "tags": ["topic1", "assessment"], "difficulty": 1}}"""
    else:
        # DEFAULT UNIT — reordered from production data (2026-08-03):
        #   lesson  → highest completion (28.8%), best opener
        #   quiz    → the momentum node (+5.1pp continuation), so it lands
        #             EARLY and again at the end; two short 5-question quizzes
        #             beat one long 12-question one
        #   audio   → neutral (85.0% continuation), sits mid-unit
        #   flashcard → costs 6.5pp of continuation, so it comes late and
        #             short (5 cards); a closing quiz restores momentum
        structure_rule = (
            "- LESSON: Introduce the topic (content comes from document)\n"
            "   - QUIZ: Short check on what was just introduced\n"
            "   - AUDIO: Listen to the topic explained out loud\n"
            "   - FLASHCARD: Key terms and definitions from that topic\n"
            "   - QUIZ: Final check on that topic"
        )
        node_types = (
            '- "lesson": Introduction to ONE topic from the document\n'
            '- "audio": Audio explanation of that topic (listen instead of reading)\n'
            f'- "flashcard": Key terms from that topic (will generate {STUDY_FLASHCARD_CARDS} cards)\n'
            f'- "quiz": Questions testing that topic (will generate {STUDY_QUIZ_QUESTIONS} questions)'
        )
        example_json_rows = f"""  {{"id": "node_1", "type": "lesson", "label": "{unique_topics[0] if unique_topics else 'Topic 1'}", "tags": ["topic1"], "difficulty": 1}},
  {{"id": "node_2", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Quiz", "tags": ["topic1", "assessment"], "difficulty": 1}},
  {{"id": "node_3", "type": "audio", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Listen", "tags": ["topic1", "audio"], "difficulty": 1}},
  {{"id": "node_4", "type": "flashcard", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Vocabulaire", "tags": ["topic1", "terms"], "difficulty": 1}},
  {{"id": "node_5", "type": "quiz", "label": "{unique_topics[0] if unique_topics else 'Topic 1'} - Final Test", "tags": ["topic1", "assessment"], "difficulty": 2}}"""

    return f"""Create a Duolingo-style study path for this document.

🚨 CRITICAL: The study path MUST be structured around these CORE TOPICS from the student's document:
{unique_topics}

KEY TERMS to include: {key_terms}

STRUCTURE RULES:
1. Create ONE learning unit per main topic (3-5 units total)
2. Each unit follows this pattern:
   {structure_rule}

3. Total: 12-20 nodes. Emit EVERY node of the unit pattern for each topic —
   if that would exceed 20 nodes, use FEWER TOPICS rather than dropping nodes
   from a unit. A complete short unit beats a truncated long one.
4. Progress through topics in logical order

NODE TYPES:
{node_types}

Return ONLY valid JSON array in {prompt_language}:
[
{example_json_rows},
  ... (repeat for each main topic)
]

IMPORTANT:
- Node labels MUST reference the actual topics from the document
- DO NOT create generic labels like "Introduction to Medicine"
- ALL labels MUST be written in {prompt_language} — translate topic names if the document is in a different language
- Keep labels concise and meaningful"""


# ══════════════════════════════════════════════════════════════════════════
# PLAN WEIGHTING — time sets the budget, the diagnostic sets the order.
#
# Neither input is enough on its own. The diagnostic knows what MATTERS; only
# the calendar knows how much FITS. A plan built from the diagnostic alone
# hands a two-day crammer eighteen nodes; a plan built from the calendar alone
# is what we ship today — the same uniform path whether she knows everything
# or nothing.
#
# Order is: gaps first, then shaky, then the topics she is already solid on as
# a closing refresh. That tail is not padding. It does three things:
#
#   1. The plan does not visibly collapse when she happens to know a lot, so a
#      personalised plan never reads as a cheaper one.
#   2. Nothing is REMOVED on the strength of one or two diagnostic questions —
#      only deferred. A bad read costs her ordering, not coverage. Being
#      wrongly told she is strong is the one failure here that costs an exam.
#   3. It ends the plan on material she is good at, a day or two before the
#      exam — which is exactly what getExamPhase() on the frontend already
#      asks for ("final: consolidate, stop starting new material") and has
#      never had content designed for.
#
# The property worth protecting: FRONT-LOAD BY NEED, BACK-LOAD BY CONFIDENCE.
# Most plans are abandoned. Under this ordering whoever falls off has lost
# only the review of material she already knew.
# ══════════════════════════════════════════════════════════════════════════

# Node budget per archetype, before exam nodes are attached. Sprint cannot
# teach new material, so it triages; master has the calendar room to cover
# everything properly.
PLAN_BUDGETS = {"sprint": 8, "focus": 14, "master": 20}

GAP_MAX_PCT = 40    # below this she has not got it
SOLID_MIN_PCT = 80  # at or above this it goes to the tail

# The unit shape each tier earns. Gap keeps the full default unit — the one
# production data settled on (lesson opens, quiz carries momentum, flashcards
# late and short, quiz closes).
TIER_UNITS = {
    "gap":      ["lesson", "quiz", "audio", "flashcard", "quiz"],
    "shaky":    ["lesson", "quiz"],
    "untested": ["lesson", "quiz"],
    "solid":    ["flashcard", "quiz"],
}

# Worst first inside the plan; solid always last.
TIER_ORDER = {"gap": 0, "shaky": 1, "untested": 2, "solid": 3}


def _tier_for_score(pct):
    """Bucket a diagnostic percentage. None means we never asked."""
    if pct is None:
        return "untested"
    if pct < GAP_MAX_PCT:
        return "gap"
    if pct >= SOLID_MIN_PCT:
        return "solid"
    return "shaky"


def _match_topic(name, candidates):
    """Loose topic match. Diagnostic topic names and plan topic names are both
    model-generated in the same session, so they usually agree — but "usually"
    is not "always", and an unmatched topic silently becomes untested and gets
    taught from scratch to someone who already knows it."""
    if not name:
        return None
    low = str(name).strip().lower()
    for c in candidates:
        if str(c).strip().lower() == low:
            return c
    for c in candidates:
        cl = str(c).strip().lower()
        if cl and (cl in low or low in cl):
            return c
    return None


def _topic_of_node(node, unique_topics):
    """Which curriculum topic a generated node belongs to."""
    label = str(node.get("label", ""))
    match = _match_topic(label, unique_topics)
    if match:
        return match
    # Labels are built as "<topic> - <kind>"; try the part before the dash.
    head = label.split(" - ")[0].strip()
    return _match_topic(head, unique_topics) or (head or None)


def _synth_node(topic_label, node_type, seq):
    """Build a missing node for a unit.

    The label is the BARE topic name, never a decorated one like
    "<topic> - Quiz". Plans are generated in the student's language and this
    function has no idea what that language is — appending an English suffix
    would drop "Quiz" into the middle of a French path. The node type already
    has its own icon, so the suffix carries nothing she cannot see.
    """
    return {
        "id": "synth_%s_%d" % (node_type, seq),
        "type": node_type,
        "label": topic_label,
        "tags": [node_type],
        "difficulty": 1,
    }


def _shape_unit(topic_label, pool, wanted_types, seq_start):
    """Fit this topic's generated nodes to the shape its tier earns.

    Reuses generated nodes wherever the type lines up — they carry real labels
    and tags from the document — and synthesises only what is missing.
    """
    remaining = list(pool)
    out = []
    seq = seq_start
    for t in wanted_types:
        match = next((n for n in remaining if n.get("type") == t), None)
        if match:
            remaining.remove(match)
            out.append(match)
        else:
            out.append(_synth_node(topic_label, t, seq))
            seq += 1
    return out


def _ensure_lesson_first(units):
    """The first node of a plan is always a lesson.

    Measured: lesson-first plans complete their first node 89.0% of the time,
    quiz-first 66.6% — with content successfully delivered in both cases. The
    opening node is the highest-leverage position in the path, and this is the
    cheapest thing we know that moves it.

    Every tier's unit already opens on a lesson except `solid`, so this only
    bites when a student is solid on absolutely everything.
    """
    if not units:
        return units
    first = units[0]
    nodes = first["nodes"]
    idx = next((i for i, n in enumerate(nodes) if n.get("type") == "lesson"), None)
    if idx is None:
        nodes.insert(0, _synth_node(first["topic"], "lesson", 900))
    elif idx > 0:
        nodes.insert(0, nodes.pop(idx))
    return units


def _apply_budget(units, budget, archetype):
    """Fit the plan to the calendar.

    Trim order is deliberate: the tail goes first, then shaky topics, and gap
    units are never truncated — a whole gap unit is dropped before a partial
    one is kept. Truncating loses the closing quiz, the node that carries
    momentum into the next topic, so half a unit is worth less than no unit.
    """
    if archetype == "sprint":
        # Two days out, five separate refresh units is not a plan, it is a
        # list. Collapse the whole tail into one review node.
        solid = [u for u in units if u["tier"] == "solid"]
        units = [u for u in units if u["tier"] != "solid"]
        if solid:
            labels = ", ".join(u["topic"] for u in solid if u["topic"])[:120]
            units.append({
                "topic": labels or "Review",
                "tier": "solid",
                "nodes": [_synth_node(labels or "Review", "quiz", 950)],
            })

    def total():
        return sum(len(u["nodes"]) for u in units)

    # 1. Drop tail units from the back.
    while total() > budget and any(u["tier"] == "solid" for u in units):
        last_solid = max(i for i, u in enumerate(units) if u["tier"] == "solid")
        units.pop(last_solid)

    # 2. Drop shaky units from the back.
    while total() > budget and any(u["tier"] in ("shaky", "untested") for u in units):
        last = max(i for i, u in enumerate(units) if u["tier"] in ("shaky", "untested"))
        units.pop(last)

    # 3. Still over: drop whole gap units from the back. Never truncate one.
    while total() > budget and len(units) > 1:
        units.pop()

    return units


def _summarize_tiers(nodes, unique_topics):
    """Topics grouped by tier, for the plan preview.

    The frontend needs this to say two things, and both are load-bearing:

      "3 topics you've got, 2 that need work — I built your path around
       those two."
      "You're solid on these 5 — they move to the end as a quick refresh."

    Without them the reshaping is invisible, and a plan that quietly reorders
    itself just looks like a plan. This is the only place the diagnostic's
    work becomes legible as work — and a student cannot perceive intelligence
    in a decision she cannot see.
    """
    out = {"gap": [], "shaky": [], "solid": [], "untested": []}
    for n in nodes:
        tier = n.get("_tier")
        if tier not in out:
            continue
        topic = _topic_of_node(n, unique_topics)
        if topic and topic not in out[tier]:
            out[tier].append(topic)
    return out


def _weight_path_by_diagnostic(nodes, diagnostic, unique_topics, days_to_exam=None):
    """Reshape a generated path using what the diagnostic learned.

    Pure and deterministic — no LLM call, no network, sub-millisecond. That is
    the point: this rule decides plan length, the number the whole activation
    thesis rests on, so it should be assertable in a test rather than
    re-negotiated by a model on every generation.

    `diagnostic` is {topic_name: percent}. A falsy diagnostic returns the nodes
    untouched, which is what a skipped diagnostic or an older client gets.
    """
    if not diagnostic or not nodes:
        return nodes

    archetype = _plan_archetype(days_to_exam)
    budget = PLAN_BUDGETS.get(archetype, PLAN_BUDGETS["master"])

    # Score every curriculum topic, matching loosely against diagnostic keys.
    diag_keys = list(diagnostic.keys())
    scores = {}
    for topic in unique_topics:
        key = _match_topic(topic, diag_keys)
        raw = diagnostic.get(key) if key else None
        try:
            scores[topic] = None if raw is None else float(raw)
        except (TypeError, ValueError):
            scores[topic] = None

    # Group generated nodes by topic, preserving order.
    grouped = {}
    order = []
    for n in nodes:
        t = _topic_of_node(n, unique_topics) or "General"
        if t not in grouped:
            grouped[t] = []
            order.append(t)
        grouped[t].append(n)

    units = []
    seq = 0
    for topic in order:
        tier = _tier_for_score(scores.get(topic))
        wanted = TIER_UNITS[tier]
        unit_nodes = _shape_unit(topic, grouped[topic], wanted, seq)
        seq += len(wanted)
        units.append({"topic": topic, "tier": tier, "nodes": unit_nodes})

    # Worst first, solid last. Ties break on score so the weakest gap opens.
    units.sort(key=lambda u: (
        TIER_ORDER[u["tier"]],
        scores.get(u["topic"]) if scores.get(u["topic"]) is not None else 999,
    ))

    units = _apply_budget(units, budget, archetype)
    units = _ensure_lesson_first(units)

    out = []
    for u in units:
        for n in u["nodes"]:
            n["_tier"] = u["tier"]
            out.append(n)
    return out


def _attach_status_and_exam_nodes(nodes: list, unique_topics: list) -> list:
    """Mirror /study/plan steps 5 + 5b: assign status, then insert exam nodes
    at topic boundaries. Returns the augmented list."""
    for i, node in enumerate(nodes):
        node["status"] = "available" if i == 0 else "locked"
        if "id" not in node:
            node["id"] = f"node_{i+1}"
        if "difficulty" not in node:
            node["difficulty"] = 1 + (i // 4)
        if "tags" not in node:
            node["tags"] = []

    nodes_with_exams = []
    current_topic = None
    exam_counter = 0
    for i, node in enumerate(nodes):
        node_topic = node.get("label", "")
        for t in unique_topics:
            if t.lower() in node_topic.lower():
                node_topic = t
                break

        next_topic = None
        if i + 1 < len(nodes):
            next_label = nodes[i + 1].get("label", "")
            for t in unique_topics:
                if t.lower() in next_label.lower():
                    next_topic = t
                    break

        nodes_with_exams.append(node)

        is_last = (i == len(nodes) - 1)
        topic_changes = (next_topic and node_topic and next_topic != node_topic)

        # A topic in the closing refresh tail gets no exam node. An exam costs
        # 10 question-units — more than the entire shaky unit it would follow —
        # and its job is to PROVE IMPROVEMENT on something that was weak. There
        # is nothing to prove on a topic the diagnostic says she already has and
        # which we deliberately chose not to teach. Untagged nodes (no
        # diagnostic, older clients) keep the old behaviour of an exam at every
        # boundary.
        is_tail = node.get("_tier") == "solid"

        if (is_last or topic_changes) and not is_tail:
            exam_counter += 1
            exam_label = node_topic if node_topic in unique_topics else (current_topic or "Review")
            nodes_with_exams.append({
                "id": f"exam_{exam_counter}",
                "type": "exam",
                "label": exam_label,
                "tags": ["exam", "mixed_types"],
                "difficulty": 2,
                "status": "locked"
            })

        current_topic = node_topic

    for i, node in enumerate(nodes_with_exams):
        node["status"] = "available" if i == 0 else "locked"

    return nodes_with_exams


@app.post("/study/start")
async def start_study_journey(request: StudyPlanRequest):
    """
    Combined plan + first-node SSE endpoint. See module-level WHY comment above.

    SSE event types emitted:
      - {"status": "session_ready"}                     — handshake
      - {"status": "plan_generating"}                   — LLM building the path
      - {"status": "plan_ready", "plan": {...}}         — full path nodes ready
      - {"status": "first_node_generating", ...}        — about to generate first node
      - {"status": "question_ready", "question": {...}} — quiz only, per-question
      - {"status": "flashcard_ready", "flashcard": {...}} — flashcard only, per-card
      - {"status": "first_node_ready", "node_id": "...", "type": "...", "content": {...}, "hash": "..."}
      - {"status": "first_node_skipped", "node_id": "...", "reason": "..."}  — for mindmap/exam
      - {"status": "complete"}                          — stream end
      - {"status": "error", "message": "..."}
    """
    print(f"\n{'='*60}")
    print(f"🚀 STUDY START (combined) - chat_id: {request.chat_id}")
    print(f"{'='*60}")

    async def stream_generator():
        try:
            # Plan-creation gate. This endpoint generates a full path AND the
            # first node, so it's the most expensive call in the product —
            # reject before any LLM work. Yielded in-stream as status:error so
            # the frontend's existing handler picks it up.
            plan_quota = usage_guard.check_plan_quota(request.chat_id)
            if not plan_quota["allowed"]:
                print(f"🚫 Plan quota exceeded for chat {request.chat_id} — rejecting /study/start")
                yield f"data: {json.dumps({'status': 'error', 'code': 'plan_quota_exceeded', 'message': usage_guard.PLAN_QUOTA_MESSAGE})}\n\n"
                return
            # ---------- STEP 1: session + topics ----------
            if request.chat_id not in ACTIVE_SESSIONS:
                ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
                await ACTIVE_SESSIONS[request.chat_id].load_file_insights_from_firebase()
                print(f"🆕 Created new session for /study/start")

            session_wrapper = ACTIVE_SESSIONS[request.chat_id]
            file_insights = getattr(session_wrapper.session, "file_insights", {})

            all_topics = []
            all_concepts = []
            for filename, insights in file_insights.items():
                if insights:
                    all_topics.extend(insights.get("topics", []))
                    all_concepts.extend(insights.get("concepts", []))

            unique_topics = list(set(all_topics))[:5] or ["Document Overview"]
            key_terms = list(set(all_concepts))[:10]

            print(f"📊 /study/start using insights topics: {unique_topics}")
            yield f"data: {json.dumps({'status': 'session_ready'})}\n\n"

            # ---------- STEP 2: plan generation (one LLM call, no redundant topic extraction) ----------
            yield f"data: {json.dumps({'status': 'plan_generating'})}\n\n"

            # ── Narrate the real decisions as they're made ────────────────
            # These aren't decorative loading messages: each one reports a fact
            # the planner has actually established. Plan generation is 3-8s of
            # otherwise-blank time, and it's exactly where the adaptation the
            # student is paying for happens — showing the work is what makes a
            # generated plan feel chosen rather than canned.
            yield f"data: {json.dumps({'status': 'plan_thinking', 'step': 'topics', 'topics': unique_topics})}\n\n"

            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
            prompt_language = _language_for_prompt(request.language)
            user_prefs = request.userPreferences or {}
            review_format = user_prefs.get("reviewFormat", "Visual Concept Maps")

            days_to_exam = _days_to_exam(user_prefs)
            archetype = _plan_archetype(days_to_exam)
            print(f"🗓️  Plan archetype: {archetype} "
                  f"(days_to_exam={days_to_exam}, hardest={user_prefs.get('hardestTopics')})")

            yield f"data: {json.dumps({'status': 'plan_thinking', 'step': 'archetype', 'archetype': archetype, 'days_to_exam': days_to_exam})}\n\n"

            hardest = [t for t in (user_prefs.get("hardestTopics") or []) if t]
            if hardest:
                yield f"data: {json.dumps({'status': 'plan_thinking', 'step': 'focus', 'hardest': hardest})}\n\n"

            yield f"data: {json.dumps({'status': 'plan_thinking', 'step': 'building'})}\n\n"
            path_prompt = _build_study_path_prompt(
                unique_topics,
                key_terms,
                review_format,
                prompt_language,
                days_to_exam=days_to_exam,
                hardest_topics=user_prefs.get("hardestTopics") or [],
            )
            response = await llm.ainvoke([{"role": "user", "content": path_prompt}])

            path_json = response.content.strip()
            if path_json.startswith("```"):
                path_json = path_json.split("```")[1]
                if path_json.startswith("json"):
                    path_json = path_json[4:]
                path_json = path_json.strip()

            try:
                nodes = json.loads(path_json)
            except json.JSONDecodeError as e:
                print(f"❌ Plan JSON parse failed: {e}")
                yield f"data: {json.dumps({'status': 'error', 'message': f'Plan JSON parse failed: {e}'})}\n\n"
                return

            nodes = _weight_path_by_diagnostic(
                nodes, request.diagnostic, unique_topics, days_to_exam
            )
            nodes = _attach_status_and_exam_nodes(nodes, unique_topics)
            print(f"✅ /study/start built path with {len(nodes)} nodes")

            plan_payload = {
                "nodes": nodes,
                "topics": unique_topics,
                "total_nodes": len(nodes),
                "estimated_time_minutes": len(nodes) * 3,
                # Lets the preview say WHY the plan looks the way it does.
                # Without this the reshaping is invisible, and an invisible
                # feature never teaches anyone to set an exam date.
                "archetype": _plan_archetype(days_to_exam),
                "days_to_exam": days_to_exam,
                "tiers": _summarize_tiers(nodes, unique_topics),
            }
            yield f"data: {json.dumps({'status': 'plan_ready', 'plan': plan_payload})}\n\n"

            # ---------- STEP 3: first-node content (overlaps with frontend's Firestore write) ----------
            if not nodes:
                yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                return

            first_node = nodes[0]
            node_id = first_node.get("id")
            node_type = first_node.get("type")
            node_label = first_node.get("label", "")

            yield f"data: {json.dumps({'status': 'first_node_generating', 'node_id': node_id, 'node_type': node_type, 'node_label': node_label})}\n\n"

            study_session = await _setup_study_session(request.chat_id, request.language)
            source = "documents" if study_session.documents and study_session.vectorstore else "scratch"

            try:
                if node_type == "lesson":
                    content = await _generate_lesson_with_context(study_session, node_label, request.language)
                    content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                    yield f"data: {json.dumps({'status': 'first_node_ready', 'node_id': node_id, 'type': 'lesson', 'content': content, 'hash': content_hash})}\n\n"

                elif node_type == "audio":
                    audio_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)
                    context = ""
                    if study_session.vectorstore:
                        docs = study_session.vectorstore.similarity_search(query=node_label, k=100)
                        context = "\n\n".join([d.page_content for d in docs])[:1000]
                    content = await _generate_audio_config(audio_llm, node_label, context, prompt_language)
                    content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                    yield f"data: {json.dumps({'status': 'first_node_ready', 'node_id': node_id, 'type': 'audio', 'content': content, 'hash': content_hash})}\n\n"

                elif node_type == "quiz":
                    # This endpoint only ever generates the plan's FIRST node,
                    # which auto-launches — so a quiz here IS the diagnostic and
                    # must match the 3-question calibration the UI promises.
                    # (/study/generate-item-stream gets this via
                    # StudyItemRequest.is_diagnostic; this path has no request
                    # flag because the node is implicit.)
                    questions = []
                    async for chunk in stream_quiz_with_bank(
                        topic=node_label,
                        difficulty="medium",
                        num_questions=STUDY_DIAGNOSTIC_QUESTIONS,
                        source=source,
                        session=study_session,
                        chat_id=study_session.chat_id,
                        question_types=STUDY_QUIZ_TYPES,
                        quiz_mode="knowledge"
                    ):
                        if chunk.get("status") == "question_ready":
                            q = chunk.get("question") or {}
                            formatted = _format_study_question(q, node_label)
                            questions.append(formatted)
                            yield f"data: {json.dumps({'status': 'question_ready', 'question': formatted, 'node_id': node_id})}\n\n"
                    content = {"questions": questions}
                    content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                    yield f"data: {json.dumps({'status': 'first_node_ready', 'node_id': node_id, 'type': 'quiz', 'content': content, 'hash': content_hash})}\n\n"

                elif node_type == "flashcard":
                    cards = []
                    async for chunk in stream_flashcards(
                        topic=node_label,
                        num_cards=STUDY_FLASHCARD_CARDS,
                        source=source,
                        session=study_session,
                        chat_id=study_session.chat_id
                    ):
                        if chunk.get("status") == "flashcard_ready":
                            fc = chunk.get("flashcard") or {}
                            formatted = {
                                "front": fc.get("front", ""),
                                "back": fc.get("back", ""),
                                "topic": fc.get("topic", node_label)
                            }
                            cards.append(formatted)
                            yield f"data: {json.dumps({'status': 'flashcard_ready', 'flashcard': formatted, 'node_id': node_id})}\n\n"
                    content = {"cards": cards}
                    content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                    yield f"data: {json.dumps({'status': 'first_node_ready', 'node_id': node_id, 'type': 'flashcard', 'content': content, 'hash': content_hash})}\n\n"

                elif node_type == "mindmap":
                    # Mindmap nodes are stubbed — actual generation happens in /study/generate-mindmap
                    content = {"topic": node_label, "depth": "medium", "mindmapData": None}
                    content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                    yield f"data: {json.dumps({'status': 'first_node_ready', 'node_id': node_id, 'type': 'mindmap', 'content': content, 'hash': content_hash})}\n\n"

                else:
                    # Exam or unknown — let the frontend handle via its own modal/flow.
                    yield f"data: {json.dumps({'status': 'first_node_skipped', 'node_id': node_id, 'reason': f'node type {node_type!r} not pre-generated'})}\n\n"
            except Exception as node_err:
                print(f"⚠️ First-node generation failed (plan still delivered): {node_err}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole stream — the user got the plan; the frontend will retry the node.
                yield f"data: {json.dumps({'status': 'first_node_skipped', 'node_id': node_id, 'reason': str(node_err)})}\n\n"

            yield f"data: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            print(f"❌ /study/start failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/study/diagnostic-quiz")
async def generate_diagnostic_quiz(request: DiagnosticQuizRequest):
    """
    Generate 5 breadth-first diagnostic questions spanning all major topics.
    Called BEFORE showing the study plan to establish a baseline proficiency score.

    Uses a single LLM call to produce one question per major topic, varying
    difficulty from easy to hard. The frontend stores results in studyPerformance
    before createStudySession is called, seeding the adaptive engine early.
    """
    print(f"\n{'='*60}")
    print(f"🔬 DIAGNOSTIC QUIZ GENERATION - chat_id: {request.chat_id}")
    print(f"{'='*60}")

    try:
        # Use the same ACTIVE_SESSIONS pattern as /study/plan so file_insights are available
        if request.chat_id not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
            await ACTIVE_SESSIONS[request.chat_id].load_file_insights_from_firebase()
            print(f"🆕 Created new session for diagnostic quiz")

        session = ACTIVE_SESSIONS[request.chat_id]
        file_insights = getattr(session.session, "file_insights", {})

        # Build topic + concept context from file_insights
        all_topics = []
        all_concepts = []
        for filename, insights in file_insights.items():
            if insights:
                all_topics.extend(insights.get("topics", []))
                all_concepts.extend(insights.get("concepts", []))
        unique_topics = list(set(all_topics))[:10]
        unique_concepts = list(set(all_concepts))[:20]

        # Get document content from vectorstore (same as /study/plan, no hard fail)
        document_content = ""
        if session.session.vectorstore:
            docs = session.session.vectorstore.similarity_search("main topics concepts definitions", k=20)
            document_content = "\n\n".join([doc.page_content for doc in docs])[:10000]

        # Build context from whichever source is available
        context_str = document_content[:8000] if document_content else ""
        if not context_str and (unique_topics or unique_concepts):
            context_str = f"Topics: {', '.join(unique_topics)}\nConcepts: {', '.join(unique_concepts)}"

        if not context_str:
            raise HTTPException(status_code=400, detail="No document content found for this session.")

        # Topics she TOLD us were hardest, in onboarding Q2. They are asked
        # first, because the most valuable thing this can produce is a
        # contradiction: "you said pharmacology, but you're solid there — it's
        # fluid balance." A self-report confirmed is worth little; a
        # self-report corrected is the moment the product stops feeling like a
        # quiz generator.
        hardest = [t for t in (request.hardestTopics or []) if t][:3]
        ordered_topics = hardest + [t for t in unique_topics if t not in hardest]
        focus_topics = ordered_topics[:DIAGNOSTIC_TOPIC_LIMIT] or unique_topics[:3]

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

        focus_block = "\n".join("- %s" % t for t in focus_topics) or "- (infer from the document)"

        prompt = f"""You are calibrating a study plan. Generate exactly {DIAGNOSTIC_QUESTION_COUNT} multiple-choice questions that reveal what this student already knows.

This is NOT a test. It is never scored or shown as a grade. Its only job is to
decide what she should spend her time on, so a question that is impossible to
get right teaches us nothing and just makes her feel behind.

PRIORITY TOPICS (ask about these first):
{focus_block}

RULES:
- Cover the priority topics above. Give the FIRST {DIAGNOSTIC_DEEP_TOPICS} topics TWO questions each
  (one easier, one harder); give any remaining topic ONE question.
- Two questions on a topic is the minimum evidence for deciding she is solid on
  it — one question is a coin flip, and being wrongly told she is strong is the
  one mistake here that costs her the exam.
- Each question must have exactly 4 options
- Keep difficulty fair: test understanding, not recall of a footnote
- Use exact terminology from the document — do NOT invent topics
- Keep questions concise (1-2 sentences max)
- "concept" must be a SHORT human label for what the question tests
  (e.g. "preload vs afterload"), 2-6 words. It is shown to the student later
  in sentences like "you were confusing X", so write it as a thing, not a
  sentence.

DOCUMENT CONTENT:
{context_str}

Return ONLY a valid JSON array (no markdown, no explanation):
[
  {{
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correctIndex": 0,
    "rationale": "One sentence on why the correct answer is right",
    "topic": "Topic name from the document",
    "concept": "short concept label"
  }}
]
(exactly {DIAGNOSTIC_QUESTION_COUNT} objects, no more, no less)"""

        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("["):
                    content = stripped
                    break

        questions = json.loads(content)
        # Safety: clamp and validate structure
        questions = [
            q for q in questions[:DIAGNOSTIC_QUESTION_COUNT]
            if isinstance(q.get("options"), list) and len(q["options"]) == 4
        ]

        # Topics as WE grouped them, so the knowledge map renders the same
        # names the plan was built from. Letting the frontend re-derive topic
        # names with its own fuzzy matcher would give two sources of truth that
        # drift, and the student would see one topic appear twice under
        # near-identical spellings.
        asked_topics = []
        for q in questions:
            t = (q.get("topic") or "").strip()
            if t and t not in asked_topics:
                asked_topics.append(t)

        print(f"✅ Generated {len(questions)} diagnostic questions over {len(asked_topics)} topics")
        return {
            "questions": questions,
            "topics": asked_topics,
            "focusTopics": focus_topics,
        }

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error in diagnostic quiz: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse diagnostic questions from AI response.")
    except Exception as e:
        print(f"❌ Diagnostic quiz generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




# ══════════════════════════════════════════════════════════════════════════
# NARRATION VOICE PASS
#
# The knowledge map's sentences are hand-written templates. That is what
# makes them safe — they cannot hallucinate a claim about a student's
# performance, and their behaviour is unit-tested. It is also what makes
# them feel templated on the second and third plan, which is the slow
# version of "this feels robotic".
#
# So: the LOGIC still decides what is true and which claims get made. This
# endpoint only rewrites the WORDING. If it fails, is slow, or returns
# anything that fails validation, the caller keeps the templates and the
# student never knows there was a second path.
# ══════════════════════════════════════════════════════════════════════════

# Register per countdown phase. The map and the dashboard already agree on
# these phases; this keeps the VOICE agreeing with them too.
NARRATION_REGISTER = {
    "steady": "relaxed and encouraging — there is genuinely plenty of time",
    "focus": "focused and businesslike, still warm",
    "final": "calm and direct, no false comfort, no panic",
    "examDay": "steady and reassuring; nothing that sounds like there is time to learn more",
    "past": "matter-of-fact and kind",
    "undated": "easy-going",
}

MAX_NARRATION_LINES = 12
MAX_NARRATION_CHARS = 260


def _digits_in(text: str) -> set:
    """Digit runs in a string. Used to prove the rewrite invented no numbers."""
    return set(re.findall(r"\d+", text or ""))


def validate_narration(originals: list, rewritten, protected_terms=None) -> bool:
    """Reject a rewrite that changed anything that matters.

    All-or-nothing on purpose. Mixing rewritten and template lines produces a
    message that changes voice halfway through, which reads worse than either
    version on its own.
    """
    if not isinstance(rewritten, list) or len(rewritten) != len(originals):
        return False

    terms = [t for t in (protected_terms or []) if t]

    for original, new in zip(originals, rewritten):
        if not isinstance(new, str):
            return False
        new = new.strip()
        if not new or len(new) > MAX_NARRATION_CHARS:
            return False

        # No invented numbers. This is the one that stops "you're 80% ready"
        # appearing under a screen that deliberately shows no percentages.
        if not _digits_in(new).issubset(_digits_in(original)):
            return False

        # A topic named in the original must still be named. Otherwise the
        # model can quietly generalise "Fluid Balance" into "your weak area"
        # and the map stops being about her.
        for term in terms:
            if term.lower() in original.lower() and term.lower() not in new.lower():
                return False

    return True


@app.post("/study/narrate")
async def narrate_study_map(request: NarrationRequest):
    """Voice pass over the knowledge-map narration. Never quota-gated: it is
    a few hundred nano-model tokens, and it runs before the student has been
    shown anything she could be charged for."""
    lines = [l for l in (request.lines or []) if isinstance(l, str) and l.strip()]
    if not lines or len(lines) > MAX_NARRATION_LINES:
        return {"lines": None, "reason": "nothing_to_do"}

    try:
        language = _language_for_prompt(request.language)
        register = NARRATION_REGISTER.get(request.phase or "", "warm and natural")

        numbered = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(lines))

        prompt = f"""Rewrite each line so it sounds like a real tutor talking, not a template.

ABSOLUTE RULES:
- Return EXACTLY {len(lines)} lines, in the same order, one rewrite per line.
- Keep the MEANING and every FACT identical. Add nothing: no new advice, no
  praise, no numbers, no claims that are not already in the line.
- NEVER introduce a number that is not already in that line.
- Keep every proper noun exactly as written (topic names, exam names).
- Keep each line roughly the same length. Short. Spoken, not written.
- Contractions are good. Sentence fragments are fine.
- No greetings, no emoji, no sign-offs, no bullet points.
- Write in {language}.

TONE: {register}

LINES:
{numbered}

Return ONLY a JSON array of {len(lines)} strings."""

        # Cheapest model in the stack. This is a paraphrase of text that is
        # already correct, which is about the easiest job an LLM can be given.
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.85)
        response = await llm.ainvoke(prompt)
        content = (response.content or "").strip()

        if "```" in content:
            for part in content.split("```"):
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("["):
                    content = stripped
                    break

        rewritten = json.loads(content)
        rewritten = [str(x).strip() for x in rewritten] if isinstance(rewritten, list) else None

        if not validate_narration(lines, rewritten, request.protected_terms):
            print("🗣️ Narration rejected by validation — falling back to templates")
            return {"lines": None, "reason": "failed_validation"}

        print(f"🗣️ Narration voiced ({len(rewritten)} lines)")
        return {"lines": rewritten}

    except Exception as e:
        # Never fatal. The templates are a complete, shipped experience.
        print(f"⚠️ Narration voice pass failed ({e}) — falling back to templates")
        return {"lines": None, "reason": "error"}

@app.post("/study/plan-review")
async def generate_review_plan(request: StudyReviewPlanRequest):
    """
    Generate a Phase 2 review study path based on Phase 1 performance.
    Uses the same session & vectorstore as the original study plan.

    Performance-based node generation:
    - WEAK topics (<60%): LESSON → FLASHCARD → QUIZ (full re-teach)
    - DEVELOPING topics (60-84%): FLASHCARD → QUIZ (reinforce)
    - STRONG topics (85%+): Single QUIZ (confidence check)
    """
    print(f"\n{'='*60}")
    print(f"📚 REVIEW PATH GENERATION - chat_id: {request.chat_id}")
    print(f"{'='*60}")

    try:
        # ------------------------------------------
        # STEP 1: Get or create session (same as /study/plan)
        # ------------------------------------------
        if request.chat_id not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[request.chat_id] = NursingTutor(request.chat_id)
            await ACTIVE_SESSIONS[request.chat_id].load_file_insights_from_firebase()
            print(f"🆕 Created new session for review plan")

        session = ACTIVE_SESSIONS[request.chat_id]

        # ------------------------------------------
        # STEP 2: Parse performance data into categories
        # ------------------------------------------
        topics_data = request.performance.get("topics", {})
        if not topics_data:
            print("⚠️ No performance data, returning empty review plan")
            return {"nodes": [], "topics": [], "total_nodes": 0, "estimated_time_minutes": 0}

        weak_topics = []
        developing_topics = []
        strong_topics = []

        for topic_name, topic_info in topics_data.items():
            missed = topic_info.get("missedConcepts", [])
            quiz_total = max(topic_info.get("questionsTotal", 1), 1)
            flash_total = max(topic_info.get("flashcardsTotal", 1), 1)
            quiz_acc = (topic_info.get("questionsCorrect", 0) / quiz_total) * 100
            flash_acc = (topic_info.get("flashcardsMastered", 0) / flash_total) * 100
            strength = topic_info.get("strengthLevel", "developing")

            entry = {
                "name": topic_name,
                "quiz_accuracy": round(quiz_acc),
                "flashcard_accuracy": round(flash_acc),
                "missed_concepts": missed[:5],
                "strength": strength
            }

            if strength == "weak":
                weak_topics.append(entry)
            elif strength == "developing":
                developing_topics.append(entry)
            else:
                strong_topics.append(entry)

        print(f"📊 Performance breakdown: {len(weak_topics)} weak, {len(developing_topics)} developing, {len(strong_topics)} strong")

        # ------------------------------------------
        # STEP 3: Build performance summary for prompt
        # ------------------------------------------
        perf_lines = []
        for t in weak_topics:
            missed_str = ", ".join(t["missed_concepts"][:3]) if t["missed_concepts"] else "N/A"
            perf_lines.append(f'- Topic "{t["name"]}": WEAK ({t["quiz_accuracy"]}% quiz, {t["flashcard_accuracy"]}% flashcard). Missed: {missed_str}')
        for t in developing_topics:
            missed_str = ", ".join(t["missed_concepts"][:3]) if t["missed_concepts"] else "N/A"
            perf_lines.append(f'- Topic "{t["name"]}": DEVELOPING ({t["quiz_accuracy"]}% quiz, {t["flashcard_accuracy"]}% flashcard). Missed: {missed_str}')
        for t in strong_topics:
            perf_lines.append(f'- Topic "{t["name"]}": STRONG ({t["quiz_accuracy"]}% quiz, {t["flashcard_accuracy"]}% flashcard).')

        performance_summary = "\n".join(perf_lines)

        # ------------------------------------------
        # STEP 4: Generate review path with LLM
        # ------------------------------------------
        prompt_language = _language_for_prompt(request.language)
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

        review_prompt = f"""Create a REVIEW study path for a student who just completed their first study round.

PERFORMANCE DATA:
{performance_summary}

STRUCTURE RULES:
1. WEAK topics (< 60% accuracy): LESSON -> FLASHCARD -> QUIZ (full re-teach, focus on missed concepts)
2. DEVELOPING topics (60-84%): FLASHCARD -> QUIZ only (reinforce weak spots)
3. STRONG topics (85%+): Single QUIZ node (quick confidence check)
4. Order: weakest topics first, strongest last
5. Total: 4-12 nodes depending on how many topics need review
6. All node labels should indicate this is a review (e.g., "Topic Name - Review")

Return ONLY valid JSON array in {prompt_language}:
[
  {{"id": "review_1", "type": "lesson", "label": "Topic Name - Review", "tags": ["review", "weak"], "difficulty": 1}},
  ...
]"""

        response = await llm.ainvoke([{"role": "user", "content": review_prompt}])

        # ------------------------------------------
        # STEP 5: Parse & validate (same as /study/plan)
        # ------------------------------------------
        path_json = response.content.strip()

        # Clean markdown code blocks if present
        if path_json.startswith("```"):
            path_json = path_json.split("```")[1]
            if path_json.startswith("json"):
                path_json = path_json[4:]
            path_json = path_json.strip()

        nodes = json.loads(path_json)

        for i, node in enumerate(nodes):
            node["status"] = "available" if i == 0 else "locked"
            if "id" not in node:
                node["id"] = f"review_{i+1}"
            if "difficulty" not in node:
                node["difficulty"] = 1
            if "tags" not in node:
                node["tags"] = ["review"]
            elif "review" not in node["tags"]:
                node["tags"].append("review")

        review_topics = [t["name"] for t in (weak_topics + developing_topics + strong_topics)]

        print(f"✅ Generated review path with {len(nodes)} nodes")
        for node in nodes:
            print(f"   - {node['type']}: {node['label']}")

        # ------------------------------------------
        # STEP 6: Return (same shape as /study/plan)
        # ------------------------------------------
        return {
            "nodes": nodes,
            "topics": review_topics,
            "total_nodes": len(nodes),
            "estimated_time_minutes": len(nodes) * 3
        }

    except Exception as e:
        print(f"❌ Review path generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exam-debrief/turn")
async def exam_debrief_turn(request: ExamDebriefTurnRequest):
    """
    One turn of the conversation we have with a student after her exam.

    Thin on purpose: everything that decides what to say lives in
    services/exam_debrief.py. This endpoint exists to keep the ANTHROPIC_API_KEY
    server-side and to give the frontend one shape to call.
    """
    student_turns = sum(1 for m in request.messages if m.role == "user")
    print(f"\n🎓 EXAM DEBRIEF TURN — exam: {request.exam_name or '(unnamed)'} "
          f"| student messages: {student_turns} | lang: {request.language}")

    try:
        result = await run_exam_debrief_turn(
            messages=[m.dict() for m in request.messages],
            exam_name=request.exam_name,
            exam_date=request.exam_date,
            days_after=request.days_after,
            study_context=request.study_context,
            language=_language_for_prompt(request.language),
        )
        print(f"   ↳ done={result['done']} | prepared={result['insights']['preparedness']} "
              f"| tags={','.join(result['insights']['gap_tags']) or '-'}")
        return result

    except Exception as e:
        # The frontend closes the conversation warmly on a failure rather than
        # leaving her typing into something that never answers, and keeps
        # whatever she already said. Nothing here is worth a retry loop.
        print(f"❌ Exam debrief turn failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/study/interpret-request")
async def interpret_study_request(request: StudyInterpretRequest):
    """
    Interpret a student's free-text request and return:
    1. An echo message confirming what the system understood
    2. A node definition that can be inserted into the path

    Bounded scope: difficulty, format, focus, or skip.
    Rejects clinical advice or out-of-scope requests with a warm redirect.
    """
    print(f"\n{'='*60}")
    print(f"💬 INTERPRET STUDENT REQUEST - chat_id: {request.chat_id}")
    print(f"   Text: {request.user_text}")
    print(f"   Context: {request.current_node_type} on {request.current_topic}")
    print(f"{'='*60}")

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        prompt_language = _language_for_prompt(request.language)

        # Build performance context if available
        perf_context = ""
        if request.score_percent is not None:
            perf_context += f"\nHer score: {request.score_percent}%"
        if request.missed_items:
            missed_list = "\n".join(f"  - {item}" for item in request.missed_items[:5])
            perf_context += f"\nSpecific questions/concepts she got wrong:\n{missed_list}"

        interpret_prompt = f"""You are an AI study coach for a nursing student. She just completed a {request.current_node_type} on "{request.current_topic}" and typed this request:

"{request.user_text}"
{perf_context}

Your job: interpret her request and return THE BEST study node to help her.

IMPORTANT INTERPRETATION RULES:
- "explain what I missed" / "review my mistakes" / "what did I get wrong" → Create a LESSON that explains the specific concepts she missed (listed above). NOT a quiz.
- "make it harder" → Create a harder quiz or flashcard on the same topic
- "flashcards" / "just flashcards" → Create flashcards on the topic
- "quiz me" → Create a quiz
- "go deeper" / "explain more" → Create a detailed lesson
- "skip ahead" → She wants to move to the next topic entirely
- If she asks about something specific ("focus on side effects"), create content on that subtopic

MATCH FORMAT TO INTENT:
- Explaining, reviewing, teaching → "lesson"
- Drilling, practicing, testing → "quiz" or "flashcard"
- Listening → "audio"
- Visualizing connections → "mindmap"

OUT OF SCOPE (warm redirect):
- Clinical advice, treatment questions, patient scenarios
- Anything not about studying for her exam

NODE TYPES you can return:
- "lesson": A reading lesson explaining concepts (USE THIS when she wants explanations of what she missed)
- "flashcard": Flashcard set (12 cards)
- "quiz": Quiz questions (12 questions)
- "audio": Audio lesson
- "mindmap": Visual concept map

Return ONLY valid JSON in {prompt_language}:
{{
    "understood": true,
    "echo": "A short, warm confirmation written TO the student in {prompt_language}. Reference what she specifically missed if relevant. e.g. 'I'll break down the tetanus immunization and LMNOP concepts you missed — let's clear those up.'",
    "node": {{
        "type": "lesson",
        "label": "Specific topic label in {prompt_language}",
        "tags": ["relevant", "tags"],
        "difficulty": 1
    }}
}}

If the request is out of scope, return:
{{
    "understood": false,
    "echo": "A warm redirect in {prompt_language}. e.g. 'I can help with study materials! Try asking for a quiz, flashcards, or a lesson on a specific topic.'",
    "node": null
}}"""

        response = await llm.ainvoke([{"role": "user", "content": interpret_prompt}])

        result_json = response.content.strip()
        if result_json.startswith("```"):
            result_json = result_json.split("```")[1]
            if result_json.startswith("json"):
                result_json = result_json[4:]
            result_json = result_json.strip()

        result = json.loads(result_json)

        print(f"✅ Interpreted request: understood={result.get('understood')}")
        print(f"   Echo: {result.get('echo')}")
        if result.get('node'):
            print(f"   Node: {result['node'].get('type')} - {result['node'].get('label')}")

        return result

    except Exception as e:
        print(f"❌ Interpret request failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════
# STUDY NOTE — what the tutor wrote down about this quiz.
#
# The format-pattern debrief below is strict on purpose, and it says nothing
# on most nodes: a standard quiz is MCQ-only, and you cannot find a pattern
# ACROSS formats in a set that only has one. So the common case fell through
# to a placeholder — "I'm learning how you think, 2 more of these and I
# should have something specific" — which is a promise to observe her rather
# than an observation, shown after she has just answered five questions we
# could have read.
#
# This reads them. The same discipline applies as everywhere else here:
#
#   PYTHON decides which honest thing there is to say, from the actual
#   right/wrong pattern. The MODEL only writes the sentence.
#
# Three cases, because they are genuinely different and a model left to pick
# would blur them:
#   · 2+ misses  — look for what the missed questions have in common
#   · 1 miss     — name that one plainly; a single miss is not a theme
#   · 0 misses   — say what she demonstrated, not "well done"
# ══════════════════════════════════════════════════════════════════════════

STUDY_NOTE_MAX_CHARS = 260      # reflection notes (lesson / audio / map)
STUDY_TAKEAWAY_MAX_CHARS = 130  # the one-line takeaway on a scored node


def _trim_note(note, max_sentences=2):
    """
    Hold a note to its sentence budget.

    The prompt states the limit and the model mostly complies, but "mostly" is
    not a limit — and the extra sentence is reliably the one that restates the
    first, on a card the student reads in about four seconds. Trimmed rather
    than rejected: a good sentence plus a redundant one is still worth showing
    once the redundant one is gone.
    """
    parts = re.findall(r"[^.!?]+[.!?]+", note or "")
    if len(parts) <= max_sentences:
        return (note or "").strip()
    return "".join(parts[:max_sentences]).strip()


async def _build_study_note(request, language_label):
    """One or two sentences about the node just finished, or None."""
    items = [i for i in (request.items or []) if getattr(i, "question", "")]
    if not items:
        return None

    missed = [i for i in items if not i.correct]
    got = [i for i in items if i.correct]

    # Flashcards are recall, not reasoning: there is no distractor to fall for
    # and no format to pattern-match, so a note that talks about "questions"
    # and "how you answered" describes work she did not do. Naming the unit
    # correctly is the whole difference between a note that sounds like it
    # watched her and one that was clearly written for something else.
    is_cards = request.node_type == "flashcard"
    unit = "cards" if is_cards else "questions"
    one_unit = "card" if is_cards else "question"
    verb = "could not recall" if is_cards else "missed"

    def brief(entries, limit=4):
        return chr(10).join(
            "- %s%s" % (
                (e.question or "")[:180],
                (" [why: %s]" % (e.rationale or "")[:120]) if e.rationale else "",
            )
            for e in entries[:limit]
        )

    # A clean sweep has to come first, because "what do the misses have in
    # common" is only answerable when something was NOT missed. Run through
    # the theme branch, a 0-of-5 produced the topic title back in longer
    # words — the emptiest thing on a screen whose whole claim is that it
    # watched her. At zero recall the valuable sentence is not a diagnosis,
    # it is the first fact she can carry into the re-teach.
    if missed and not got:
        mode = "blank"
        task = (
            "She missed EVERY one. There is no contrast to draw and no theme to "
            "find, so do not look for one — and do not tell her what the topic "
            "was, she has just spent ten minutes on it. Give her ONE thing to "
            "hold onto instead: the single most useful fact from the answers "
            "below, stated plainly enough that she could repeat it back. This is "
            "the last thing she reads before being taught this again, so make it "
            "the foothold."
        )
        body = "SHE MISSED ALL OF THESE. THE CORRECT ANSWERS:" + chr(10) + brief(missed, 5)
    elif len(missed) >= 2:
        mode = "theme"
        task = (
            f"She {verb} these. Say what TRIPPED HER UP — what the ones she got "
            "wrong have in common, the kind of thinking they share, not the topic "
            "name. Make it unmistakable that you are describing the misses: a "
            "sentence that could be read as praise for a thing she failed at is "
            "worse than saying nothing. If they genuinely have nothing in common, "
            "say that instead; do not invent a theme."
        )
        body = "MISSED:" + chr(10) + brief(missed) + chr(10) + chr(10) + "GOT RIGHT:" + chr(10) + brief(got, 3)
    elif len(missed) == 1:
        mode = "single"
        task = (
            f"She {verb} exactly one. Name what that single {one_unit} turned on, in "
            "plain words. Do NOT describe it as a pattern or a weakness - it is one "
            f"{one_unit}."
        )
        body = "MISSED:" + chr(10) + brief(missed) + chr(10) + chr(10) + "GOT RIGHT:" + chr(10) + brief(got, 3)
    else:
        mode = "clean"
        task = (
            "She got everything right. Name in ONE short sentence the kind of "
            f"{'recall' if is_cards else 'reasoning'} this took — the skill, not the "
            f"content. Do NOT list or summarise what the {unit} were about: she just "
            "answered them and reading them back to her is the least interesting "
            "thing you could do. It should land as an observation about her, not as "
            "congratulation. No exclamation marks."
        )
        body = "ALL CORRECT:" + chr(10) + brief(got)

    prompt = f"""You are a tutor naming the ONE thing a student should take away from
the {unit} she has just finished.

{task}

THIS IS A TAKEAWAY, NOT A SENTENCE ABOUT HER. It is displayed on its own as a
single line she can carry away, so name the THING — never narrate what happened.

  BAD:  "You missed the question about why oxygen at 15 L/min and bag-mask
         ventilation are used in respiratory distress; it turned on what the
         high flow actually does."
  GOOD: "Why high-flow oxygen and bag-mask ventilation are used."

  BAD:  "You couldn't recall what the first step in airway management is."
  GOOD: "Jaw-thrust first, not head-tilt — the neck may be injured."

RULES:
- ONE line. Under 14 words. No second sentence.
- No preamble. Never open with "You missed", "You couldn't recall", "You
  struggled with", "The question about" or "This card was about".
- Plain, everyday words. Say "hard to tell apart" and not "difficult to
  differentiate", "what to do first" and not "prioritisation of interventions".
  Keep the clinical terms she is actually studying; simplify everything around
  them. If it needs reading twice, rewrite it.
- Concrete and specific to the {unit} below. No generic study advice.
- NEVER just restate the topic in longer words. She has just spent ten minutes
  on it, so a line she could have written before starting is worth nothing.
- No score, no percentages, no numbers you were not given.
- No praise words ("great", "excellent", "well done"). Observe, do not cheer.
- Write in {language_label}.

TOPIC: {request.topic}

{body}

Return ONLY the note text."""

    try:
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
        response = await llm.ainvoke(prompt)
        note = _trim_note((response.content or "").strip().strip('"'), max_sentences=1)

        # Cheap guards. The takeaway asserts something about her work, so a
        # runaway or empty response is dropped rather than shown. The cap is
        # tight because this renders as ONE line under a "Focus on" label —
        # a paragraph there is the thing the whole screen was redesigned away
        # from, and truncating mid-thought would be worse than showing nothing.
        if not note or len(note) > STUDY_TAKEAWAY_MAX_CHARS:
            return None
        if re.search(r"\d+\s*%", note):     # never smuggle a score back in
            return None
        # A preamble survived the instruction; the line is about the event
        # rather than the content, which is exactly what it must not be.
        if re.match(r"^\s*(you|she)\b", note, re.I):
            return None

        print(f"   study note ({mode}): {note[:70]}")
        return {"text": note, "mode": mode}
    except Exception as e:
        print(f"   study note failed ({e})")
        return None


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION NOTE — the tutor's note on a node with no right or wrong.
#
# Lessons, audio and concept maps are most of a plan and produced nothing:
# a green tick, "You finished the lesson on X", "Up next: quiz on Y". True,
# and written before she arrived. A student paying for insight into her own
# performance got insight on roughly half her nodes and inventory on the rest.
#
# There is no score to diagnose here, so the temptation is to praise — and
# praise for reading a page is exactly the flattery this codebase refuses
# everywhere else. What actually makes a note on a lesson worth reading is
# CONNECTION: what she just studied, set against what her record says she
# keeps getting wrong.
#
#     "This one wasn't filler for you. Afterload has cost you three
#      questions in this plan — it's the thing you keep half-remembering."
#
# That is a claim about HER, and every part of it is computed here:
#
#   PYTHON decides whether the node genuinely touched a live struggle, by
#   token overlap against the concept ledger. The MODEL only writes the
#   sentence, and is told explicitly when it may NOT claim a link.
#
# Four modes, because they are different situations and a model left to
# choose would flatten them into the same warm paragraph:
#   · addressed  — the node covered something she is currently missing
#   · groundwork — she has a record, but this node did not touch it
#   · fresh      — no record yet; say what this sets up, claim nothing
#   · skipped    — she skipped it. Do not narrate studying that didn't happen.
# ══════════════════════════════════════════════════════════════════════════

# Words that overlap between any two nursing topics and so prove nothing.
_LINK_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "you",
    "nursing", "patient", "patients", "care", "management", "assessment",
    "review", "intro", "introduction", "overview", "basics", "concepts",
    "understanding", "clinical", "practice", "study", "part", "key",
}


def _link_tokens(text):
    """Significant lowercase tokens of a label, for overlap matching."""
    return {
        w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
        if w not in _LINK_STOPWORDS
    }


def _matched_struggles(topic, covered, struggles, limit=2):
    """
    Which of her live struggles this node actually touched.

    Deliberately conservative: a shared significant token, nothing cleverer.
    A false positive here produces the single worst sentence this product can
    say — "this covered the thing you keep missing" about a lesson that did
    not — so the bar is a word she can see in both places, and the fallback
    when nothing matches is to claim no link at all.
    """
    haystack = _link_tokens(topic)
    for c in covered or []:
        haystack |= _link_tokens(c)
    if not haystack:
        return []

    hits = []
    for label in struggles or []:
        if _link_tokens(label) & haystack:
            hits.append(label)
    return hits[:limit]


async def _build_reflection_note(request, language_label):
    """
    One or two sentences about an unscored node, plus the evidence behind it.

    Returns {"text", "mode", "linked"} or None. `linked` is the struggle
    labels that were actually matched — the UI shows those itself, so the
    number of times something has cost her is never phrased by the model.
    """
    topic = request.topic or ""
    covered = [c for c in (request.covered or []) if c][:6]
    linked = _matched_struggles(topic, covered, request.struggles)
    # Something she used to miss and has since answered right twice running.
    # Only consulted when nothing is still live, so a note never celebrates a
    # fix while quietly sitting on a gap that is still open.
    fixed = [] if linked else _matched_struggles(topic, covered, request.resolved)

    if request.skipped:
        mode = "skipped"
        task = (
            "She SKIPPED this one — she did not study it. Do not describe it as "
            "finished, do not praise her, and do not scold her. Acknowledge the "
            "skip in a single neutral clause and say plainly what it means for "
            "what comes next. Skipping is allowed here."
        )
    elif linked:
        mode = "addressed"
        task = (
            "This is the important case. What she just studied covers something "
            "she has been GETTING WRONG, listed below under STILL MISSING. Tell "
            "her that connection directly — that this one was not filler for "
            "her specifically, and name the thing. Do not give a count or any "
            "number; the evidence is shown separately underneath your note."
        )
    elif fixed:
        mode = "reinforced"
        task = (
            "What she just studied covers something she USED TO get wrong and "
            "has since answered correctly, listed below under ALREADY TURNED "
            "AROUND. Say that — she fixed this, and this was another pass over "
            "it. State it as something she did, not as a compliment, and give "
            "no count; the evidence is shown separately underneath your note."
        )
    elif request.struggles:
        mode = "groundwork"
        task = (
            "She has a record of things she is getting wrong, but this node did "
            "NOT cover any of them. You therefore may NOT claim it addressed a "
            "weakness — saying so would be false. Instead say what this one "
            "builds toward, concretely, based on what it covered."
        )
    else:
        mode = "fresh"
        task = (
            "This is early — there is no record of her missing anything yet, so "
            "you know nothing about her performance and must not imply that you "
            "do. Say what she now has in hand from this, and what it sets up. "
            "Forward-looking, not evaluative."
        )

    NL = chr(10)
    covered_block = NL.join("- %s" % c[:120] for c in covered) if covered else "  (not itemised)"
    # The heading has to match the mode: handing the model a list headed
    # "STILL MISSING" when the point is that she fixed those would produce a
    # note that contradicts the evidence chips rendered right beneath it.
    if fixed:
        evidence_labels = fixed
        evidence_heading = "ALREADY TURNED AROUND (she used to miss these, now she doesn't)"
    else:
        evidence_labels = (linked or request.struggles or [])[:4]
        evidence_heading = "STILL MISSING (things she has answered wrong in this study plan)"
    struggle_block = NL.join("- %s" % s[:80] for s in evidence_labels) or "  (nothing yet)"
    next_block = (
        f"{request.next_type or 'step'} on \"{request.next_label}\""
        if request.next_label else "(end of her plan for now)"
    )

    node_word = {
        "lesson": "lesson", "audio": "audio explanation", "mindmap": "concept map",
    }.get(request.node_type, request.node_type or "step")

    prompt = f"""You are a tutor writing a short private note to a nursing student who has
just worked through a {node_word} on "{topic}".

{task}

WHAT IT COVERED:
{covered_block}

{evidence_heading}:
{struggle_block}

UP NEXT: {next_block}

RULES:
- ONE sentence. Two only if the second earns its place. Under 35 words total.
- Plain, everyday words. Short sentences. Keep the clinical terms she is
  actually studying; simplify everything around them. If a sentence needs
  reading twice, rewrite it.
- Address her as "you". Use contractions. Sound like a person.
- NEVER just restate what the node was about in longer words — she has just
  worked through it, so a sentence she could have written before starting is
  worth nothing.
- No numbers, no percentages, no counts.
- No praise words ("great", "excellent", "well done", "nice work"), no
  exclamation marks, and never congratulate her for reading something.
- Never use the words "analysis", "performance", "detected" or "weakness".
- Do not claim anything about how she answers that is not stated above.
- Write in {language_label}.

Return ONLY the note text."""

    try:
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
        response = await llm.ainvoke(prompt)
        note = _trim_note((response.content or "").strip().strip('"'))

        # Same guards as the scored note: this asserts something about her, so
        # a runaway, empty or number-smuggling response is dropped rather than
        # shown. A missing note degrades to the plain acknowledgement.
        if not note or len(note) > STUDY_NOTE_MAX_CHARS:
            return None
        if re.search(r"\d+\s*%", note):
            return None

        print(f"   reflection note ({mode}): {note[:70]}")
        return {"text": note, "mode": mode, "linked": fixed or linked}
    except Exception as e:
        print(f"   reflection note failed ({e})")
        return None


@app.post("/study/node-debrief")
async def node_debrief(request: NodeDebriefRequest):
    """
    Post-node coaching moment: "I noticed something about how you answer."

    The goal is DISCOVERY, not reporting. A score tells a student how she did;
    this is meant to tell her something about herself she could not have seen —
    specifically the difference between not knowing the content and knowing it
    but reasoning through the question the wrong way. Those need opposite fixes
    and a percentage cannot tell them apart.

    Split of responsibility, deliberately:
      * PYTHON decides whether a pattern exists and computes the evidence.
      * The MODEL only writes the voice around numbers it was handed.

    A model asked to find its own pattern will always find one, and a
    confidently invented pattern is worse than no insight at all — it teaches
    the student to distrust everything else the product says. So when the
    evidence is thin this returns hasPattern=False and admits it.

    Never raises: the transition screen must render regardless.
    """
    NEWLINE = chr(10)

    print("")
    print("=" * 60)
    print(f"NODE INSIGHT - chat: {request.chat_id} | topic: {request.topic}")
    print(f"   {request.score_percent}% over {len(request.items)} items")
    print("=" * 60)

    # ── Unscored nodes: there is no pattern to find ──────────────────────
    # A lesson, an audio explanation or a concept map produce no right/wrong,
    # so every one of the format buckets below is empty and the whole pattern
    # apparatus would return "no pattern yet" — a true statement dressed as a
    # finding. These get the reflection note instead, which is built from her
    # record rather than from answers she never gave.
    UNSCORED = ("lesson", "audio", "mindmap")
    if request.node_type in UNSCORED:
        note = await _build_reflection_note(request, _language_for_prompt(request.language))
        return {
            "hasPattern": False,
            "noticed": "",
            "evidence": [],
            "pattern": "",
            "skill": "",
            "toPattern": 0,
            "note": (note or {}).get("text", ""),
            "noteMode": (note or {}).get("mode", ""),
            "linked": (note or {}).get("linked", []),
            "stillLooking": "",
            "generated": bool(note),
        }

    # ── Flashcards: scored, but not across formats ───────────────────────
    # Every card is the same shape, so the strong-bucket/weak-bucket split
    # below can never fire and would spend a round trip proving it. The note
    # reads the cards directly, which is the only real finding available.
    if request.node_type == "flashcard":
        note = await _build_study_note(request, _language_for_prompt(request.language))
        return {
            "hasPattern": False,
            "noticed": "",
            "evidence": [],
            "pattern": "",
            "skill": "",
            "toPattern": 0,
            "note": (note or {}).get("text", ""),
            "noteMode": (note or {}).get("mode", ""),
            "linked": [],
            "stillLooking": "",
            "generated": bool(note),
        }

    # ── Buckets ──────────────────────────────────────────────────────────
    # Three ways a nursing question can be hard, each needing a different fix:
    #   knowledge  — do you know the fact (mcq)
    #   priority   — can you order actions / read a scenario (casestudy)
    #   multi      — can you judge each option independently (sata)
    BUCKET_OF = {"mcq": "knowledge", "casestudy": "priority", "sata": "multi"}
    BUCKET_NAME = {
        "knowledge": "knowledge",
        "priority": "prioritization",
        "multi": "select-all-that-apply",
    }
    BUCKETS = ("knowledge", "priority", "multi")

    def tally(items):
        out = {b: {"correct": 0, "total": 0} for b in BUCKETS}
        for it in items:
            b = BUCKET_OF.get(it.question_type)
            if not b:
                continue
            out[b]["total"] += 1
            if it.correct:
                out[b]["correct"] += 1
        return out

    node = tally(request.items)

    # Plan-wide totals, so a pattern can be claimed across the session rather
    # than off one unlucky quiz.
    plan = {b: {"correct": 0, "total": 0} for b in BUCKETS}
    for pf in request.plan_formats:
        b = BUCKET_OF.get(pf.get("type"))
        if not b:
            continue
        plan[b]["correct"] += pf.get("correct") or 0
        plan[b]["total"] += pf.get("total") or 0

    # Prefer whichever view has more evidence behind it.
    combined = {}
    for b in BUCKETS:
        src = plan[b] if plan[b]["total"] >= node[b]["total"] else node[b]
        combined[b] = dict(src)

    missed_items = [i for i in request.items if not i.correct]
    correct_items = [i for i in request.items if i.correct]

    def acc(d):
        return (d["correct"] / d["total"]) if d["total"] else None

    # ── Does a real pattern exist? ───────────────────────────────────────
    # Requires BOTH sides: demonstrated strength somewhere, and a clearly
    # sampled weakness somewhere else. Without the strong side this is just
    # "you are bad at this", which is not an insight and does not land.
    MIN_WEAK = 3
    MIN_STRONG = 3

    # Every accuracy test below is written `is not None` rather than leaning on
    # truthiness, and that is not style — it is the bug this replaces.
    #
    # acc() returns None for an unsampled bucket and a float otherwise, so the
    # guards used to read `(acc(...) or 1) < 0.55` to mean "unsampled buckets
    # can't be the weak one". But 0.0 is falsy too. A bucket she got ENTIRELY
    # wrong scored as 1.0 and was dropped from the candidates — so the student
    # who missed every single select-all, the clearest weakness this endpoint
    # can ever observe, was the one student it had nothing to say to. She got
    # "I'm still figuring out your pattern" instead of the pattern.
    #
    # The totals are already checked on the same line, so acc() cannot be None
    # by the time it is compared; the explicit test is kept anyway so nobody
    # reintroduces a fallback to make it read shorter.
    strong_bucket = None
    k = combined["knowledge"]
    k_acc = acc(k)
    if k["total"] >= MIN_STRONG and k_acc is not None and k_acc >= 0.75:
        strong_bucket = "knowledge"

    weak_candidates = []
    for b in ("priority", "multi"):
        b_acc = acc(combined[b])
        if combined[b]["total"] >= MIN_WEAK and b_acc is not None and b_acc < 0.55:
            weak_candidates.append((b, b_acc))
    # Weakest first — sorting on the accuracy already computed, rather than
    # recomputing it through another falsiness fallback.
    weak_candidates.sort(key=lambda kv: kv[1])
    weak_bucket = weak_candidates[0][0] if weak_candidates else None

    if not (strong_bucket and weak_bucket):
        # How much evidence is actually missing, so the screen can say "N more
        # of these and I'll have something specific" instead of an open-ended
        # "keep going". Only a real SAMPLING shortfall counts: if both sides
        # are sampled and she is simply good at all of them, more questions
        # will not produce a pattern and the promise would be a lie. 0 means
        # "don't promise anything".
        shortfalls = []
        if not strong_bucket and combined["knowledge"]["total"] < MIN_STRONG:
            shortfalls.append(MIN_STRONG - combined["knowledge"]["total"])
        if not weak_bucket:
            under_sampled = [
                MIN_WEAK - combined[b]["total"]
                for b in ("priority", "multi")
                if combined[b]["total"] < MIN_WEAK
            ]
            if under_sampled:
                shortfalls.append(min(under_sampled))
        to_pattern = max(shortfalls) if shortfalls else 0

        print(f"   no pattern yet - insufficient evidence (needs {to_pattern} more)")

        # No format pattern does not mean nothing to say. Read the actual
        # questions and write a note about THIS quiz, which is what the
        # student just spent her time on.
        note = await _build_study_note(request, _language_for_prompt(request.language))

        return {
            "hasPattern": False,
            "noticed": "",
            "evidence": [],
            "pattern": "",
            "skill": "",
            "toPattern": to_pattern,
            "note": (note or {}).get("text", ""),
            "noteMode": (note or {}).get("mode", ""),
            "stillLooking": "I'm still figuring out your pattern. Keep going and I'll look for one.",
            "generated": False,
        }

    # ── Evidence: computed, never written by the model ───────────────────
    s, w = combined[strong_bucket], combined[weak_bucket]
    evidence = [
        f"{s['correct']} of {s['total']} {BUCKET_NAME[strong_bucket]} questions correct",
        f"{w['correct']} of {w['total']} on {BUCKET_NAME[weak_bucket]}",
    ]
    node_w = node[weak_bucket]
    node_misses_in_weak = node_w["total"] - node_w["correct"]
    if len(missed_items) >= 2 and node_misses_in_weak >= 2:
        evidence.append(
            f"{node_misses_in_weak} of your {len(missed_items)} misses here "
            f"involved {BUCKET_NAME[weak_bucket]}"
        )

    print(f"   pattern: strong={strong_bucket} weak={weak_bucket}")
    print(f"   evidence: {evidence}")

    skill = BUCKET_NAME[weak_bucket]

    def fallback():
        """Deterministic phrasing. Same shape, no model involved."""
        if weak_bucket == "priority":
            pattern = (
                "You tend to identify the right clinical problem, but sometimes choose an "
                "intervention before deciding what has to happen first. That's a reasoning "
                "pattern, not a gap in what you know."
            )
        else:
            pattern = (
                "You know the material, but these are graded option by option, and one wrong "
                "pick loses the whole question. That's a technique pattern, not a gap in what "
                "you know."
            )
        return {
            "hasPattern": True,
            "noticed": "You're stronger on this content than your score suggests.",
            "evidence": evidence,
            "pattern": pattern,
            "skill": skill,
            "stillLooking": "",
            "generated": False,
        }

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
        prompt_language = _language_for_prompt(request.language)

        def render(items, cap):
            out = []
            for i in items[:cap]:
                label = BUCKET_NAME.get(BUCKET_OF.get(i.question_type), "other")
                out.append(f"  - [{label}] {i.question[:200]}")
            return NEWLINE.join(out) if out else "  (none)"

        exam_note = ""
        if request.days_until_exam is not None and 0 <= request.days_until_exam <= 14:
            when = "today" if request.days_until_exam == 0 else (
                "tomorrow" if request.days_until_exam == 1
                else f"in {request.days_until_exam} days"
            )
            exam_note = NEWLINE + f"Her exam is {when}."

        insight_prompt = f"""You are a nursing tutor sitting next to a student who has just finished a
{request.node_type} on "{request.topic}". You have spotted something about HOW she answers.

WHAT SHE GOT RIGHT:
{render(correct_items, 6)}

WHAT SHE MISSED:
{render(missed_items, 6)}

THE PATTERN, ALREADY CONFIRMED FROM HER DATA. Do not question it, do not compute your
own, do not invent any other statistic:
  Strong at: {BUCKET_NAME[strong_bucket]} - {s['correct']} of {s['total']}
  Weak at:   {BUCKET_NAME[weak_bucket]} - {w['correct']} of {w['total']}{exam_note}

Write the moment she realises this about herself, in {prompt_language}. Two fields.

VOICE: a tutor who has just noticed something interesting, talking to her directly as
"you". Warm, curious, on her side. Use contractions. Sound like a person, never a report.
Never use the words "detected", "analysis", "performance" or "weakness", and no
exclamation marks. Do not congratulate her generically. She is preparing for a real exam
and being patronised will lose her.

- "noticed": ONE sentence, at most 20 words. What you noticed about her, leading with the
  STRENGTH - she is better than her score suggests. No numbers; the evidence is shown
  separately underneath.
- "pattern": 2 to 3 sentences, at most 55 words. Explain the DISTINCTION: what she does
  right, then the specific move that trips her up. End by naming it as a reasoning or
  technique pattern rather than a knowledge gap. This is the sentence that should make her
  think "that is exactly what I do".

Return ONLY valid JSON:
{{"noticed": "...", "pattern": "..."}}"""

        response = await llm.ainvoke([{"role": "user", "content": insight_prompt}])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        if not result.get("noticed") or not result.get("pattern"):
            print("Insight missing fields - using fallback")
            return fallback()

        print(f"Insight: {result['noticed'][:70]}")
        return {
            "hasPattern": True,
            "noticed": result["noticed"],
            "evidence": evidence,
            "pattern": result["pattern"],
            "skill": skill,
            "stillLooking": "",
            "generated": True,
        }

    except Exception as e:
        print(f"Insight generation failed, serving fallback: {e}")
        return fallback()


@app.post("/study/generate-exam")
async def generate_study_exam(request: StudyExamRequest):
    """
    Generate a mixed-format NCLEX-style exam for a study session.
    Reuses the existing quiz generation infrastructure with question_types support.

    Returns questions with questionType field so the frontend routes to the
    correct renderer (MCQ, SATA, CaseStudy).
    """
    print(f"\n{'='*60}")
    print(f"📝 EXAM GENERATION - chat_id: {request.chat_id}")
    print(f"   Topic: {request.topic}")
    print(f"   Types: {request.question_types}")
    print(f"   Count: {request.question_count}")
    print(f"   Custom: {request.custom_instructions or 'none'}")
    print(f"{'='*60}")

    # Free-tier quota gate. NOTE: matches the client's exam grace — any
    # remaining budget admits the full exam; only an empty bucket blocks.
    # (Raised before the try so `except Exception` can't turn it into a 500.)
    quota = usage_guard.check_quota(request.chat_id)
    if not quota["allowed"]:
        print(f"🚫 Quota exceeded for chat {request.chat_id} — rejecting exam")
        raise HTTPException(status_code=429, detail={
            "code": "quota_exceeded", "message": usage_guard.QUOTA_MESSAGE
        })

    try:
        session = await _setup_study_session(request.chat_id, request.language)
        source = "documents" if session.documents and session.vectorstore else "scratch"

        # Build the topic string — include custom instructions if provided
        exam_topic = request.topic
        if request.custom_instructions:
            exam_topic = f"{request.topic}. Student instructions: {request.custom_instructions}"

        questions = []
        async for chunk in stream_quiz_with_bank(
            topic=exam_topic,
            difficulty="medium",
            num_questions=request.question_count,
            source=source,
            session=session,
            chat_id=session.chat_id,
            question_types=request.question_types,
            quiz_mode="knowledge"
        ):
            if chunk.get("status") == "question_ready":
                question = chunk.get("question")
                if question:
                    q_type = question.get('questionType', 'mcq')

                    if q_type == 'mcq':
                        answer = question.get('answer', 'A)')
                        answer_letter = answer[0] if answer else 'A'
                        correct_index = ord(answer_letter) - ord('A')
                        questions.append({
                            "questionType": "mcq",
                            "question": question.get('question', ''),
                            "options": question.get('options', []),
                            "correctIndex": correct_index,
                            # Legacy field kept for back-compat with old saved quizzes.
                            "rationale": question.get('justification', ''),
                            # New one-sentence summary; full rationale on demand.
                            "correctBlurb": question.get('correct_blurb', ''),
                            "topic": question.get('topic', request.topic)
                        })
                    elif q_type == 'sata':
                        questions.append({
                            **question,
                            "questionType": "sata",
                            "topic": question.get('topic', request.topic)
                        })
                    elif q_type == 'casestudy':
                        questions.append({
                            **question,
                            "questionType": "casestudy",
                            "topic": question.get('topic', request.topic)
                        })
                    else:
                        questions.append({
                            **question,
                            "topic": question.get('topic', request.topic)
                        })

        content_hash = hashlib.md5(json.dumps(questions, sort_keys=True, default=str).encode()).hexdigest()[:12]

        print(f"✅ Generated exam with {len(questions)} questions")
        type_counts = {}
        for q in questions:
            qt = q.get('questionType', 'mcq')
            type_counts[qt] = type_counts.get(qt, 0) + 1
        print(f"   Distribution: {type_counts}")

        return {
            "questions": questions,
            "hash": content_hash,
            "examConfig": {
                "questionTypes": request.question_types,
                "questionCount": request.question_count,
                "customInstructions": request.custom_instructions
            }
        }

    except Exception as e:
        print(f"❌ Exam generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/study/generate-item")
async def generate_study_item(request: StudyItemRequest):
    """
    Generate content for a single study node on-demand.

    WHY ON-DEMAND:
    - Saves cost (only generate what user actually views)
    - Feels more dynamic and personalized
    - Uses anti-repeat hashes to avoid duplicate content

    CONTENT TYPES:
    - lesson: {title, body, keyPoints[]}
    - flashcard: {front, back}
    - quiz: {question, options[], correctIndex, rationale}
    - audio: {topic, intent, suggestedDuration}

    Returns:
    - type: the node type
    - content: the generated content object
    - hash: content hash for anti-repeat tracking
    """
    print(f"\n{'='*60}")
    print(f"📝 STUDY ITEM GENERATION - {request.node_type}: {request.node_label}")
    print(f"{'='*60}")

    # Free-tier quota gate (server-side; raised before the try so the
    # blanket `except Exception` can't turn the 429 into a 500).
    quota = usage_guard.check_quota(request.chat_id)
    if not quota["allowed"]:
        print(f"🚫 Quota exceeded for chat {request.chat_id} — rejecting study item")
        raise HTTPException(status_code=429, detail={
            "code": "quota_exceeded", "message": usage_guard.QUOTA_MESSAGE
        })

    try:
        # ------------------------------------------
        # STEP 1: Setup session using PersistentSessionContext
        # This mirrors the orchestrator's session setup for consistency
        # ------------------------------------------
        session = await _setup_study_session(request.chat_id, request.language)

        # ------------------------------------------
        # STEP 2: Generate content based on type
        # Uses the SAME streaming generators as chat tools
        # ------------------------------------------
        content = None

        if request.node_type == "lesson":
            # Lessons use enhanced context retrieval (k=1000)
            content = await _generate_lesson_with_context(
                session, request.node_label, request.language
            )

        elif request.node_type == "flashcard":
            # Flashcards use stream_flashcards (same as chat tools)
            content = await _generate_flashcard_via_stream(
                session, request.node_label, num_cards=STUDY_FLASHCARD_CARDS
            )

        elif request.node_type == "quiz":
            # Quizzes use stream_quiz_with_bank (same as chat tools)
            content = await _generate_quiz_via_stream(
                session, request.node_label, num_questions=STUDY_QUIZ_QUESTIONS
            )

        elif request.node_type == "audio":
            # Audio config generation (actual audio generated separately)
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)
            prompt_language = _language_for_prompt(request.language)

            # Get context for audio config
            context = ""
            if session.vectorstore:
                docs = session.vectorstore.similarity_search(query=request.node_label, k=100)
                context = "\n\n".join([doc.page_content for doc in docs])[:1000]

            content = await _generate_audio_config(
                llm, request.node_label, context, prompt_language
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown node type: {request.node_type}")

        # ------------------------------------------
        # STEP 4: Generate content hash for anti-repeat
        # ------------------------------------------
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]

        print(f"✅ Generated {request.node_type} content (hash: {content_hash})")

        return {
            "type": request.node_type,
            "content": content,
            "hash": content_hash
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Study item generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/study/generate-item-stream")
async def generate_study_item_stream(request: StudyItemRequest):
    """
    SSE streaming endpoint for study mode content generation.
    Provides real-time progress updates to frontend.

    This is an OPTIONAL enhancement - the regular /study/generate-item
    endpoint works fine for most use cases.

    Streams status updates in the same format as chat tools:
    - {"status": "generating", "current": 1, "total": 5}
    - {"status": "question_ready", "question": {...}}
    - {"status": "complete", "content": {...}}
    """
    print(f"\n{'='*60}")
    print(f"🌊 STUDY ITEM STREAMING - {request.node_type}: {request.node_label}")
    print(f"{'='*60}")

    async def stream_generator():
        try:
            # Free-tier quota gate — rejected in-stream (status: error) so the
            # frontend's existing stream-error handling picks it up.
            quota = usage_guard.check_quota(request.chat_id)
            if not quota["allowed"]:
                print(f"🚫 Quota exceeded for chat {request.chat_id} — rejecting item stream")
                yield f"data: {json.dumps({'status': 'error', 'code': 'quota_exceeded', 'message': usage_guard.QUOTA_MESSAGE})}\n\n"
                return

            # Setup session
            session = await _setup_study_session(request.chat_id, request.language)

            # Determine source
            source = "documents" if session.documents and session.vectorstore else "scratch"

            if request.node_type == "quiz":
                # Stream quiz generation.
                # Diagnostic (first node of a plan) is deliberately short — it
                # exists to calibrate the plan, not to test. See StudyItemRequest.
                quiz_size = request.num_questions or (
                    STUDY_DIAGNOSTIC_QUESTIONS if request.is_diagnostic
                    else STUDY_QUIZ_QUESTIONS
                )
                questions = []
                async for chunk in stream_quiz_with_bank(
                    topic=request.node_label,
                    difficulty="medium",
                    num_questions=quiz_size,
                    source=source,
                    session=session,
                    chat_id=session.chat_id,
                    question_types=STUDY_QUIZ_TYPES,
                    quiz_mode="knowledge"
                ):
                    # Forward status updates to frontend
                    yield f"data: {json.dumps(chunk)}\n\n"

                    # Collect questions for final response
                    if chunk.get("status") == "question_ready":
                        question = chunk.get("question")
                        if question:
                            questions.append(
                                _format_study_question(question, request.node_label)
                            )

                # Send final content
                content = {"questions": questions}
                content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                yield f"data: {json.dumps({'status': 'complete', 'type': 'quiz', 'content': content, 'hash': content_hash})}\n\n"

            elif request.node_type == "flashcard":
                # Stream flashcard generation
                cards = []
                async for chunk in stream_flashcards(
                    topic=request.node_label,
                    num_cards=STUDY_FLASHCARD_CARDS,
                    source=source,
                    session=session,
                    chat_id=session.chat_id
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"

                    if chunk.get("status") == "flashcard_ready":
                        flashcard = chunk.get("flashcard")
                        if flashcard:
                            cards.append({
                                "front": flashcard.get("front", ""),
                                "back": flashcard.get("back", ""),
                                "topic": flashcard.get("topic", request.node_label)
                            })

                content = {"cards": cards}
                content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                yield f"data: {json.dumps({'status': 'complete', 'type': 'flashcard', 'content': content, 'hash': content_hash})}\n\n"

            elif request.node_type == "lesson":
                # Streams page-by-page (see _stream_lesson_with_context) so the
                # student can start reading page 1 while the rest is written.
                # Falls back to the blocking generator if the stream dies before
                # producing a usable payload.
                yield f"data: {json.dumps({'status': 'generating', 'message': 'Creating lesson...'})}\n\n"

                content = None
                try:
                    async for chunk in _stream_lesson_with_context(
                        session, request.node_label, request.language
                    ):
                        if chunk.get("status") == "lesson_content":
                            content = chunk.get("content")
                        else:
                            yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as lesson_err:
                    print(f"⚠️ Lesson stream failed, falling back to blocking: {lesson_err}")

                if not content or not content.get("pages"):
                    content = await _generate_lesson_with_context(
                        session, request.node_label, request.language
                    )

                content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                yield f"data: {json.dumps({'status': 'complete', 'type': 'lesson', 'content': content, 'hash': content_hash})}\n\n"

            elif request.node_type == "audio":
                yield f"data: {json.dumps({'status': 'generating', 'message': 'Preparing audio config...'})}\n\n"

                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)
                prompt_language = _language_for_prompt(request.language)
                context = ""
                if session.vectorstore:
                    docs = session.vectorstore.similarity_search(query=request.node_label, k=100)
                    context = "\n\n".join([doc.page_content for doc in docs])[:1000]

                content = await _generate_audio_config(
                    llm, request.node_label, context, prompt_language
                )
                content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                yield f"data: {json.dumps({'status': 'complete', 'type': 'audio', 'content': content, 'hash': content_hash})}\n\n"

            elif request.node_type == "mindmap":
                # Return a stub so the frontend can render StudyMindmapCard,
                # which auto-triggers /study/generate-mindmap for the actual map.
                yield f"data: {json.dumps({'status': 'generating', 'message': 'Preparing concept map...'})}\n\n"
                content = {
                    "topic": request.node_label,
                    "depth": "medium",
                    "mindmapData": None  # populated later by /study/generate-mindmap
                }
                content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
                yield f"data: {json.dumps({'status': 'complete', 'type': 'mindmap', 'content': content, 'hash': content_hash})}\n\n"

            else:
                yield f"data: {json.dumps({'status': 'error', 'message': f'Unknown node type: {request.node_type}'})}\n\n"

        except Exception as e:
            print(f"❌ Study item streaming failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


@app.post("/study/generate-audio")
async def generate_study_audio(request: StudyAudioRequest):
    """
    Generate audio content for a study mode node.

    This is a streaming endpoint that yields progress updates and finally
    the audio data (base64 encoded).

    Reuses the same AudioGenerator as the chat tools for consistency.
    """
    print(f"\n{'='*60}")
    print(f"🎵 STUDY AUDIO GENERATION - {request.topic}")
    print(f"{'='*60}")

    async def stream_generator():
        try:
            # Free-tier quota gate — audio streams use 'audio_error' as their
            # error status, so reject with that shape.
            quota = usage_guard.check_quota(request.chat_id)
            if not quota["allowed"]:
                print(f"🚫 Quota exceeded for chat {request.chat_id} — rejecting audio")
                yield f"data: {json.dumps({'status': 'audio_error', 'code': 'quota_exceeded', 'message': usage_guard.QUOTA_MESSAGE})}\n\n"
                return

            # Setup session
            session = await _setup_study_session(request.chat_id, request.language)

            # Import AudioGenerator
            from services.audio_generator import AudioGenerator
            generator = AudioGenerator(session)

            # Convert duration from int minutes to string format (e.g., "2min")
            duration_str = f"{request.duration}min"

            # Stream audio generation
            async for chunk in generator.generate_audio_stream(
                topic=request.topic,
                intent=request.intent,
                duration=duration_str,
                language=request.language
            ):
                # chunk already has \n at end, strip it and format properly for SSE
                chunk_clean = chunk.rstrip('\n')

                # Log when sending audio_ready (large payload)
                if '"status": "audio_ready"' in chunk_clean:
                    print(f"📤 Sending audio_ready response ({len(chunk_clean)} bytes)")

                yield f"data: {chunk_clean}\n\n"

            print("✅ Audio stream complete")

        except Exception as e:
            print(f"❌ Study audio generation failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'audio_error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


@app.post("/study/generate-mindmap")
async def generate_study_mindmap(request: StudyMindmapRequest):
    """
    Generate a concept map for a study mode node.

    Streaming endpoint that reuses the existing mindmap_generator service.
    Yields mindmap_generating → mindmap_complete (or error).
    """
    print(f"\n{'='*60}")
    print(f"🧠 STUDY MINDMAP GENERATION - {request.topic}")
    print(f"{'='*60}")

    async def stream_generator():
        try:
            # Free-tier quota gate — rejected in-stream (status: error).
            quota = usage_guard.check_quota(request.chat_id)
            if not quota["allowed"]:
                print(f"🚫 Quota exceeded for chat {request.chat_id} — rejecting mindmap")
                yield f"data: {json.dumps({'status': 'error', 'code': 'quota_exceeded', 'message': usage_guard.QUOTA_MESSAGE})}\n\n"
                return

            session = await _setup_study_session(request.chat_id, request.language)
            session.user_language = request.language

            from services.mindmap_generator import stream_mindmap_data

            async for chunk in stream_mindmap_data(
                topic=request.topic,
                depth=request.depth,
                session=session,
                chat_id=request.chat_id
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

            print("✅ Mindmap stream complete")

        except Exception as e:
            print(f"❌ Study mindmap generation failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


# ============================================================================
# STUDY ITEM GENERATION HELPERS (LEGACY)
# These are kept for backward compatibility but the new streaming wrappers
# above are preferred as they use the same generators as chat tools
# ============================================================================

async def _generate_lesson(llm, label: str, context: str, language: str, asked_hashes: list) -> dict:
    """
    Generate a MULTI-PAGE lesson with swipeable cards.

    Duolingo-style - ONE concept per page, fun and engaging!
    STRICT: Only use content from provided documents.
    """
    prompt = f"""Create a MULTI-PAGE lesson about: {label}

🚨 CRITICAL: USE ONLY THIS DOCUMENT CONTENT - DO NOT HALLUCINATE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context[:8000] if context else "ERROR: No document context provided"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. ONLY teach concepts that appear in the document above
2. Use the EXACT terminology from the document
3. If a concept is not in the document, DO NOT include it
4. Quote or paraphrase directly from the document content

STYLE: Duolingo-style swipeable cards. ONE concept per page.

STRUCTURE:
- Page 1: Introduction (what topic is this about, from the document)
- Pages 2-5: ONE key concept per page FROM THE DOCUMENT
- Last page: Quick summary of what was taught

EACH PAGE MUST HAVE:
- "title": Short catchy title (3-5 words)
- "content": 1-2 sentences MAX with **bold** key terms FROM THE DOCUMENT
- "highlight": The ONE thing to remember (optional)

Write in {language}.

Return ONLY valid JSON:
{{
  "title": "Main Lesson Title",
  "pages": [
    {{"title": "👋 Welcome!", "content": "...", "highlight": null}},
    {{"title": "Key Point 1", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 2", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 3", "content": "...", "highlight": "..."}},
    {{"title": "🎯 Summary", "content": "...", "highlight": null}}
  ]
}}"""

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    # Parse response
    content_json = response.content.strip()
    if content_json.startswith("```"):
        content_json = content_json.split("```")[1]
        if content_json.startswith("json"):
            content_json = content_json[4:]
        content_json = content_json.strip()

    return json.loads(content_json)


async def _generate_flashcard(llm, label: str, context: str, language: str, asked_hashes: list) -> dict:
    """
    Generate 12 high-quality flashcards for a single node using the enhanced generator.

    Uses the _generate_single_flashcard from flashcard_tools which produces:
    - Ultra-short, scannable answers with bullets and bold formatting
    - Topic assignment for organization
    - Better deduplication across cards

    Returns an object with a 'cards' array containing 12 flashcards.
    Each card has front/back/topic.
    """
    from tools.flashcard_tools import _generate_single_flashcard

    # Build content context for the generator
    if context:
        content_context = f"Document content:\n{context[:15000]}"
    else:
        content_context = f"""You are generating flashcards about: {label}

        Cover key concepts, definitions, and important facts that students need to memorize.
        If this is a broad topic, ensure diverse coverage of subtopics."""

    cards = []
    generated_fronts = []
    existing_topics = []

    # Generate 12 flashcards one at a time for better quality
    for card_num in range(1, 13):
        flashcard_data = await _generate_single_flashcard(
            content=content_context,
            topic=label,
            card_num=card_num,
            language=language,
            cards_to_avoid=generated_fronts,
            existing_topics=existing_topics
        )

        if flashcard_data:
            # Track front for deduplication
            generated_fronts.append(flashcard_data['front'])

            # Track topic for consistency
            card_topic = flashcard_data.get('topic', '')
            if card_topic and card_topic not in existing_topics:
                existing_topics.append(card_topic)

            cards.append(flashcard_data)
            print(f"✅ Flashcard {card_num}/12 generated - Topic: {flashcard_data.get('topic', 'N/A')}")

    # Ensure we have at least some cards
    if not cards:
        # Fallback to simple generation if enhanced method fails
        print("⚠️ Enhanced flashcard generation failed, using fallback")
        cards = [
            {"front": f"What is {label}?", "back": f"Definition of {label}...", "topic": label},
            {"front": f"Key features of {label}?", "back": f"The key features are...", "topic": label},
            {"front": f"Why is {label} important?", "back": f"It is important because...", "topic": label},
            {"front": f"How is {label} applied?", "back": f"It is applied by...", "topic": label},
            {"front": f"Common mistakes with {label}?", "back": f"Common mistakes include...", "topic": label}
        ]

    print(f"📇 Generated {len(cards)} high-quality flashcards")
    return {"cards": cards}


async def _generate_quiz_item(llm, label: str, context: str, language: str, asked_hashes: list) -> dict:
    """
    Generate 12 high-quality multiple choice questions for a single node.

    Uses the _generate_single_question from quiztools which produces:
    - Random answer positioning (not always B or C)
    - Topic assignment for organization
    - Better justifications with bold formatting
    - Knowledge mode (factual recall, not NCLEX scenarios)

    Returns an object with a 'questions' array containing 12 quiz questions.
    Each question has: question, options, correctIndex, rationale, topic.
    """
    import random
    from tools.quiztools import _generate_single_question

    # Build content context for the generator
    if context:
        content_context = f"Document content:\n{context[:12000]}"
    else:
        content_context = f"""You are generating questions about: {label}

            If this is a broad topic (like 'research design', 'pharmacology', 'cardiac care'),
            ensure you test diverse subtopics and concepts within that domain."""

    questions = []
    generated_question_texts = []
    existing_topics = []

    # Generate 12 questions one at a time for better quality
    for question_num in range(1, 13):
        # Random answer position for each question
        random_target_letter = random.choice(['A', 'B', 'C', 'D'])

        question_data = await _generate_single_question(
            content=content_context,
            topic=label,
            difficulty="medium",
            question_num=question_num,
            language=language,
            questions_to_avoid=generated_question_texts,
            target_letter=random_target_letter,
            existing_topics=existing_topics,
            quiz_mode="knowledge"  # Use knowledge mode for study path (factual, not NCLEX scenarios)
        )

        if question_data:
            # Track question text for deduplication
            generated_question_texts.append(question_data['question'])

            # Track topic for consistency
            question_topic = question_data.get('topic', '')
            if question_topic and question_topic not in existing_topics:
                existing_topics.append(question_topic)

            # Convert format for frontend compatibility
            # Frontend expects: { question, options, correctIndex, rationale }
            answer = question_data.get('answer', 'A)')
            answer_letter = answer[0] if answer else 'A'
            correct_index = ord(answer_letter) - ord('A')

            formatted_question = {
                "question": question_data['question'],
                "options": question_data['options'],
                "correctIndex": correct_index,
                # Legacy field kept for back-compat with old saved quizzes.
                "rationale": question_data.get('justification', ''),
                # New one-sentence summary; full rationale on demand.
                "correctBlurb": question_data.get('correct_blurb', ''),
                "topic": question_data.get('topic', label)
            }

            questions.append(formatted_question)
            print(f"✅ Question {question_num}/12 generated - Answer: {answer_letter}, Topic: {formatted_question['topic']}")

    # Ensure we have at least some questions
    if not questions:
        # Fallback to simple generation if enhanced method fails
        print("⚠️ Enhanced quiz generation failed, using fallback")
        questions = [
            {
                "question": f"What is the primary purpose of {label}?",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "correctIndex": 0,
                "rationale": f"Option A is correct because it describes {label}.",
                "topic": label
            }
        ]

    print(f"❓ Generated {len(questions)} high-quality quiz questions")
    return {"questions": questions}


async def _generate_audio_config(llm, label: str, context: str, language: str) -> dict:
    """
    Generate configuration for an audio lesson.

    This returns the TOPIC and INTENT, not the actual audio.
    The frontend will use this to call the existing audio generation endpoint.
    """
    prompt = f"""Create an audio lesson config for: {label}

CONTEXT:
{context[:1000] if context else "No specific context"}

Return a configuration for a 1-2 minute audio explanation.

Return ONLY valid JSON:
{{
  "topic": "Clear topic title for audio",
  "intent": "teach",
  "suggestedDuration": 2
}}"""

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    content_json = response.content.strip()
    if content_json.startswith("```"):
        content_json = content_json.split("```")[1]
        if content_json.startswith("json"):
            content_json = content_json[4:]
        content_json = content_json.strip()

    return json.loads(content_json)


# ============================================================================
# STUDY MODE - STREAMING GENERATOR WRAPPERS
# These functions reuse the same generators as the chat tools for consistency
# ============================================================================

async def _setup_study_session(chat_id: str, language: str = "en") -> PersistentSessionContext:
    """
    Create a properly configured session for study mode that mirrors
    the orchestrator's session setup. This ensures document context
    is available to the streaming generators.

    Warm path: if the chat already has a NursingTutor in ACTIVE_SESSIONS with
    its vectorstore loaded (e.g. populated during upload or a prior /study/*
    call), reuse that session instead of re-downloading the FAISS index from
    Firebase Storage. The cold path is unchanged.
    """
    # Warm path — reuse the session that the upload flow / prior calls left behind.
    if chat_id in ACTIVE_SESSIONS:
        existing = ACTIVE_SESSIONS[chat_id].session
        if existing.vectorstore is not None:
            existing.user_language = language
            set_session_context(existing)
            print(f"♻️ Study session reused for {chat_id} (vectorstore already loaded)")
            return existing

    # Cold path — build a fresh session.
    session = PersistentSessionContext(chat_id)
    session.user_language = language

    # Load vectorstore (document embeddings) - same as orchestrator
    session.vectorstore = get_chat_vectorstore(chat_id)

    # Load file list - same as orchestrator
    session.documents = load_files_for_chat(chat_id)

    # Set as global context for tools to access
    set_session_context(session)

    print(f"📚 Study session setup: {len(session.documents)} documents, vectorstore: {session.vectorstore is not None}")

    return session


async def _generate_quiz_via_stream(
    session: PersistentSessionContext,
    topic: str,
    num_questions: int = 5
) -> dict:
    """
    Generate quiz questions using stream_quiz_with_bank.
    Collects all results before returning (sync response for study mode).

    Uses the SAME generator as the chat tools, ensuring:
    - Document context via k=1000 similarity search
    - Proper deduplication
    - Consistent question quality
    """
    questions = []

    # Determine source based on available documents
    source = "documents" if session.documents and session.vectorstore else "scratch"

    print(f"🎯 Generating quiz via stream_quiz_with_bank: topic='{topic}', source='{source}'")

    async for chunk in stream_quiz_with_bank(
        topic=topic,
        difficulty="medium",
        num_questions=num_questions,
        source=source,
        session=session,
        chat_id=session.chat_id,
        question_types=STUDY_QUIZ_TYPES,  # MCQ-weighted, one SATA guaranteed
        quiz_mode="knowledge"    # Study mode uses knowledge mode (factual questions)
    ):
        if chunk.get("status") == "question_ready":
            question = chunk.get("question")
            if question:
                formatted_question = _format_study_question(question, topic)
                questions.append(formatted_question)
                print(f"✅ Quiz question {len(questions)}/{num_questions} collected")

    print(f"❓ Generated {len(questions)} quiz questions via streaming generator")
    return {"questions": questions}


async def _generate_flashcard_via_stream(
    session: PersistentSessionContext,
    topic: str,
    num_cards: int = 5
) -> dict:
    """
    Generate flashcards using stream_flashcards.
    Collects all results before returning (sync response for study mode).

    Uses the SAME generator as the chat tools, ensuring:
    - Document context via k=1000 similarity search
    - Proper deduplication
    - Consistent flashcard quality
    """
    cards = []

    # Determine source based on available documents
    source = "documents" if session.documents and session.vectorstore else "scratch"

    print(f"📇 Generating flashcards via stream_flashcards: topic='{topic}', source='{source}'")

    async for chunk in stream_flashcards(
        topic=topic,
        num_cards=num_cards,
        source=source,
        session=session,
        chat_id=session.chat_id
    ):
        if chunk.get("status") == "flashcard_ready":
            flashcard = chunk.get("flashcard")
            if flashcard:
                cards.append({
                    "front": flashcard.get("front", ""),
                    "back": flashcard.get("back", ""),
                    "topic": flashcard.get("topic", topic)
                })
                print(f"✅ Flashcard {len(cards)}/{num_cards} collected")

    print(f"📇 Generated {len(cards)} flashcards via streaming generator")
    return {"cards": cards}


async def _generate_lesson_with_context(
    session: PersistentSessionContext,
    topic: str,
    language: str
) -> dict:
    """
    Generate lesson using document context properly.
    Uses the SAME document retrieval pattern as other tools (k=1000).
    STRICT: Only generates content from the uploaded documents.
    """
    # Use same retrieval pattern as flashcards/quizzes: k=1000
    context = ""
    if session.vectorstore:
        docs = session.vectorstore.similarity_search(query=topic, k=1000)
        full_text = "\n\n".join([doc.page_content for doc in docs])
        context = full_text[:12000]  # Increased limit for better coverage
        print(f"📚 Lesson context: {len(docs)} chunks, {len(context)} chars")

    # If no documents, we cannot generate - this should not happen in study mode
    if not context:
        print(f"⚠️ WARNING: No document context for lesson on '{topic}'")
        context = "NO DOCUMENT CONTENT AVAILABLE"

    # Generate lesson with proper context
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)  # Lower temp for accuracy
    prompt_language = _language_for_prompt(language)

    prompt = f"""Create a MULTI-PAGE lesson about: {topic}

🚨🚨🚨 CRITICAL INSTRUCTION 🚨🚨🚨
You MUST ONLY use information from the document content below.
DO NOT add any information from your general knowledge.
DO NOT hallucinate or make up facts.
If something is not in the document, DO NOT include it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT'S DOCUMENT CONTENT (USE ONLY THIS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context[:10000]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
1. Extract ONLY facts/concepts that appear in the document above
2. Use the EXACT terminology and definitions from the document
3. Every "content" field must be paraphrased from the document
4. Every "highlight" must quote or summarize document content
5. If the document doesn't cover enough for 5 pages, use fewer pages

STYLE: Duolingo-style swipeable cards. ONE concept per page.

STRUCTURE:
- Page 1: Introduction to the topic (based on document)
- Pages 2-4: ONE key concept per page FROM THE DOCUMENT
- Last page: Summary of document concepts taught

EACH PAGE:
- "title": Short catchy title (3-5 words)
- "content": 1-2 sentences with **bold** key terms FROM DOCUMENT
- "highlight": Key fact FROM DOCUMENT (or null)

Write in {prompt_language}.

Return ONLY valid JSON:
{{
  "title": "Main Lesson Title",
  "pages": [
    {{"title": "👋 Welcome!", "content": "...", "highlight": null}},
    {{"title": "Key Point 1", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 2", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 3", "content": "...", "highlight": "..."}},
    {{"title": "🎯 Summary", "content": "...", "highlight": null}}
  ]
}}"""

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    # Parse response
    content_json = response.content.strip()
    if content_json.startswith("```"):
        content_json = content_json.split("```")[1]
        if content_json.startswith("json"):
            content_json = content_json[4:]
        content_json = content_json.strip()

    return json.loads(content_json)


def _strip_code_fence(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()


def _drain_json_objects(buf: str, cursor: int):
    """Pull every COMPLETE brace-balanced object out of `buf` starting at `cursor`.

    Lets us emit lesson pages the moment each one finishes streaming, without
    waiting for the enclosing JSON document to close. Respects string literals
    and escapes so braces inside content text don't corrupt the depth count.

    Returns (list_of_object_strings, new_cursor). An object still mid-stream is
    left alone — the next call picks it up once more tokens arrive.
    """
    out = []
    i = cursor
    n = len(buf)
    while i < n:
        while i < n and buf[i] != '{':
            i += 1
        if i >= n:
            break
        depth = 0
        in_str = False
        esc = False
        j = i
        closed = False
        while j < n:
            c = buf[j]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        closed = True
                        break
            j += 1
        if not closed:
            break  # incomplete tail — wait for more tokens
        out.append(buf[i:j + 1])
        i = j + 1
        cursor = i
    return out, cursor


async def _stream_lesson_with_context(
    session: PersistentSessionContext,
    topic: str,
    language: str
):
    """Streaming twin of _generate_lesson_with_context.

    WHY: lessons were the only auto-launched node type generated in a single
    blocking call. In production, lesson-first plans persisted node-0 content
    for only 79% of sessions vs 95% for quiz-first — users abandon a dead
    loading screen. Emitting each page as it completes puts readable content on
    screen in seconds instead of ~30.

    Yields dicts:
        {"status": "lesson_title", "title": str}
        {"status": "lesson_page_ready", "page": {...}, "index": int}
        {"status": "lesson_content", "content": {...}}   # terminal, full payload
    """
    context = ""
    if session.vectorstore:
        docs = session.vectorstore.similarity_search(query=topic, k=1000)
        full_text = "\n\n".join([doc.page_content for doc in docs])
        context = full_text[:12000]
        print(f"📚 Lesson context (stream): {len(docs)} chunks, {len(context)} chars")

    if not context:
        print(f"⚠️ WARNING: No document context for lesson on '{topic}'")
        context = "NO DOCUMENT CONTENT AVAILABLE"

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5, streaming=True)
    prompt_language = _language_for_prompt(language)

    prompt = f"""Create a MULTI-PAGE lesson about: {topic}

🚨🚨🚨 CRITICAL INSTRUCTION 🚨🚨🚨
You MUST ONLY use information from the document content below.
DO NOT add any information from your general knowledge.
DO NOT hallucinate or make up facts.
If something is not in the document, DO NOT include it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT'S DOCUMENT CONTENT (USE ONLY THIS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context[:10000]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
1. Extract ONLY facts/concepts that appear in the document above
2. Use the EXACT terminology and definitions from the document
3. Every "content" field must be paraphrased from the document
4. Every "highlight" must quote or summarize document content
5. If the document doesn't cover enough for 5 pages, use fewer pages

STYLE: Duolingo-style swipeable cards. ONE concept per page.

STRUCTURE:
- Page 1: Introduction to the topic (based on document)
- Pages 2-4: ONE key concept per page FROM THE DOCUMENT
- Last page: Summary of document concepts taught

EACH PAGE:
- "title": Short catchy title (3-5 words)
- "content": 1-2 sentences with **bold** key terms FROM DOCUMENT
- "highlight": Key fact FROM DOCUMENT (or null)

Write in {prompt_language}.

IMPORTANT: emit "title" FIRST, then "pages" in order, so the student can begin
reading page 1 while the remaining pages are still being written.

Return ONLY valid JSON:
{{
  "title": "Main Lesson Title",
  "pages": [
    {{"title": "👋 Welcome!", "content": "...", "highlight": null}},
    {{"title": "Key Point 1", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 2", "content": "...", "highlight": "..."}},
    {{"title": "Key Point 3", "content": "...", "highlight": "..."}},
    {{"title": "🎯 Summary", "content": "...", "highlight": null}}
  ]
}}"""

    buf = ""
    cursor = -1          # -1 until the start of the pages array is located
    pages = []
    title = None

    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
        piece = getattr(chunk, "content", "") or ""
        if not piece:
            continue
        buf += piece

        # Lesson title — emit as soon as its string literal closes.
        if title is None:
            m = _lang_re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
            if m:
                try:
                    title = json.loads(f'"{m.group(1)}"')
                except Exception:
                    title = m.group(1)
                yield {"status": "lesson_title", "title": title}

        # Locate `"pages": [` once, then drain complete page objects after it.
        if cursor < 0:
            pm = _lang_re.search(r'"pages"\s*:\s*\[', buf)
            if pm:
                cursor = pm.end()

        if cursor >= 0:
            objs, cursor = _drain_json_objects(buf, cursor)
            for obj_str in objs:
                try:
                    page = json.loads(obj_str)
                except Exception:
                    continue
                pages.append(page)
                yield {
                    "status": "lesson_page_ready",
                    "page": page,
                    "index": len(pages) - 1,
                }

    # Authoritative parse of the whole document; fall back to the pages already
    # emitted if the model closed the JSON badly.
    content = None
    try:
        content = json.loads(_strip_code_fence(buf))
    except Exception as e:
        print(f"⚠️ Lesson stream: full-JSON parse failed ({e}); using streamed pages")
    if not isinstance(content, dict) or not content.get("pages"):
        content = {"title": title or topic, "pages": pages}

    yield {"status": "lesson_content", "content": content}


@app.post("/chat/generate-title", response_model=GenerateTitleResponse)
async def generate_chat_title(request: GenerateTitleRequest):
    # Never return None: an empty message or a failed LLM call (e.g. OpenAI
    # quota/429) must still produce a valid response, otherwise FastAPI raises
    # ResponseValidationError and the client gets a 500.
    fallback_title = "New Chat"

    message = (request.message or "").strip()
    if not message:
        return GenerateTitleResponse(title=fallback_title)

    try:
        prompt = PromptTemplate(
            template="""
            You are an AI assistant for nursing students.

            Based on the following user message, generate a short, descriptive chat title in 3 to 6 words.

            Requirements:
            - Be concise and clear
            - Max 6 words
            - CRITICAL: The title MUST be in the same language as the message
              below. If the message is in English, the title MUST be in
              English — never French, Spanish, Portuguese, or any other
              language. Do not translate.

            Message:
            {message}
            """,
            input_variables=["message"]
        )

        # Use gpt-4.1-nano for title generation (simple classification task).
        # Low temperature: at 0.8 the nano model drifted into random languages
        # (French/Russian titles on English chats).
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4.1-nano",
            streaming=False
        )

        chain = prompt | llm | StrOutputParser()
        generated_title = await asyncio.wait_for(
            chain.ainvoke({"message": message}),
            timeout=15.0
        )

        # Clean up the title
        cleaned_title = generated_title.replace('"', '').replace("'", "").strip()

        return GenerateTitleResponse(title=cleaned_title or fallback_title)

    except Exception as title_error:
        print(f"⚠️ Title generation failed, using fallback: {title_error}")
        return GenerateTitleResponse(title=fallback_title)

# ============================================================================
# REWRITE ENDPOINT — natural-style paraphrase of an AI-generated message
# ============================================================================
@app.post("/chat/rewrite", response_model=RewriteResponse)
async def rewrite_message(request: RewriteRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # 12k-char input cap. The frontend should never send anything close to
    # this; the cap is here so a runaway client can't burn tokens.
    if len(text) > 12000:
        raise HTTPException(status_code=413, detail="text exceeds 12000 chars")

    language = (request.language or "en").split("-")[0].lower()

    system_prompt = """You rewrite text so it reads as if a real student wrote it.

CADENCE
- Mix short and long sentences. Some under 8 words, some over 20.
- Vary sentence openers. Don't start consecutive sentences with the same word.
- Occasional sentence fragments are fine where natural.

VOICE
- Use contractions (it's, you're, doesn't, won't).
- Prefer natural connectors: so, but, though, also, and. Avoid: therefore, however, furthermore, moreover, thus, hence, in addition.
- Write in active voice when possible.

BANNED WORDS AND PHRASES (do not use any of these)
delve, tapestry, navigate, multifaceted, crucial, paramount, robust, leverage, harness, realm, landscape, underscores, emphasizes the importance, plays a vital role, plays a key role, in the realm of, in conclusion, it's important to note, it is worth noting, essentially, ultimately, embark, journey, foster, cultivate, intricate, nuanced, meticulously.

FACTUAL ACCURACY (this is the highest priority)
- Preserve every drug name, dose, route, frequency, lab value, vital sign, and unit verbatim. mg vs mcg vs g matters. Do not round, convert, or paraphrase numbers.
- Preserve every disease name, anatomical term, and clinical procedure verbatim.
- Do not add facts that weren't in the input. Do not remove clinical detail.
- If the input has a list of steps, signs, or symptoms, the rewrite must contain the same items.

LENGTH
- Output length must be within ~15% of input length. Don't pad or strip.

OUTPUT DISCIPLINE
- Output the rewritten text only. No preamble. No commentary. No explanation. No quotation marks wrapping the result.
- Match the language of the input. If the user requests a specific language, output in that language.
- Preserve markdown structure (headings, lists, bold) if the input uses it."""

    user_prompt = (
        f"Rewrite the text below. Output language: {language}.\n\n"
        f"---\n{text}\n---"
    )

    try:
        # gpt-4.1-mini matches the rest of the codebase for instruction-following
        # tasks. temp=0.7 gives enough cadence variety without risking factual
        # drift on clinical content (doses, units, drug names).
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
        chain = llm | StrOutputParser()
        result = await chain.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        rewritten = (result or "").strip()
        # Strip wrapping quotes if the model added them despite instructions.
        if len(rewritten) >= 2 and rewritten[0] in ('"', "'") and rewritten[-1] == rewritten[0]:
            rewritten = rewritten[1:-1].strip()
        if not rewritten:
            raise HTTPException(status_code=502, detail="empty rewrite from model")
        return RewriteResponse(rewritten=rewritten)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Rewrite failed: {e}")
        raise HTTPException(status_code=500, detail="rewrite failed")

# ============================================================================
# GLOSSARY ENDPOINT — clickable medical-term popovers in quiz rationales
# ============================================================================
from pydantic import BaseModel as _GlossaryBaseModel
from services.glossary import get_term_definition
from services.explain import explain_selection
from services.quiz_rationale import generate_rationale as generate_quiz_rationale

class GlossaryRequest(_GlossaryBaseModel):
    term: str

@app.post("/glossary")
async def glossary(request: GlossaryRequest):
    term = (request.term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term is required")
    if len(term) > 120:
        raise HTTPException(status_code=400, detail="term too long")
    try:
        result = await get_term_definition(term)
        return result
    except Exception as e:
        print(f"⚠️ Glossary lookup failed for {term!r}: {e}")
        raise HTTPException(status_code=500, detail="glossary lookup failed")


# ============================================================================
# EXPLAIN ENDPOINT — free-form explanation for user-selected text in the app
# ============================================================================
from typing import Optional as _ExplainOptional

class ExplainRequest(_GlossaryBaseModel):
    text: str
    context: _ExplainOptional[str] = "chat"  # chat | rationale | quiz | flashcard
    language: _ExplainOptional[str] = "en"   # ISO 639-1 (en, fr, …)

@app.post("/explain")
async def explain(request: ExplainRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 800:
        raise HTTPException(status_code=400, detail="text too long")
    try:
        result = await explain_selection(
            text,
            request.context or "chat",
            request.language or "en",
        )
        return result
    except Exception as e:
        print(f"⚠️ Explain failed for {text[:60]!r}: {e}")
        raise HTTPException(status_code=500, detail="explain failed")


# ============================================================================
# QUIZ RATIONALE ENDPOINT — generate per-option rationale on demand when the
# user clicks "Learn more" on a quiz question. Lets the original quiz
# generation skip the (expensive) inline justification.
# ============================================================================
from typing import List as _RationaleList

class QuizRationaleRequest(_GlossaryBaseModel):
    question: str
    options: _RationaleList[str]
    correct_index: int
    language: _ExplainOptional[str] = "en"

class QuizExtendRequest(_GlossaryBaseModel):
    """Ask for more questions on a quiz the student is already working through."""
    chat_id: str
    topic: str
    count: int = 5
    difficulty: _ExplainOptional[str] = "medium"
    question_types: _ExplainOptional[_RationaleList[str]] = None
    quiz_mode: _ExplainOptional[str] = "knowledge"
    learning_objective: _ExplainOptional[str] = "general"
    language: _ExplainOptional[str] = "en"
    # Question text already on screen. Drives both concept-avoidance and the
    # index the new questions are numbered from.
    existing_questions: _ExplainOptional[_RationaleList[str]] = None


@app.post("/quiz/extend-stream")
async def extend_quiz_stream(request: QuizExtendRequest):
    """
    Generate the NEXT batch of questions for an in-progress chat quiz.

    WHY THIS EXISTS

    stream_quiz_questions fires one LLM call per question, all in parallel, the
    moment a quiz is requested. A 15-question quiz therefore costs 15 questions
    the instant it is asked for — and measured over 776 real 15-question
    quizzes, 21.5% were never answered at all and only 40.3% were finished. The
    in-loop cancellation check cannot recover any of that, because the calls are
    already in flight before it runs.

    So the quiz now starts short and grows on demand: the client asks for more
    as the student approaches the end of what it has. Questions nobody reaches
    are never generated, and because the client prefetches a batch ahead, the
    student never waits for one.

    Streams NDJSON, matching the existing quiz stream so the frontend's chunk
    handling is reused as-is.
    """
    async def stream_generator():
        try:
            quota = usage_guard.check_quota(request.chat_id)
            if not quota["allowed"]:
                yield json.dumps({
                    "status": "error",
                    "code": "quota_exceeded",
                    "message": usage_guard.QUOTA_MESSAGE
                }) + "\n"
                return

            session = await _setup_study_session(request.chat_id, request.language)
            source = "documents" if session.documents and session.vectorstore else "scratch"

            existing = request.existing_questions or []
            # Clamped: this runs unattended from a client timer, so a bad or
            # hostile count must not translate into an unbounded fan-out of
            # parallel LLM calls.
            count = max(1, min(10, request.count or 5))

            async for chunk in stream_quiz_with_bank(
                topic=request.topic,
                difficulty=request.difficulty or "medium",
                num_questions=count,
                source=source,
                session=session,
                chat_id=request.chat_id,
                question_types=request.question_types or ["mcq"],
                quiz_mode=request.quiz_mode or "knowledge",
                learning_objective=request.learning_objective or "general",
                existing_questions=existing,
                index_offset=len(existing),
            ):
                yield json.dumps(chunk) + "\n"

        except Exception as e:
            print(f"❌ Quiz extend failed: {e}")
            # Same error shape the chat stream uses, so the client's existing
            # error handling covers this without a special case.
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.post("/quiz_rationale")
async def quiz_rationale(request: QuizRationaleRequest):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if not request.options or len(request.options) < 2:
        raise HTTPException(status_code=400, detail="at least 2 options required")
    if request.correct_index < 0 or request.correct_index >= len(request.options):
        raise HTTPException(status_code=400, detail="correct_index out of range")
    if len(question) > 1200:
        raise HTTPException(status_code=400, detail="question too long")
    try:
        result = await generate_quiz_rationale(
            question,
            request.options,
            request.correct_index,
            request.language or "en",
        )
        return result
    except Exception as e:
        print(f"⚠️ Quiz rationale failed for {question[:60]!r}: {e}")
        raise HTTPException(status_code=500, detail="quiz rationale failed")

# ============================================================================
# QUESTION BANK IMPORT ENDPOINT
# ============================================================================
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QuestionMetadata(BaseModel):
    """Metadata object matching LLM output format."""
    sourceLanguage: str = "en"
    topic: str = ""
    category: str = "nursing"
    difficulty: str = "medium"
    correctAnswerIndex: int = 0
    sourceDocument: str = "admin_import"
    keywords: List[str] = []

class QuestionImport(BaseModel):
    """
    Single question for import.
    Matches the EXACT format your LLM generates.

    Backwards-compat: `justification` and `correct_blurb` are both optional.
    Legacy generations shipped a full per-option `justification` HTML blob;
    the new pipeline ships a one-sentence `correct_blurb` and defers the
    full per-option rationale to /quiz_rationale on demand. Admin imports
    can use either field.
    """
    question: str
    options: List[str]  # ["A) ...", "B) ...", "C) ...", "D) ..."]
    answer: str         # The correct option text (e.g., "B) ...")
    justification: Optional[str] = ""   # Legacy full-rationale HTML
    correct_blurb: Optional[str] = ""   # New one-sentence rationale
    topic: str          # e.g., "cardiac medications"
    metadata: Optional[QuestionMetadata] = None  # Optional - matches LLM output

class BulkImportRequest(BaseModel):
    """Request body for bulk question import."""
    questions: List[QuestionImport]

@app.post("/admin/import-questions")
async def import_questions(request: BulkImportRequest):
    """
    Import AI-generated questions into the Question Bank.

    Accepts questions in the EXACT format your LLM generates.
    The metadata object is optional but will be used if provided.

    Example request body (matches LLM output):
    {
        "questions": [
            {
                "question": "A patient receiving digoxin reports nausea...",
                "options": [
                    "A) Continue the medication as prescribed",
                    "B) Hold the medication and notify the provider",
                    "C) Administer an antiemetic",
                    "D) Document the findings and reassess later"
                ],
                "answer": "B) Hold the medication and notify the provider",
                "justification": "<strong>Option B is correct</strong> because...",
                "topic": "Cardiac Medications",
                "metadata": {
                    "sourceLanguage": "en",
                    "topic": "Cardiac Medications",
                    "category": "nursing",
                    "difficulty": "medium",
                    "correctAnswerIndex": 1,
                    "sourceDocument": "conversational_generation",
                    "keywords": ["digoxin", "toxicity", "cardiac"]
                }
            }
        ]
    }
    """
    from services.question_bank import question_bank

    results = {
        "total": len(request.questions),
        "imported": 0,
        "duplicates": 0,
        "errors": [],
        "imported_ids": []
    }

    for i, q in enumerate(request.questions):
        try:
            # Extract language and difficulty from metadata if available
            language = "en"
            difficulty = "medium"

            if q.metadata:
                language = q.metadata.sourceLanguage or "en"
                difficulty = q.metadata.difficulty or "medium"

            # Normalize language code
            if language.lower().startswith("fr"):
                language = "fr"
            elif language.lower().startswith("es"):
                language = "es"
            else:
                language = "en"

            # The question_data format expected by save_question.
            # Both legacy (justification) and new (correct_blurb) fields are
            # forwarded so existing question-bank rows keep their full HTML
            # while new ones store the short blurb.
            question_data = {
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "justification": q.justification or "",
                "correct_blurb": q.correct_blurb or "",
                "topic": q.topic
            }

            # Save to question bank
            doc_id = await question_bank.save_question(
                question_data=question_data,
                topic=q.topic,
                language=language,
                difficulty=difficulty,
                chat_id="admin_import"  # Mark as admin import
            )

            if doc_id:
                results["imported"] += 1
                results["imported_ids"].append(doc_id)
                print(f"✅ Imported question {i+1}: {doc_id}")
            else:
                results["duplicates"] += 1
                print(f"⚠️ Question {i+1} skipped (duplicate)")

        except Exception as e:
            error_msg = f"Question {i+1}: {str(e)}"
            results["errors"].append(error_msg)
            print(f"❌ Error importing question {i+1}: {e}")

    print(f"📊 Import complete: {results['imported']} imported, {results['duplicates']} duplicates, {len(results['errors'])} errors")

    return results


@app.get("/admin/question-bank-stats")
async def get_question_bank_stats():
    """
    Get statistics about the Question Bank.

    Returns total questions, breakdown by language and category.
    """
    from services.question_bank import question_bank

    stats = await question_bank.get_bank_stats()
    return stats


# ============================================================================
# SPEECH-TO-TEXT (Voice Input)
# ============================================================================
from openai import OpenAI

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Transcribe audio to text using OpenAI Whisper API.

    Accepts audio files (webm, mp3, wav, m4a, etc.)
    Returns the transcribed text that can be used as a chat prompt.

    Cost: ~$0.006 per minute of audio
    """
    try:
        # Validate file type
        content_type = audio.content_type or ""
        if not any(t in content_type for t in ["audio", "video/webm"]):
            # Also check file extension as fallback
            ext = audio.filename.split(".")[-1].lower() if audio.filename else ""
            if ext not in ["webm", "mp3", "wav", "m4a", "ogg", "mp4"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported audio format: {content_type}. Use webm, mp3, wav, m4a, or ogg."
                )

        # Read the audio file
        audio_bytes = await audio.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Max 25MB (Whisper API limit)
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")

        # Create a temp file for the audio
        ext = audio.filename.split(".")[-1] if audio.filename else "webm"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Call OpenAI Whisper API
            client = OpenAI()

            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

            # Clean up the transcription
            transcribed_text = transcript.strip() if isinstance(transcript, str) else transcript

            print(f"🎤 [STT] Transcribed {len(audio_bytes)} bytes -> '{transcribed_text[:100]}...'")

            return {
                "success": True,
                "text": transcribed_text,
                "audio_size_bytes": len(audio_bytes)
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [STT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ============================================================================
# FILE PROXY — serves chat uploads with a clean URL + inline disposition
# so embedded viewers (Office Online for .docx/.pptx/.xlsx, browser PDF
# viewer, etc.) can fetch them reliably. Firebase Storage download URLs
# with `?alt=media&token=...` fail in third-party viewers because of the
# query string and Content-Disposition headers Firebase sets.
# ============================================================================
from fastapi.responses import Response
import mimetypes as _mimetypes


@app.get("/files/proxy/{chat_id}/{filename}")
async def files_proxy(chat_id: str, filename: str):
    """
    Stream a chat-attached file from Firebase Storage.

    Security note: this endpoint is currently unauthenticated. The chat_id is
    a UUID so unguessable in practice, but a leaked URL grants read access to
    the file. Add auth (verify Firebase ID token + chat ownership) before
    treating any of this as private.
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"chats/{chat_id}/uploads/{filename}")

        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Most class files are <50MB; download into memory.
        content = blob.download_as_bytes()

        # Prefer the blob's stored content-type, fall back to the extension.
        content_type = blob.content_type
        if not content_type or content_type == "application/octet-stream":
            guessed, _ = _mimetypes.guess_type(filename)
            content_type = guessed or content_type or "application/octet-stream"

        # `inline` so viewers render rather than trigger a download.
        # Quote the filename in case it contains spaces.
        safe_name = filename.replace('"', "")
        headers = {
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Cache-Control": "private, max-age=3600",
            "Content-Length": str(len(content)),
        }
        return Response(content=content, media_type=content_type, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [Files] Proxy failed for chats/{chat_id}/uploads/{filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy failed: {e}")


# ============================================================================
# FILE PROXY
# Streams chat-attached files with proper Content-Type so that 3rd-party
# viewers (e.g., Microsoft Office Online) can render them. Firebase Storage
# download URLs don't always play well with Office viewer — this proxy
# normalizes the response.
# ============================================================================
_OFFICE_MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pdf": "application/pdf",
    "txt": "text/plain; charset=utf-8",
    "md": "text/plain; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
}


@app.get("/files/proxy/{chat_id}/{filename:path}")
async def files_proxy(chat_id: str, filename: str):
    """
    Stream a file from chats/{chat_id}/uploads/{filename} with a clean
    Content-Type and inline disposition. The URL ends in the file's real
    extension, which Office Online's viewer needs to recognize the format.
    """
    if not chat_id or not filename:
        raise HTTPException(status_code=400, detail="Missing chat_id or filename")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if ".." in chat_id or "/" in chat_id:
        raise HTTPException(status_code=400, detail="Invalid chat_id")

    storage_path = f"chats/{chat_id}/uploads/{filename}"
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {storage_path}")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        content_type = _OFFICE_MIME_TYPES.get(ext) or blob.content_type or "application/octet-stream"

        data = blob.download_as_bytes()

        return Response(
            content=data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "public, max-age=300",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [Files proxy] failed for {storage_path}: {e}")
        raise HTTPException(status_code=500, detail=f"File proxy failed: {e}")


# ============================================================================
# HEALTH CHECK
# ============================================================================
@app.post("/warm_up")
async def warm_up():
    return {"status": "ok", "message": "Server warmed up successfully"}

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment (Cloud Run sets this to 8080)
    port = int(os.getenv("PORT", 8080))
    
    print(f"Starting server on port {port}...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # CRITICAL: Must be 0.0.0.0, not 127.0.0.1
        port=port,
        reload=False,  # CRITICAL: No reload in production
    )
