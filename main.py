# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import os
from dotenv import load_dotenv

import asyncio
import json
# Load environment variables
load_dotenv()

# Import your models
from models.requests import StatelessChatRequest, DocumentsEmbedRequest, SummaryRequest,SectionRequest,PlanRequest

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
from models.reponses import GenerateTitleResponse
from models.requests import GenerateTitleRequest

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

import json

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
        return TextLoader(path)
    elif ext == ".csv":  #excel support
        return CSVLoader(path)
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
cred = credentials.Certificate("FireBaseAccess.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": os.getenv("FIREBASE_BUCKET", "docai-efb03.firebasestorage.app")
})

# cred = credentials.Certificate("service-account-key.json")  # Path to key file
#     firebase_admin.initialize_app(cred, {
#         'storageBucket': 'docai-efb03.firebasestorage.app'
# })


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_API_KEY")
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
        "https://chats.nursequizai.com",
        "https://ragfastapi-1075876064685.europe-west1.run.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session storage
ACTIVE_SESSIONS: Dict[str, NursingTutor] = {}

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.cancellation_flags: Dict[str, bool] = {}  # Track if chat should be cancelled

    async def connect(self, websocket: WebSocket, chat_id: str):
        await websocket.accept()
        self.active_connections[chat_id] = websocket
        self.cancellation_flags[chat_id] = False  # Reset cancellation flag
        print(f"✅ WebSocket connected for chat_id: {chat_id}")

    def disconnect(self, chat_id: str):
        if chat_id in self.active_connections:
            del self.active_connections[chat_id]
            print(f"❌ WebSocket disconnected for chat_id: {chat_id}")
        if chat_id in self.cancellation_flags:
            del self.cancellation_flags[chat_id]

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

# Global connection manager
manager = ConnectionManager()

# Set the manager reference in quiztools for cancellation checks
from tools.quiztools import set_connection_manager, get_chat_context_from_db
set_connection_manager(manager)

# WebSocket endpoint
@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    #connect
    await manager.connect(websocket, chat_id) 
    try:
        while True:
            # Wait for incoming message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
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

    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))
               
async def process_chat_message(chat_id: str, message: dict, websocket: WebSocket):
    """Process chat messages through the existing NursingTutor"""
    try:
        # Extract message data
        user_input = message.get("input", "")

        # Get or create session (same as existing logic)
        if chat_id not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)
            # Still reload insights in case new files were uploaded
            await ACTIVE_SESSIONS[chat_id].load_file_insights_from_firebase()

        nursing_tutor = ACTIVE_SESSIONS[chat_id]

        # Get chat history for context-aware language detection
        full_context_from_db = await get_chat_context_from_db(chat_id)
        chat_history = full_context_from_db.get("conversation", [])[-10:] if full_context_from_db else []

        # Detect language with chat context
        language = await LanguageDetector.detect_language(user_input, chat_history)
        print(f"Input was entered in language: {language}")
        
        # Send status update
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "processing",
            "message": "Processing your message..."
        }))
        
        # Process message and stream responses
        async for chunk in nursing_tutor.process_message(user_input, language):
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
        
        # Send completion signal
        manager.reset_cancellation(chat_id)  # Reset cancellation flag
        await websocket.send_text(json.dumps({
            "type": "stream_complete",
            "message": "Response complete"
        }))
        
    except Exception as e:
        print(f"Error processing chat message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Processing failed: {str(e)}"
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
    print(F"*Upload received: user{ user_id}, chat:{chat_id}, language:{language}*")
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
            
            async def process_single_file_data(file_data):
                async with semaphore:
                    return await process_file_from_bytes(
                        file_data["bytes"],
                        file_data["filename"],
                        chat_id,
                        user_id,
                        language  # Pass browser language
                    )
            
            # Start all file processing tasks
            tasks = [process_single_file_data(fd) for fd in valid_files]
            
            # Stream results as they complete
            total_words = 0
            completed_files = []
            file_documents = {}  # filename -> documents
            file_bytes_map = {}  # filename -> bytes (for background upload)
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    
                    # Extract updates, documents, insights, and file bytes
                    updates = result.get("updates", [])
                    documents = result.get("documents", [])
                    insights = result.get("insights")  # ← NEW
                    filename = result.get("filename", "unknown")
                    file_bytes = result.get("file_bytes")
                    
                    print(f"🔍 DEBUG: insights type = {type(insights)}")  # ← ADD THIS
                    print(f"🔍 DEBUG: insights value = {insights}") 
                    
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
                    # 🆕 STORE INSIGHTS IN SESSION
                    # ========================================
                    if insights and chat_id in ACTIVE_SESSIONS:
                        session = ACTIVE_SESSIONS[chat_id]
                        #update the session language using the front-end browser language when uploading
                        session.session.user_language = language 
                        if not hasattr(session.session, "file_insights"):
                            session.session.file_insights = {}
                        session.session.file_insights[filename] = insights
                    
                    # Stream JSON updates to frontend
                    for update in updates:
                        if not isinstance(update, dict):
                            continue
                        
                        yield json.dumps(update) + "\n"
                        
                        if update.get("type") == "file_complete":
                            total_words += update.get("word_count", 0)
                            completed_files.append(update.get("file_id"))
                
                except Exception as e:
                    print(f"❌ Task error: {e}")
                    import traceback
                    traceback.print_exc()
                    yield json.dumps({
                        "type": "file_error",
                        "message": str(e)
                    }) + "\n"
                
            
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
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
                        
                        
                        summary_prompt = f"""Generate a brief 1-2 sentence summary about what these uploaded documents contain.

                        Files: {len(file_insights)} document(s)
                        Topics found: {', '.join(unique_topics)}
                        Document types: {', '.join(unique_doc_types) if unique_doc_types else 'various'}

                        Write a natural, conversational summary that tells the student what content was found.
                        Examples:
                        - "I found materials about cardiac pharmacology and arrhythmia management."
                        - "Les documents sur la pharmacologie cardiaque et la gestion des arythmies."

                        IMPORTANT REQUIREMENT: The entire summary must be written in {language}.

                        Return ONLY the summary text in {language}, nothing else."""
                        
                        summary_response = await llm.ainvoke([
                            {"role": "user", "content": summary_prompt}
                        ])
                        
                        upload_summary = summary_response.content.strip()
                        
                        # Yield summary to frontend
                        yield json.dumps({
                            "type": "upload_summary",
                            "summary": upload_summary,
                            "file_count": len(valid_files),
                            "filenames": [f["filename"] for f in valid_files]
                        }) + "\n"
                        
                        print(f"📝 Generated upload summary: {upload_summary}")
                        
                        
                        title_prompt = f"""Generate a short, descriptive chat title in 3 to 6 words based on these uploaded study materials.

                       
                        Topics Found: {', '.join(unique_topics) if unique_topics else 'medical content'}
                        Document Types: {', '.join(unique_doc_types) if unique_doc_types else 'study materials'}

                        Requirements:
                        - Be concise and clear
                        - Focus on the main topic/subject area
                        - Max 6 words
                        - Make it specific to the content (e.g., "Cardiac Pharmacology Notes", "NCLEX Respiratory Review")
                        - Write in {language}

                        Return ONLY the title, no quotes or extra text."""
                        
                        title_response = await llm.ainvoke([
                            {"role": "user", "content": title_prompt}
                        ])
                        
                        chat_title = title_response.content.strip().replace('"', '').replace("'", "")

                        # Update Firebase chat document with new title
                        from firebase_admin import firestore
                        db = firestore.client()

                        db.collection('chats').document(chat_id).update({
                            'title': chat_title,
                            'updatedAt': firestore.SERVER_TIMESTAMP
                        })

                        print(f"✅ Auto-generated and saved chat title: '{chat_title}'")
                        
                    except Exception as summary_error:
                        print(f"⚠️ Summary generation failed: {summary_error}")
            
            # ========================================
            # Build Suggestions after user uploads
            # ========================================

            if chat_id in ACTIVE_SESSIONS:
                session = ACTIVE_SESSIONS[chat_id]
                file_insights = getattr(session.session, "file_insights", {})
                
                print(f"🔍 DEBUG: file_insights keys = {list(file_insights.keys())}")
                print(f"🔍 DEBUG: file_insights content = {file_insights}")
                
                if file_insights:
                    print(f"📝 Generating post-upload suggestions for {len(file_insights)} files...")
                    try:
                        from services.orchestrator import generate_post_upload_suggestions
                        suggestions = await generate_post_upload_suggestions(
                            session=session,
                            file_insights=file_insights
                        )
                        
                        if suggestions:
                            # 🆕 SAVE DIRECTLY TO FIREBASE (instead of just yielding)
                            try:
                                from firebase_admin import firestore
                                db = firestore.client()
                                
                                suggestion_message = {
                                    "role": "assistant",
                                    "type": "suggested_prompts",
                                    "suggestions": suggestions,
                                    "timestamp": firestore.SERVER_TIMESTAMP,
                                    "isStreaming": False
                                }
                                
                                # Save to Firestore
                                db.collection('chats').document(chat_id).collection('messages').add(suggestion_message)
                                print(f"✅ Saved {len(suggestions)} suggestions to Firebase")
                                
                            except Exception as firebase_error:
                                print(f"⚠️ Firebase save failed: {firebase_error}")
                                # Still yield for fallback
                                yield json.dumps({
                                    "type": "suggested_prompts",
                                    "suggestions": suggestions
                                }) + "\n"
                            
                        else:
                            print(f"chatID :{chat_id} FOUND, insights:{file_insights}, but no suggestions")
                    except Exception as e:
                        print(f"⚠️ Suggestion generation failed: {e}")
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
        media_type="application/x-ndjson"
    )

# ============================================================================
# HELPERS FOR FILE UPLOAD START
# ============================================================================
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
        
        # Use GPT-4o-mini for fast, cheap analysis                
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Determine response language        
        prompt = f"""Analyze this nursing/medical document excerpt and extract key information in {language}.

        Document: {filename}
        Content sample:
        {sample_text[:2000]}

        Identify:
        1. Main topics (3-5 broad themes)
        2. Specific concepts (5-10 specific medical/nursing terms or procedures)
        3. Document type (textbook, lecture notes, clinical guide, reference, etc.)

        Return ONLY valid JSON with content in this language {language}:
        {{
        "topics": ["topic1", "topic2", "topic3"],
        "concepts": ["concept1", "concept2", ...],
        "document_type": "type"
        }}
        """
        
        response = await llm.ainvoke([
            {"role": "system", "content": f"You extract key information from medical documents. Return only valid JSON with all content in {language}."},
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
                "document_type": default_type
            }
        
        print(f"✅ Extracted insights from {filename}:")
        print(f"   Topics: {insights.get('topics', [])}")
        print(f"   Concepts: {insights.get('concepts', [])[:3]}...")
        
        # Stream insight batch to frontend
        updates.append({
            "type": "insight_batch",
            "file_id": file_id,
            "filename": filename,
            "topics": insights.get("topics", []),
            "concepts": insights.get("concepts", [])[:5],  # Limit to 5 for UX
            "document_type": insights.get("document_type", "")
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

async def process_file_from_bytes(file_bytes: bytes, filename: str, chat_id: str, user_id: str, language: str = "english"):
    """Process a file from bytes and return list of progress updates."""
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
        
        # Load document
        loader = get_loader_for_file(temp_path)
        pages = loader.load()
        
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
        
        # Get embeddings
        embeddings = OpenAIEmbeddings()
        
        # Get session
        if chat_id not in ACTIVE_SESSIONS:
            print(f"⚠️ No session found for {chat_id}, creating...")
            ACTIVE_SESSIONS[chat_id] = NursingTutor(chat_id)
        
        session = ACTIVE_SESSIONS[chat_id]
        
        # Add to combined vectorstore
        if session.session.vectorstore:
            print(f"➕ Adding to existing vectorstore")
            session.session.vectorstore.add_documents(documents)
        else:
            print(f"🆕 Creating new vectorstore")
            session.session.vectorstore = FAISS.from_documents(documents, embeddings)
        
        print(f"✅ Embedding complete for {filename}")
        
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
    
    # === FIX FOR ATTRIBUTE ERROR ===
    # The NursingTutor object seems to be missing the chat_id attribute internally.
    # We enforce its existence here before passing it to set_session_context.
    if not hasattr(nursing_tutor, 'chat_id'):
        nursing_tutor.chat_id = request.chat_id
        
    set_session_context(nursing_tutor)
    
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
        # This creates an LLM instance
        llm = ChatOpenAI(
            model="gpt-4o-mini",           # Which AI model to use
            temperature=0.3          # How creative (0=focused, 1=creative)
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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)    
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
        
@app.post("/chat/generate-title", response_model=GenerateTitleResponse)
async def generate_chat_title(request: GenerateTitleRequest):
    
    if request.message:
        prompt = PromptTemplate(
            template="""
            You are an AI assistant for nursing students.

            Based on the following user message, generate a short, descriptive chat title in 3 to 6 words.

            Requirements:
            - Be concise and clear
            - Write a title in the language of Message you received
            - Max 6 words

            Message:
            {message}
            """,
            input_variables=["message"]
        )
        
       
        llm = ChatOpenAI(
            temperature=0.8,
            model="gpt-4o-mini",
            streaming=False
        )

        chain = prompt | llm | StrOutputParser()
        generated_title = await chain.ainvoke({"message": request.message})
        
        # Clean up the title
        cleaned_title = generated_title.replace('"', '').replace("'", "").strip()

        return GenerateTitleResponse(title=cleaned_title)

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