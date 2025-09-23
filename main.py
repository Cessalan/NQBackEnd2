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
from models.requests import StatelessChatRequest, DocumentsEmbedRequest

# Import your orchestrator
from services.orchestrator import NursingTutor

# Import Firebase initialization
import firebase_admin
from firebase_admin import credentials

# embedding
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

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

from core.imageloader import OCRImageLoader

# langchain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# firebase access
import firebase_admin
from firebase_admin import credentials,storage

from core.language import LanguageDetector

# to load documents
def get_loader_for_file(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf": # pdf file support
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
        return UnstructuredPowerPointLoader(path)
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_API_KEY")

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
    print(f"nursing tutor created")
    
    # Update session with documents if provided
    if request.documents:
        nursing_tutor.session.documents = request.documents
    
    
    user_language = LanguageDetector.detect_language(request.input)
    
    # Process message and stream response
    return StreamingResponse(
        nursing_tutor.process_message(
            # feed the tutor the user input
            user_input=request.input,
            # feed the tutor the chat history
            chat_history=request.chat_history,
            # feed the tutor the language the user's browser
            language=user_language
        ),
        media_type="application/json"
    )

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

@app.post("/chat/embed")
def embed_documents(request:DocumentsEmbedRequest): 
    try:
        if request.documents:
            all_chunks = []
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            
            for doc in request.documents:
                response = requests.get(doc.source)
                print("fetch firebase response", response)
                suffix = os.path.splitext(doc.filename)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(response.content)
                    temp_path = f.name
                try:
                    loader = get_loader_for_file(temp_path)  
                    print(f"✅ Loader created for {doc.filename}: {type(loader)}")
                    
                    pages = loader.load()                   
                    print(f"✅ Loaded {len(pages)} pages from {doc.filename}")
                    
                    
                    text = "\n\n".join([p.page_content for p in pages])
                    print(f"✅ Extracted {len(text)} characters from {doc.filename}")
                    
                    # get the word count to estimate reading time in front-end
                    word_count = 0
                    word_count = len(text.split())
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                       all_chunks.append(type("Document", (), {"page_content": chunk,"metadata": {"source": doc.filename}})())

                        #this is how langchain expects to receive metadata
                        #all_chunks.append(Document(page_content=chunk, metadata={"source": doc.filename}))
                except Exception as e:
                    print(f"❌ Failed to process {doc.filename}: {e}")
                    print(f"File extension: {os.path.splitext(doc.filename)[-1].lower()}")
                    continue  # Skip this file and continue with others
                finally:
                    os.unlink(temp_path)
                    with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tempdir:
                        vectorstore = FAISS.from_documents(all_chunks, embedding=OpenAIEmbeddings())
                        vectorstore.save_local(tempdir)

                        # Upload both files to Firebase
                        bucket = storage.bucket()
                                        
                        # upload it to the file vectorstore specific knowledge for a file (quiz, summary, mise en situation)
                        blob_faiss_file = bucket.blob(f"FileVectorStore/{request.chatId}/{doc.filename}/index.faiss")
                        blob_faiss_file.upload_from_filename(os.path.join(tempdir, "index.faiss"))
                        
                        blob_pkl_file = bucket.blob(f"FileVectorStore/{request.chatId}/{doc.filename}/index.pkl")
                        blob_pkl_file.upload_from_filename(os.path.join(tempdir, "index.pkl"))
                            
            # Save vector store locally to /tmp
            with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tempdir:
                vectorstore = FAISS.from_documents(all_chunks, embedding=OpenAIEmbeddings())
                vectorstore.save_local(tempdir)

                # Upload both files to Firebase
                bucket = storage.bucket()
                
                # upload it in the chat vector store (general knowledge for the chat)
                blob_faiss_chat = bucket.blob(f"vectorstores/{request.chatId}/index.faiss")
                blob_faiss_chat.upload_from_filename(os.path.join(tempdir, "index.faiss"))

                blob_pkl_chat = bucket.blob(f"vectorstores/{request.chatId}/index.pkl")
                blob_pkl_chat.upload_from_filename(os.path.join(tempdir, "index.pkl"))
                
            return {
                "status": "success",
                "word-count":word_count,
                "firebase_path": f"vectorstores/{request.chatId}/"
            }
        return {"status": "no_documents"}
    except Exception as e:
        print("🔥🔥🔥 ERROR:", e)
        return {"status": "error", "message": str(e)}

@app.post("/chat/generate-quiz")
async def generate_quiz(request):
    """Your existing quiz generation"""
    # Keep your existing implementation
    pass

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
            model_name="gpt-4o-mini",
            streaming=False
        )

        chain = prompt | llm | StrOutputParser()
        generated_title = await chain.ainvoke({"message": request.message})
        
        # Clean up the title
        cleaned_title = generated_title.replace('"', '').replace("'", "").strip()

        return GenerateTitleResponse(title=cleaned_title)

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
@app.post("/warm_up")
async def warm_up():
    return {"status": "ok", "message": "Server warmed up successfully"}

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
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )