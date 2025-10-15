# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import os
from dotenv import load_dotenv

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

from core.imageloader import OCRImageLoader

# langchain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# firebase access
import firebase_admin
from firebase_admin import credentials,storage

from core.language import LanguageDetector

import json

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
    
    #not using this anymore fetching directly from firebase
    # print("HIIIIISTORY", request.chat_history)
    user_language = LanguageDetector.detect_language(request.input)
    
    # Process message and stream response
    return StreamingResponse(
        nursing_tutor.process_message(
            # feed the tutor the user input
            user_input=request.input,
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
def embed_documents(request: DocumentsEmbedRequest): 
    try:
        if request.documents:
            all_chunks = []
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            word_count = 0
            
            for doc in request.documents:
                print(f"📄 Processing: {doc.filename}")
                
                response = requests.get(doc.source)
                print(f"✅ Downloaded {doc.filename}")
                
                suffix = os.path.splitext(doc.filename)[-1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(response.content)
                    temp_path = f.name
                
                try:
                    loader = get_loader_for_file(temp_path)  
                    print(f"✅ Loader created for {doc.filename}")
                    
                    pages = loader.load()                   
                    print(f"✅ Loaded {len(pages)} pages from {doc.filename}")
                    
                    text = "\n\n".join([p.page_content for p in pages])
                    print(f"✅ Extracted {len(text)} characters from {doc.filename}")
                    
                    word_count += len(text.split())
                    chunks = text_splitter.split_text(text)
                    
                    file_chunks = []
                    
                    for chunk in chunks:
                        # ✅ FIXED: Use proper Document class
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata={"source": doc.filename}
                        )
                        
                        file_chunks.append(chunk_doc)
                        all_chunks.append(chunk_doc)
                    
                    print(f"✅ Created {len(file_chunks)} chunks for {doc.filename}")
                    
                    # Upload file-specific vectorstore
                    with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tempdir:
                        print(f"📤 Creating file-specific vectorstore for {doc.filename}")
                        
                        file_vectorstore = FAISS.from_documents(
                            file_chunks,
                            embedding=OpenAIEmbeddings()
                        )
                        file_vectorstore.save_local(tempdir)
                        
                        bucket = storage.bucket()
                        
                        blob_faiss_file = bucket.blob(
                            f"FileVectorStore/{request.chatId}/{doc.filename}/index.faiss"
                        )
                        blob_faiss_file.upload_from_filename(
                            os.path.join(tempdir, "index.faiss")
                        )
                        
                        blob_pkl_file = bucket.blob(
                            f"FileVectorStore/{request.chatId}/{doc.filename}/index.pkl"
                        )
                        blob_pkl_file.upload_from_filename(
                            os.path.join(tempdir, "index.pkl")
                        )
                        
                        print(f"✅ Uploaded file-specific vectorstore for {doc.filename}")
                    
                except Exception as e:
                    print(f"❌ Failed to process {doc.filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
                finally:
                    try:
                        os.unlink(temp_path)
                        print(f"🗑️ Cleaned up temp file for {doc.filename}")
                    except Exception as cleanup_error:
                        print(f"⚠️ Cleanup failed: {cleanup_error}")
            
            # Chat-level vectorstore
            print(f"📤 Creating chat-level vectorstore with {len(all_chunks)} total chunks")
            
            with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tempdir:
                chat_vectorstore = FAISS.from_documents(
                    all_chunks,
                    embedding=OpenAIEmbeddings()
                )
                chat_vectorstore.save_local(tempdir)

                bucket = storage.bucket()
                
                blob_faiss_chat = bucket.blob(f"vectorstores/{request.chatId}/index.faiss")
                blob_faiss_chat.upload_from_filename(os.path.join(tempdir, "index.faiss"))

                blob_pkl_chat = bucket.blob(f"vectorstores/{request.chatId}/index.pkl")
                blob_pkl_chat.upload_from_filename(os.path.join(tempdir, "index.pkl"))
                
                print(f"✅ Uploaded chat-level vectorstore")
            
            print(f"✅✅✅ EMBED COMPLETE - Total words: {word_count}")
            
            return {
                "status": "success",
                "word-count": word_count,
                "firebase_path": f"vectorstores/{request.chatId}/"
            }
            
        return {"status": "no_documents"}
        
    except Exception as e:
        print(f"🔥🔥🔥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

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