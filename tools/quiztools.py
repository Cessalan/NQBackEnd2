from typing import Dict, Any, Optional,List
from datetime import datetime
from langchain_core.tools import tool
from models.session import PersistentSessionContext
from langchain_openai import ChatOpenAI
# Manual prompt variable injection, including memory if used (ideal for custom stuff)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from fastapi import HTTPException
from typing import AsyncGenerator
import os, tempfile
import json
import random

#to format llm response into string
from langchain_core.output_parsers import StrOutputParser

# firebase that stores data needed for context (conversation)
from firebase_admin import firestore

# firebase that stores the vector store files
from firebase_admin import storage

# Global reference to connection manager (will be set by main.py)
_CONNECTION_MANAGER = None

def set_connection_manager(manager):
    """Set the global connection manager reference"""
    global _CONNECTION_MANAGER
    _CONNECTION_MANAGER = manager

def get_connection_manager():
    """Get the global connection manager"""
    return _CONNECTION_MANAGER


def get_firestore_client():
    return firestore.client()

# Global session access - this will be injected by NursingTutor
_CURRENT_SESSION: Optional[PersistentSessionContext] = None

    
def set_session_context(session: PersistentSessionContext):
    """Set the current session context for tool access"""
    global _CURRENT_SESSION
    session.vectorstore = get_chat_vectorstore(session.chat_id)
    session.documents = load_files_for_chat(session.chat_id)
    _CURRENT_SESSION = session

def get_session() -> PersistentSessionContext:
    """Get current session context"""
    if _CURRENT_SESSION is None:
        raise RuntimeError("No session context available. This is a system error.")
    return _CURRENT_SESSION


def _get_gemini_model(model_env_var: str = "GEMINI_MODEL", default_model: str = "gemini-2.5-flash"):
    """Configure and return a Gemini model if GOOGLE_API_KEY is set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = os.getenv(model_env_var, default_model)
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": "block_none"},
        ]
        return genai.GenerativeModel(model_name=model_name, safety_settings=safety_settings)
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        return None


def load_files_for_chat(chat_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves a list of all files uploaded for a specific chat from Firebase Storage,
    including their name, path, and a temporary download URL.

    This is the Python equivalent of the client-side JavaScript 'loadFilesForChat'
    using the Firebase Admin SDK to list blobs and generate signed URLs.

    Args:
        chat_id: The ID of the chat/user whose files are to be loaded.

    Returns:
        A list of dictionaries, each containing 'name', 'path', and 'downloadURL'.
    """
    try:
        # Import storage here to avoid issues if the module isn't loaded yet
        from firebase_admin import storage
        
        # Get the default storage bucket (assumes service account has access)
        bucket = storage.bucket()
        
        # Define the path prefix to list files within
        uploads_path_prefix = f"chats/{chat_id}/uploads/"
        
        # List all blobs (files) under the given prefix
        # By default, this lists non-directory objects
        blobs = bucket.list_blobs(prefix=uploads_path_prefix)

        file_infos = []
        for blob in blobs:
            # Exclude the directory itself or any folder markers if they appear
            if blob.name == uploads_path_prefix or blob.name.endswith('/'):
                continue
                        
            # os.path.basename extracts the filename from the full path
            file_infos.append({
                "filename": os.path.basename(blob.name), 
                "path": blob.name,
            })


        print("files in chat according to storage", file_infos)
        return file_infos

    except Exception as error:
        # Log the error and return an empty list upon failure, matching the JavaScript
        print(f"≡ƒöÑ Failed to list files for chat {chat_id}: {error}")
        return []
# ============================================================================
# FIXED TOOLS WITH PROPER DECORATORS
# ============================================================================

@tool
async def respond_to_student(
    user_message: str,
    response_focus: str = "general"
) -> Dict[str, Any]:
    """
    Provide educational responses and explanations to student questions.
    
    Use this for general tutoring, explanations, answering questions, providing
    feedback, or any conversational response that doesn't require other tools.
    
    Use spacing, font-weight , line breaks and emojis to make the text easier to read for dyslexic people
    
    Args:
        user_message: The student's question or message
        response_focus: Type of response needed ("explanation", "feedback", "encouragement", "general")
    
    Returns:
        Dictionary with response content for streaming
    """
    try:
        # This tool doesn't need to do heavy processing
        # # It just signals that the LLM should provide a direct response
        return {
            "status": "respond_directly",
            "user_message": user_message,
            "response_focus": response_focus,
            "message": "Providing educational response to student"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Response generation failed: {str(e)}"
        }


#tool study sheet
@tool
async def generate_study_sheet(user_request: str) -> Dict[str, Any]:
    """Generate or Update a personalized study sheet based on user's request."""
    try:
        session = get_session()
        document_content="none"
        
        if(session.documents):
            document_content = await _get_study_sheet_content(session)
            print("DOCUMENT CONTENT FOUND: ", document_content)
              
        # Generate complete HTML study sheet
        html_content = await _create_study_sheet_with_anthropic(
            document_content=document_content,
            user_request=user_request,
            language=session.user_language
        )
             
        study_sheet_result = {
            "status": "success",
            "html_content": html_content,
            "message": f"Created HTML study sheet based on: {user_request}"
        }
        
        print("HTML STUDY SHEET", study_sheet_result)
        
        return study_sheet_result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Study sheet generation failed: {str(e)}"
        }

@tool
async def generate_study_sheet_stream(
    topic: str,
    num_sections: Optional[int] = 6
) -> dict:
    """
    Generate a comprehensive, interactive study guide with progressive loading.
    
    Use this tool ONLY when the user explicitly asks for a study sheet on certain subject or for specific goal:
    - "Make me a study sheet for[topic]"
    - "Fais moi une fuille d'etude sur[topic]" 
    - "Based on the last quiz, make me a study sheet based on my results"
    - "Make me a study sheet based on my files"  

    DO NOT use for:
    - Simple questions like "What is X?" (use search_documents instead)
    - Quiz generation (use generate_quiz)
    - Document summaries (use summarize_document)
    
    Args:
        topic: The topic to create a study guide for
        num_sections: Number of sections (default 6)
    
    Returns:
        dict with type "study_guide_trigger" to initiate progressive loading
    """
    
    # Signal to orchestrator to start progressive generation
    return {
        "type": "study_guide_trigger",
        "topic": topic,
        "num_sections": num_sections,
        "message": f"I'll create a comprehensive study guide about {topic}."
    }

async def _get_study_sheet_content(session: PersistentSessionContext) -> str:
    """Get content from uploaded documents for study sheet generation"""
    
    print("Checking for docs to create study sheet from!!")
    
    try:
        if not session.vectorstore:
            print(" no vectorstore found when trying to find docs")

            fetched_vectorstore = get_chat_vectorstore(chat_id=session.chat_id)
        
            if not fetched_vectorstore:
                print("tried to fetch vectorstore from firebase but it did not work")
            else:
                session.vectorstore = fetched_vectorstore
        
        # Get all document chunks (similar to your quiz generation)
        docs = session.vectorstore.similarity_search(query="", k=1000)
        
        if not docs:
            print("WOW, no docs could be create from vectorstore")
            return ""
        
        # Combine content, limiting size for API
        full_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"CONTENT FOR STUDY SHEET!!! {full_text}")
        return full_text[:12000]  # Limit for Anthropic API
    except Exception as e:
        print("Excpetion occured in get_study_sheet_content() ",e)
        return ""

def get_study_sheet_messages(chat_id):
    """Get all study sheet messages from a specific chat"""
    try:
        # Reference the messages collection for the chat
        
        db = get_firestore_client()
        
        messages_ref = db.collection("chats").document(chat_id).collection("messages")
        
        # Query messages ordered by timestamp (same as your JavaScript)
        messages_query = messages_ref.order_by("timestamp", direction=firestore.Query.ASCENDING)
        
        # Get all messages
        messages = messages_query.stream()
        
        study_sheet_messages = []
        
        for doc in messages:
            message_data = doc.to_dict()
            message_data['id'] = doc.id
            
            # Filter for study sheet, by looking for html object in the messages
            if (message_data.get('html')):  
                study_sheet_messages.append(message_data)
        
        return study_sheet_messages
        
    except Exception as e:
        print(f"Error fetching study sheet messages: {e}")
        return []

async def get_chat_context_from_db(chat_id: str) -> dict:
    """Query Firebase directly to get structured chat context"""
    try:
        db = get_firestore_client()
        
        # Query messages ordered by timestamp
        messages_ref = db.collection("chats").document(chat_id).collection("messages")
        messages_query = messages_ref.order_by("timestamp", direction=firestore.Query.ASCENDING)
        messages = messages_query.stream()
        
        conversation_history = []
        quizzes_created = []
        study_sheets_created = []
        
        for doc in messages:
            message_data = doc.to_dict()

            # Regular conversation messages (skip quiz, scenario, and other non-text message types)
            message_type = message_data.get('type', '')
            skip_types = ['quiz',
                          'scenario',
                          'study_sheet',
                          'flashcards',
                          'suggested_prompts',
                          'upload_loading']
            # to make sure we skip quizzes
            has_quiz_data = message_data.get('quizData') is not None

            # get role and conent
            if message_data.get('role') and message_data.get('content') and message_type not in skip_types and not has_quiz_data:
                if isinstance(message_data.get('content'), str):
                    conversation_history.append({
                        'role': message_data['role'],
                        'content': message_data['content']
                    })
            
            # Extract quizzes separately to build the context
            if message_data.get('quizData'):
                quizzes_created.append({
                    'timestamp': message_data.get('timestamp'),
                    'quiz_data': message_data['quizData']
                })
            
            # Extract study sheets separately to build the context
            if message_data.get('html'):
                study_sheets_created.append({
                    'timestamp': message_data.get('timestamp'),
                    'html_content': message_data['html']
                })
        
        return {
            'conversation': conversation_history[-20:],
            'quizzes': quizzes_created,
            'study_sheets': study_sheets_created
        }
        
    except Exception as e:
        print(f"Error querying chat context: {e}")
        return {'conversation': [], 'quizzes': [], 'study_sheets': []}
    
async def _create_study_sheet_with_anthropic(
    document_content: str,
    user_request: str,
    language: str,
) -> str:  # Return HTML string instead of dict
    """Generate complete HTML study sheet with Anthropic primary, OpenAI fallback"""
    
    session = get_session()
    context_study_sheet = await get_chat_context_from_db(session.chat_id)
    
    prompt = f"""
        Create a complete HTML study sheet with the following exact specifications:

        Your primary goal is to display the information to make it easy to read and understand
        Focus on the User experience during the design process to display the information
        
        USER REQUEST: {user_request}
        DOCUMENT CONTENT: {document_content if document_content else "No document content provided"}
        CONVERSATION CONTEXT: {context_study_sheet["conversation"] if context_study_sheet["conversation"] else "No previous conversation"}
        PREVIOUS STUDY SHEETS: {context_study_sheet["study_sheets"] if context_study_sheet["study_sheets"] else "No study sheets create previously" }
        PREVIOUS QUIZZES: {context_study_sheet["quizzes"] if context_study_sheet["quizzes"] else "No quiz created previously." }
        
        Language of the content : {language}
        
        DESIGN REQUIREMENTS (follow exactly):
        - Color scheme: Primary #b58cd6, Secondary #d9b8f4, with MOSTLY white minimalist background
        - No pink background, background color is always white
        - Glass morphism design with subtle transparency and backdrop blur effects
        - Responsive layout that works on mobile and desktop
        - Clean, modern typography with good readability
        - Interactive collapsible sections with smooth animations

        CONTENT STRUCTURE (include these sections in order):
        1. Header with study sheet title
        2. "Document Summary" section (if document_content exists) - key concepts from uploaded materials
        3. "Conversation Focus" section - topics discussed/questions asked that indicate learning gaps
        4. "Key Concepts" section - important definitions and explanations
        5. "Quick Reference" section - bullet points and essential facts
        6. Interactive elements: collapsible sections (if it enhances the UX), print button
        
        MEDICAL/NURSING FRAMEWORK SECTIONS (include ONLY IF an illness or a health condition is mentionned in the context):
        4. "Physiopathologie" - disease mechanisms, pathological processes
        5. "Causes" - etiology, precipitating factors  
        6. "Facteurs de risque" - risk factors, predisposing conditions
        7. "Investigations / Examen para-clinique / Laboratoire" - diagnostic tests, lab values, imaging
        8. "Traitement pharmacologique" - medications, dosages, mechanisms, side effects
        9. "Traitement non-pharmacologique" - non-drug interventions, lifestyle modifications
        10. "Interventions infirmi├¿res" - nursing assessments, interventions, monitoring, patient education
        
        TECHNICAL REQUIREMENTS:
        - Self-contained HTML with embedded CSS and JavaScript
        - Glass effect using backdrop-filter and rgba colors
        - Print-friendly styles that remove glass effects
        - Mobile-first responsive design
        - Smooth transitions and hover effects

        CONTENT PRIORITY:
        1. Base content primarily on document materials if available
        2. Address specific topics from conversation (these indicate knowledge gaps)
        3. Create comprehensive coverage of the subject
        4. Include practical study aids and memory techniques
        
        IMPORTANT ALL THE TEXT SHOULD BE IN THE LANGUAGE ASKED 

        Return ONLY the complete HTML code with no explanations or markdown formatting.
        """
    
    # Try Anthropic first
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html_content = response.content[0].text
        print("Γ£à Study sheet generated successfully with Anthropic")
        
    except Exception as e:
        print(f"Anthropic API error: {e}")
        
        # Fallback to OpenAI
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=6000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            html_content = response.choices[0].message.content
            print("Γ£à Study sheet generated successfully with OpenAI (fallback)")
            
        except Exception as openai_error:
            print(f"OpenAI API error: {openai_error}")
            raise Exception(f"Both AI providers failed. Anthropic: {str(e)}, OpenAI: {str(openai_error)}")
    
    # Clean any markdown formatting if present (works for both providers)
    if html_content.startswith("```html"):
        html_content = html_content.split("```html")[1].split("```")[0]
    elif html_content.startswith("```"):
        html_content = html_content.split("```")[1].split("```")[0]
    
    # Validate HTML output
    html_content = html_content.strip()
    if not html_content.startswith("<!DOCTYPE html") and not html_content.startswith("<html"):
        raise Exception("Generated content is not valid HTML format")
        
    return html_content

# tools/enhanced_summary_tools.py

@tool
async def summarize_document(
    filename: str,
    detail_level: str = "detailed",
    query: str = "main concepts key points summary"
) -> dict:
    """
    Generate a comprehensive summary by searching vector store for relevant content.
    
    Args:
        filename: Name of the uploaded file to summarize
        detail_level: "brief", "detailed", or "comprehensive"
        query: Search query to find relevant chunks (optional)
    
    Returns:
        Dict with relevant chunks ready for summary streaming
    """
    try:
        session = get_session()
        
        print("SEARCHING FOR REVELENT STUFF FOR SUMMARY")       
        
        # Search vector store for relevant chunks
        relevant_chunks = await _search_vectorstore_for_summary(
            filename, 
            session.chat_id, 
            query,
            detail_level
        )
        
        if not relevant_chunks:
            print("SEARCHING THROUGH DOCUMENTS FOR SUMMARY FAILED")
            return {
                "error": f"Could not find content for document: {filename}",
                "status": "failed"
            }
        
        print("FOUND RELEVANT STUFF FOR SUMMARY")
        return {
            "status": "ready_for_streaming",
            "filename": filename,
            "detail_level": detail_level,
            "relevant_chunks": relevant_chunks,
            "chunk_count": len(relevant_chunks)
        }
        
    except Exception as e:
        return {
            "error": f"Vector store search failed: {str(e)}",
            "status": "failed"
        }


async def _search_vectorstore_for_summary(
    filename: str, 
    chat_id: str, 
    query: str,
    detail_level: str
) -> list:
    """
    Search vector store for relevant chunks to create summary.
    
    CACHE-FIRST ARCHITECTURE:
    1. Check in-memory session.vectorstore (FAST - 0ms)
    2. Fallback to Firebase download if cache miss (SLOW - 500ms)
    
    Args:
        filename: Requested file to summarize (may be empty)
        chat_id: Chat session ID
        query: Search query (usually empty for full doc summary)
        detail_level: "brief" | "detailed" | "comprehensive"
    
    Returns:
        List of chunk dicts: [{"content": str, "metadata": dict}]
    """
    print("\n" + "="*80)
    print("≡ƒöì _search_vectorstore_for_summary CALLED")
    print(f"   ≡ƒôä filename: {filename}")
    print(f"   ≡ƒÆ¼ chat_id: {chat_id}")
    print(f"   ≡ƒöÄ query: {query}")
    print(f"   ≡ƒôè detail_level: {detail_level}")
    print("="*80)
    
    try:
        # ========================================
        # STEP 1: Get session and determine file to summarize
        # ========================================
        print("\n≡ƒôª STEP 1: Retrieving session...")
        session = get_session()
        print(f"   Γ£à Session retrieved: {session}")
        
        if not session.documents:
            print("   ΓÜá∩╕Å WARNING: No documents in session!")
            return []
        
        print(f"   ≡ƒôÜ Total documents in session: {len(session.documents)}")
        print(f"   ≡ƒôï Document list: {[doc.get('filename') for doc in session.documents]}")
        
        # Get last uploaded file as default
        lastfile = session.documents[-1]
        print(f"\n   ≡ƒùé∩╕Å Last file uploaded: {lastfile}")
        
        # Default to last file
        fileToSummarize = lastfile['filename']
        print(f"   ≡ƒôî Default file to summarize: {fileToSummarize}")
        
        # Override if specific file requested and exists
        is_requested_file_uploaded = any(doc.get('filename') == filename for doc in session.documents)
        print(f"\n   ≡ƒöì Checking if requested file '{filename}' exists in uploads...")
        print(f"   {'Γ£à' if is_requested_file_uploaded else 'Γ¥î'} Requested file found: {is_requested_file_uploaded}")
        
        if is_requested_file_uploaded:
            fileToSummarize = filename
            print(f"   ≡ƒÄ» Using requested file: {fileToSummarize}")
        else:
            print(f"   ≡ƒöä Falling back to last file: {fileToSummarize}")
        
        print(f"\n   Γ£à FINAL DECISION: Will summarize '{fileToSummarize}'")
        
        # ========================================
        # STEP 2: CHECK IN-MEMORY CACHE FIRST ΓÜí
        # ========================================
        print("\n" + "="*80)
        print("ΓÜí STEP 2: Checking in-memory cache...")
        print("="*80)
        
        if session.vectorstore:
            print("   Γ£à Cache HIT - Vectorstore found in memory!")
            print(f"   ≡ƒôè Vectorstore type: {type(session.vectorstore)}")
            
            try:
                # Get document count (if supported)
                print(f"   ≡ƒöó Attempting to count documents in cache...")
                
                # Perform similarity search with filter
                print(f"\n   ≡ƒöÄ Searching cache for chunks from '{fileToSummarize}'...")
                print(f"   ≡ƒöì Search params:")
                print(f"      - query: '{query}' (empty = get all)")
                print(f"      - k: 1000 (max chunks)")
                print(f"      - filter: {{'source': '{fileToSummarize}'}}")
                
                docs = session.vectorstore.similarity_search(
                    query="",  # Empty query = get all chunks
                    k=1000,    # Get up to 1000 chunks
                    filter={"source": fileToSummarize}
                )
                
                print(f"   Γ£à Search complete!")
                print(f"   ≡ƒôª Found {len(docs)} chunks in cache")
                
                if docs:
                    # Log first chunk preview
                    first_chunk_preview = docs[0].page_content[:100] + "..." if len(docs[0].page_content) > 100 else docs[0].page_content
                    print(f"\n   ≡ƒôä First chunk preview:")
                    print(f"      Length: {len(docs[0].page_content)} chars")
                    print(f"      Content: {first_chunk_preview}")
                    print(f"      Metadata: {docs[0].metadata}")
                    
                    # Apply detail level limits
                    print(f"\n   Γ£é∩╕Å Applying detail level limits...")
                    chunk_limits = {
                        "brief": 20000,
                        "detailed": 60000,
                        "comprehensive": 120000
                    }
                    
                    limit = chunk_limits.get(detail_level, 20000)
                    print(f"   ≡ƒôÅ Detail level '{detail_level}' ΓåÆ limit: {limit} chars")
                    
                    # Combine all chunks
                    print(f"   ≡ƒöù Combining {len(docs)} chunks...")
                    full_text = "\n\n".join([doc.page_content for doc in docs])
                    total_chars = len(full_text)
                    print(f"   ≡ƒôè Total combined text: {total_chars} chars")
                    
                    # Truncate if needed
                    truncated_text = full_text[:limit]
                    if len(truncated_text) < len(full_text):
                        print(f"   Γ£é∩╕Å Truncated from {total_chars} to {len(truncated_text)} chars")
                    else:
                        print(f"   Γ£à No truncation needed ({total_chars} < {limit})")
                    
                    result = [{"content": truncated_text, "metadata": {"source": fileToSummarize}}]
                    
                    print(f"\n   ≡ƒÄë SUCCESS - Returning from CACHE")
                    print(f"   ≡ƒôª Result: 1 chunk with {len(truncated_text)} chars")
                    print("="*80)
                    
                    return result
                else:
                    print(f"   ΓÜá∩╕Å No chunks found in cache for '{fileToSummarize}'")
                    print(f"   ≡ƒñö This might mean:")
                    print(f"      - File wasn't uploaded yet")
                    print(f"      - Filename mismatch")
                    print(f"      - Vectorstore doesn't have this file")
                    print(f"   ≡ƒôÑ Will try downloading from Firebase...")
                    
            except Exception as cache_error:
                print(f"   Γ¥î Cache search failed: {cache_error}")
                import traceback
                traceback.print_exc()
                print(f"   ≡ƒôÑ Will try downloading from Firebase...")
        else:
            print("   Γ¥î Cache MISS - No vectorstore in memory")
            print("   ≡ƒñö Possible reasons:")
            print("      - First time accessing this session")
            print("      - Vectorstore not loaded from Firebase yet")
            print("      - Session was cleared")
            print("   ≡ƒôÑ Will download from Firebase...")
        
        # ========================================
        # STEP 3: FALLBACK - DOWNLOAD FROM FIREBASE ≡ƒöÑ
        # ========================================
        print("\n" + "="*80)
        print("≡ƒôÑ STEP 3: Downloading from Firebase (FALLBACK)")
        print("="*80)
        
        with tempfile.TemporaryDirectory() as tempdir:
            print(f"   ≡ƒôü Created temp directory: {tempdir}")
            
            bucket = storage.bucket()
            print(f"   ≡ƒ¬ú Firebase bucket: {bucket.name}")
            
            # Construct Firebase paths
            firebase_base = f"FileVectorStore/{chat_id}/{fileToSummarize}"
            faiss_path_firebase = f"{firebase_base}/index.faiss"
            pkl_path_firebase = f"{firebase_base}/index.pkl"
            
            print(f"\n   ≡ƒù║∩╕Å Firebase paths:")
            print(f"      FAISS: {faiss_path_firebase}")
            print(f"      PKL:   {pkl_path_firebase}")
            
            # Local paths
            faiss_path = os.path.join(tempdir, "index.faiss")
            pkl_path = os.path.join(tempdir, "index.pkl")
            
            print(f"\n   ≡ƒÆ╛ Local paths:")
            print(f"      FAISS: {faiss_path}")
            print(f"      PKL:   {pkl_path}")
            
            # Get blobs
            print(f"\n   ≡ƒöì Checking if files exist in Firebase...")
            faiss_blob = bucket.blob(faiss_path_firebase)
            pkl_blob = bucket.blob(pkl_path_firebase)
            
            faiss_exists = faiss_blob.exists()
            pkl_exists = pkl_blob.exists()
            
            print(f"      FAISS exists: {faiss_exists}")
            print(f"      PKL exists:   {pkl_exists}")
            
            if not faiss_exists or not pkl_exists:
                print(f"\n   Γ¥î ERROR: Vectorstore files not found in Firebase!")
                print(f"      This means the file was never uploaded or upload failed")
                return []
            
            # Download files
            print(f"\n   ≡ƒôÑ Downloading FAISS file...")
            faiss_blob.download_to_filename(faiss_path)
            faiss_size = os.path.getsize(faiss_path)
            print(f"      Γ£à Downloaded: {faiss_size} bytes")
            
            print(f"\n   ≡ƒôÑ Downloading PKL file...")
            pkl_blob.download_to_filename(pkl_path)
            pkl_size = os.path.getsize(pkl_path)
            print(f"      Γ£à Downloaded: {pkl_size} bytes")
            
            # Load vector store
            print(f"\n   ≡ƒöñ Loading vectorstore from downloaded files...")
            vectorstore = FAISS.load_local(
                tempdir, 
                OpenAIEmbeddings(), 
                allow_dangerous_deserialization=True
            )
            print(f"      Γ£à Vectorstore loaded successfully")
            print(f"      Type: {type(vectorstore)}")
            
            # Search vectorstore
            print(f"\n   ≡ƒöÄ Searching downloaded vectorstore...")
            print(f"      - query: '{query}'")
            print(f"      - k: 1000")
            print(f"      - filter: {{'source': '{fileToSummarize}'}}")
            
            docs = vectorstore.similarity_search(
                query="", 
                k=1000, 
                filter={"source": fileToSummarize}
            )
            
            print(f"      Γ£à Found {len(docs)} chunks")
            
            if not docs:
                print(f"\n   Γ¥î No documents found with source '{fileToSummarize}'")
                print(f"      This might mean a metadata mismatch")
                return []
            
            # Log first chunk
            first_chunk_preview = docs[0].page_content[:100] + "..." if len(docs[0].page_content) > 100 else docs[0].page_content
            print(f"\n   ≡ƒôä First chunk preview:")
            print(f"      Length: {len(docs[0].page_content)} chars")
            print(f"      Content: {first_chunk_preview}")
            print(f"      Metadata: {docs[0].metadata}")
            
            # Apply limits
            print(f"\n   Γ£é∩╕Å Applying detail level limits...")
            chunk_limits = {
                "brief": 20000,
                "detailed": 60000,
                "comprehensive": 120000
            }
            
            limit = chunk_limits.get(detail_level, 20000)
            print(f"   ≡ƒôÅ Detail level '{detail_level}' ΓåÆ limit: {limit} chars")
            
            # Combine chunks
            print(f"   ≡ƒöù Combining {len(docs)} chunks...")
            full_text = "\n\n".join([doc.page_content for doc in docs])
            total_chars = len(full_text)
            print(f"   ≡ƒôè Total combined text: {total_chars} chars")
            
            # Truncate
            truncated_text = full_text[:limit]
            if len(truncated_text) < len(full_text):
                print(f"   Γ£é∩╕Å Truncated from {total_chars} to {len(truncated_text)} chars")
            else:
                print(f"   Γ£à No truncation needed")
            
            result = [{"content": truncated_text, "metadata": {"source": fileToSummarize}}]
            
            print(f"\n   ≡ƒÄë SUCCESS - Returning from FIREBASE")
            print(f"   ≡ƒôª Result: 1 chunk with {len(truncated_text)} chars")
            print("="*80)
            
            return result
            
    except Exception as e:
        print("\n" + "="*80)
        print(f"≡ƒöÑ FATAL ERROR in _search_vectorstore_for_summary")
        print("="*80)
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {e}")
        print(f"\n   ≡ƒôï Full traceback:")
        import traceback
        traceback.print_exc()
        print("="*80)
        return []
    
def get_chat_vectorstore(
    chat_id: str, 
) -> FAISS:
    """
    Search vector store for relevant chunks to create summary
    """
    try:
        session = get_session()
        if(session.documents):
            with tempfile.TemporaryDirectory() as tempdir:
                # Use same pattern as your working code
                bucket = storage.bucket()
                
                faiss_blob = bucket.blob(f"vectorstores/{chat_id}/index.faiss")
                pkl_blob = bucket.blob(f"vectorstores/{chat_id}/index.pkl")
                
                faiss_path = os.path.join(tempdir, "index.faiss")
                pkl_path = os.path.join(tempdir, "index.pkl")
                
                # Download files (same as your working code)
                faiss_blob.download_to_filename(faiss_path)
                pkl_blob.download_to_filename(pkl_path)
                
                # Load vector store (same as your working code)
                vectorstore = FAISS.load_local(
                    tempdir, 
                    OpenAIEmbeddings(), 
                    allow_dangerous_deserialization=True
                )
                
                return vectorstore
        else:
            return None
                
    except Exception as e:
        print(f"HEY, COULD NOT FIND vectorstore for {chat_id}, exception",e)
        return None

@tool
async def search_documents(query: str, max_results: int = 8) -> Dict[str, Any]:
    """
    Search uploaded nursing documents for relevant information.
    
    Use this when students ask questions about their uploaded study materials,
    textbooks, or need specific information from their documents.
    
    Args:
        query: What to search for (e.g., "cardiac medications", "NCLEX tips")
        max_results: Maximum number of document chunks to return (default: 8)
    
    Returns:
        Dictionary with search results and context
    """
    try:
        session = get_session()
        
        if(session.documents):
            # Load vectorstore if not already loaded
            if session.vectorstore is None and session.documents:
                session.vectorstore = await load_vectorstore_from_firebase(session)
                session.vectorstore_loaded = True
            
            if session.vectorstore:
                # Search for relevant content
                docs = session.vectorstore.similarity_search(query, k=max_results)
                for i, doc in enumerate(docs):
                    print(f"Chunk {i}: {doc.page_content[:200]}...")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Cache the retrieval
                session.last_retrieval = context
                
                # Record tool call
                session.tool_calls.append({
                    "tool": "search_documents",
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "chunks_found": len(docs)
                })
                
                return {
                    "status": "success",
                    "context": context,
                    "num_chunks": len(docs),
                    "message": f"Found {len(docs)} relevant sections in your documents."
                }
            
            return {
                "status": "no_documents",
                "message": "No documents available for search. Ask the student to upload study materials first."
            }
        else:
         return {
                "status": "no_documents",
                "message": "No documents available for search. Ask the student to upload study materials first."
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Document search failed: {str(e)}"
        }

@tool
async def generate_quiz_stream(
    topic: str,
    difficulty: str = "medium",
    num_questions: int = 4,
    source_preference: str = "auto",
    question_types: List[str] = None,
    empathetic_message: str = None
) -> Dict[str, Any]:
    """
    Generate a nursing quiz for the student with support for multiple question formats.

    Use this when students request practice questions, want to test their knowledge,
    or need NCLEX-style questions on specific topics.

    IMPORTANT - Detecting Question Types from User Messages:
    - If user EXPLICITLY asks for BOTH types (e.g., "1 mcq and 1 sata", "mix of mcq and sata")
      ΓåÆ question_types=["mcq", "sata"]
    - If user ONLY mentions "SATA", "select all that apply" without mentioning MCQ
      ΓåÆ question_types=["sata"]
    - If user says "NGN format", "next generation", or "mixed format"
      ΓåÆ question_types=["mcq", "sata", "casestudy"]
    - If user just wants regular questions or doesn't specify ΓåÆ use ["mcq"]
    - If user asks for "case study", "drag and drop", "ordering", "prioritization", "bowtie"
      ΓåÆ include "casestudy" in question_types

    Examples of user intents and question_types to use:
    - "Give me a quiz on cardiac care" ΓåÆ question_types=["mcq"]
    - "SATA questions on diabetes" ΓåÆ question_types=["sata"] (ONLY sata requested)
    - "1 mcq and 1 sata question" ΓåÆ question_types=["mcq", "sata"] (BOTH requested)
    - "2 questions, one multiple choice one select all" ΓåÆ question_types=["mcq", "sata"]
    - "NGN format questions on endocrine" ΓåÆ question_types=["mcq", "sata", "casestudy"]
    - "Mixed format NCLEX prep" ΓåÆ question_types=["mcq", "sata", "casestudy"]
    - "Case study question on cardiac" ΓåÆ question_types=["casestudy"]
    - "Drag and drop prioritization question" ΓåÆ question_types=["casestudy"]
    - "Give me a bowtie question" ΓåÆ question_types=["casestudy"]

    Args:
        topic: Subject area (e.g., "pharmacology", "cardiac care", "NCLEX prep")
        difficulty: Question difficulty ("easy", "medium", "hard")
        num_questions: Number of questions to generate (1-50, default: 4)
        source_preference: "documents" (from uploads), "scratch" (general), or "auto"
        question_types: List of question formats to include. Options:
            - "mcq" = Multiple choice (single answer) - DEFAULT
            - "sata" = Select All That Apply (multiple correct answers)
            - "casestudy" = NGN-style case study with drag-and-drop ordering
            If None or empty, defaults to ["mcq"]
        empathetic_message: Optional empathetic understanding text to show before quiz

    Returns:
        Dictionary signaling quiz streaming should begin with question type info
    """

    print("≡ƒÄ» QUIZ TOOL: Initiating streaming quiz generation")

    try:
        session = get_session()

        # Normalize difficulty
        difficulty_map = {
            "facile": "easy", "easy": "easy",
            "moyen": "medium", "medium": "medium", "normal": "medium",
            "difficile": "hard", "hard": "hard", "challenging": "hard"
        }

        normalized_difficulty = difficulty_map.get(difficulty.lower(), difficulty)

        # Determine source
        if source_preference == "auto":
            source = "documents" if session.documents else "scratch"
        else:
            source = source_preference

        # Validate num_questions
        num_questions = max(1, min(15, num_questions))

        # Normalize question_types - default to MCQ if not specified
        if question_types is None or len(question_types) == 0:
            question_types = ["mcq"]

        # Validate question types
        valid_types = ["mcq", "sata", "casestudy", "ordering", "bowtie"]
        question_types = [qt.lower() for qt in question_types if qt.lower() in valid_types]

        # Normalize aliases: ordering and bowtie both map to casestudy
        question_types = ["casestudy" if qt in ["ordering", "bowtie"] else qt for qt in question_types]

        # Fallback to MCQ if no valid types
        if not question_types:
            question_types = ["mcq"]

        print(f"≡ƒôï Question types requested: {question_types}")

        # Record tool call
        session.tool_calls.append({
            "tool": "generate_quiz_stream",
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": normalized_difficulty,
            "num_questions": num_questions,
            "source": source,
            "question_types": question_types,
            "status": "streaming_initiated",
            "has_empathetic_message": bool(empathetic_message)
        })

        # Return signal to orchestrator to handle streaming
        return {
            "status": "quiz_streaming_initiated",
            "metadata": {
                "topic": topic,
                "difficulty": normalized_difficulty,
                "num_questions": num_questions,
                "source": source,
                "language": session.user_language,
                "question_types": question_types,  # NEW: Pass question types to orchestrator
                "empathetic_message": empathetic_message
            },
            "message": f"Starting quiz generation: {num_questions} questions on {topic} (formats: {', '.join(question_types)})"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Quiz generation failed: {str(e)}"
        }


@tool
async def generate_mindmap_stream(
    topic: str = "",
    depth: str = "medium"
) -> Dict[str, Any]:
    """
    Generate a visual mindmap from the student's uploaded documents.

    Use this when students ask to:
    - Create a mindmap
    - Visualize concepts
    - Show a concept map
    - Map out the material
    - Create a visual summary
    - "carte mentale" (French)
    - "schéma conceptuel" (French)

    Args:
        topic: Optional focus topic (empty = entire document)
        depth: How deep to go ("shallow" = main topics only, "medium" = 2 levels, "deep" = 3+ levels)

    Returns:
        Signal to orchestrator to begin mindmap generation
    """
    print("🧠 MINDMAP TOOL: Initiating mindmap generation")

    try:
        session = get_session()

        # Determine source - must have documents
        if not session.documents:
            return {
                "status": "error",
                "message": "No documents uploaded. Please upload study materials first to create a mindmap."
            }

        # Normalize depth
        depth_map = {
            "shallow": "shallow", "simple": "shallow", "basic": "shallow",
            "medium": "medium", "normal": "medium", "standard": "medium",
            "deep": "deep", "detailed": "deep", "comprehensive": "deep"
        }
        normalized_depth = depth_map.get(depth.lower(), "medium")

        print(f"📊 Mindmap parameters: topic='{topic}', depth='{normalized_depth}'")
        print(f"📄 Documents available: {len(session.documents)}")

        return {
            "status": "mindmap_streaming_initiated",
            "metadata": {
                "topic": topic,
                "depth": normalized_depth,
                "language": session.user_language,
                "document_count": len(session.documents)
            },
            "message": f"Creating mindmap{'for ' + topic if topic else ''} from your documents..."
        }

    except Exception as e:
        print(f"❌ Mindmap tool error: {e}")
        return {
            "status": "error",
            "message": f"Mindmap generation failed: {str(e)}"
        }


# Helper function for streaming quiz generation
async def stream_quiz_questions(
    topic: str,
    difficulty: str,
    num_questions: int,
    source: str,
    session: PersistentSessionContext,
    empathetic_message: str = None,
    chat_id: str = None,
    question_types: List[str] = None
):
    """
    Generator that yields complete quiz questions one at a time.
    Called by orchestrator after tool signals streaming intent.

    Supports mixed question types - can generate MCQ, SATA, or a mix based on
    what the user requested via the question_types parameter.

    Args:
        topic: Subject area for the quiz
        difficulty: Question difficulty level
        num_questions: Number of questions to generate
        source: Source preference ("documents" or "scratch")
        session: Current session context
        empathetic_message: Optional empathetic understanding text to stream first
        chat_id: Chat ID for cancellation checking
        question_types: List of question types to generate ["mcq", "sata", "ordering"]
                       If None or empty, defaults to ["mcq"]

    Yields:
        Status updates and complete questions
    """
    # Import SATA and Case Study generators
    from tools.sata_prompts import generate_sata_question, distribute_question_types
    from tools.casestudy_prompts import generate_casestudy_question

    # Default to MCQ if no types specified
    if question_types is None or len(question_types) == 0:
        question_types = ["mcq"]

    print(f"≡ƒôï Generating quiz with question types: {question_types}")

    # Helper to check cancellation
    def is_cancelled():
        manager = get_connection_manager()
        if manager and chat_id:
            return manager.is_cancelled(chat_id)
        return False

    # ≡ƒåò PHASE 1: Stream empathetic message if provided
    if empathetic_message:
        print(f"≡ƒÆ¼ Starting empathetic message streaming...")

        # Check cancellation before starting
        if is_cancelled():
            print(f"≡ƒ¢æ Quiz generation cancelled before empathetic message")
            return

        # Yield start signal
        yield {
            "status": "empathetic_message_start",
            "message": "Understanding your learning needs..."
        }

        # Stream the empathetic message word by word for a human-like effect
        words = empathetic_message.split()
        current_text = ""

        for i, word in enumerate(words):
            # Check cancellation
            if is_cancelled():
                print(f"≡ƒ¢æ Quiz generation cancelled during empathetic message")
                return

            current_text += word + " "

            # Stream in small chunks (every 3-5 words) for better UX
            if (i + 1) % 4 == 0 or i == len(words) - 1:
                yield {
                    "status": "empathetic_message_chunk",
                    "chunk": current_text.strip(),
                    "progress": int((i + 1) / len(words) * 100)
                }

        # Signal empathetic message complete
        yield {
            "status": "empathetic_message_complete",
            "full_message": empathetic_message
        }

        print(f"Γ£à Empathetic message streaming complete")

    # ≡ƒåò PHASE 2: Generate quiz questions
    # Get content based on source
    if source == "documents" and session.vectorstore:
        docs = session.vectorstore.similarity_search(query=topic, k=1000)
        full_text = "\n\n".join([doc.page_content for doc in docs])[:12000]
        content_context = f"Document content:\n{full_text}"
    else:
        content_context = f"""You are generating questions about: {topic}

            If this is a broad topic (like 'research design', 'pharmacology', 'cardiac care'),
            ensure you test diverse subtopics and concepts within that domain."""

    # Track previously generated questions (for deduplication only)
    generated_questions = _extract_previous_questions(session=session, limit=30)

    # Distribute question types across the quiz
    # e.g., for 10 questions with ["mcq", "sata"], might get:
    # ['mcq', 'mcq', 'sata', 'mcq', 'mcq', 'sata', 'mcq', 'mcq', 'sata', 'mcq']
    question_type_sequence = distribute_question_types(num_questions, question_types)
    print(f"≡ƒô¥ Question type distribution: {question_type_sequence}")

    # Generate questions one at a time
    for question_num in range(1, num_questions + 1):

        # ≡ƒ¢æ Check cancellation before generating each question
        if is_cancelled():
            print(f"≡ƒ¢æ Quiz generation cancelled at question {question_num}/{num_questions}")
            return  # Stop generating questions

        # Get the question type for this question
        current_question_type = question_type_sequence[question_num - 1]

        # Yield progress with question type info
        yield {
            "status": "generating",
            "current": question_num,
            "total": num_questions,
            "question_type": current_question_type
        }

        question_data = None

        # Generate based on question type
        if current_question_type == "sata":
            # Generate SATA question
            print(f"≡ƒö╖ Q{question_num}: Generating SATA question")
            question_data = await generate_sata_question(
                topic=topic,
                difficulty=difficulty,
                question_num=question_num,
                language=session.user_language,
                content_context=content_context,
                questions_to_avoid=generated_questions
            )
        elif current_question_type == "casestudy":
            # Generate Case Study / NGN question
            print(f"≡ƒÅÑ Q{question_num}: Generating Case Study question")
            question_data = await generate_casestudy_question(
                topic=topic,
                difficulty=difficulty,
                question_num=question_num,
                language=session.user_language,
                content_context=content_context,
                questions_to_avoid=generated_questions
            )
        else:
            # Generate MCQ question (default)
            print(f"≡ƒö╢ Q{question_num}: Generating MCQ question")

            # ≡ƒÄ▓ Simply pick a random letter for each question
            random_target_letter = random.choice(['A', 'B', 'C', 'D'])
            print(f"≡ƒÄ▓ Randomly assigned correct answer position = {random_target_letter}")

            question_data = await _generate_single_question(
                content=content_context,
                topic=topic,
                difficulty=difficulty,
                question_num=question_num,
                language=session.user_language,
                questions_to_avoid=generated_questions,
                target_letter=random_target_letter
            )

            # Ensure MCQ has questionType field for frontend routing
            if question_data and 'questionType' not in question_data:
                question_data['questionType'] = 'mcq'

        if question_data:
            generated_questions.append(question_data['question'])

            # Log success
            q_type = question_data.get('questionType', 'mcq')
            if q_type == 'sata':
                num_correct = len(question_data.get('answer', []))
                print(f"Γ£à Q{question_num} SATA generated - {num_correct} correct answers")
            elif q_type == 'casestudy':
                num_items = len(question_data.get('correctOrder', []))
                print(f"Γ£à Q{question_num} Case Study generated - {num_items} items to order")
            else:
                answer = question_data.get('answer', '')
                answer_letter = answer[0] if answer else 'A'
                print(f"Γ£à Q{question_num} MCQ generated - Correct answer: {answer_letter}")

            # ≡ƒ¢æ Check cancellation before yielding question
            if is_cancelled():
                print(f"≡ƒ¢æ Quiz generation cancelled after question {question_num} generated")
                return

            yield {
                "status": "question_ready",
                "question": question_data,
                "index": question_num - 1
            }

    yield {
        "status": "quiz_complete",
        "total_generated": num_questions,
        "question_types_used": list(set(question_type_sequence))
    }
async def _generate_single_question(
    content: str,
    topic: str,
    difficulty: str,
    question_num: int,
    language: str,
    questions_to_avoid: list = None,
    target_letter: str = None,
    existing_topics: list = None
) -> dict:
    """
    Generate ONE complete quiz question using LLM.
    Returns fully-formed question object.

    Args:
        existing_topics: List of topic names the user already has in their progress.
                        LLM will try to match to these topics when the question
                        content fits, reducing topic fragmentation.
    """

    if questions_to_avoid is None:
        questions_to_avoid = []
    if existing_topics is None:
        existing_topics = []

    avoid_text = "\n".join([f"- {q}" for q in questions_to_avoid]) if questions_to_avoid else "None - this is the first question"

    if target_letter:
        answer_instruction = f"""
        CRITICAL REQUIREMENT - CORRECT ANSWER POSITION:
        You MUST make option **{target_letter})** the correct answer for this question.

        Design your question and options so that {target_letter} is the most appropriate clinical response.
        - All 4 options should be plausible
        - But {target_letter} should be the BEST choice based on evidence-based practice
        - The other options should be reasonable but less optimal or incorrect
        """
    else:
        answer_instruction = "You can choose any option (A, B, C, or D) as the correct answer."

    if existing_topics:
        topics_list = "\n".join([f"      - {t}" for t in existing_topics[:30]])
        existing_topics_instruction = f"""
    ?? EXISTING TOPICS (PRIORITY MATCHING):
    The user already has these topics in their progress tracking. If the question content
    fits one of these existing topics, USE THAT EXACT TOPIC NAME instead of creating a new one.
    This prevents topic fragmentation and helps the user track their progress better.

    User's existing topics:
{topics_list}

    Rules:
    - If the question fits an existing topic, use that EXACT topic name (case-sensitive)
    - Only create a NEW topic if the question doesn't fit any existing topic
    - Don't force a poor match - if nothing fits well, create a new descriptive topic
    """
    else:
        existing_topics_instruction = ""

    print(f"\n{'='*60}")
    print(f"Generating Question {question_num}")
    print(f"Target answer position: {target_letter}")
    print(f"Existing topics available: {len(existing_topics) if existing_topics else 0}")
    print(f"{'='*60}\n")
    
    prompt = PromptTemplate(
        input_variables=["content", "topic", "difficulty", "question_num", "language",
                    "questions_to_avoid", "answer_instruction", "existing_topics_instruction"],
    template="""
    You are a {language}-speaking nursing quiz generator creating NCLEX-style questions.

    Generate **EXACTLY ONE high-quality multiple choice question** about: {topic}

    Difficulty: {difficulty}
    Question number: {question_num}

    {answer_instruction}

    CRITICAL - DO NOT repeat these questions:
    {questions_to_avoid}

    Context:
    {content}

    Requirements:
    - Test critical thinking and clinical judgment
    - Create a COMPLETELY NEW question testing a DIFFERENT concept/scenario
    - Use realistic nursing scenarios with specific patient details (age, condition, symptoms)
    - Create 4 plausible options (A-D) where ALL options could seem reasonable to a novice
    - The correct answer should be the BEST choice based on evidence-based practice
    {existing_topics_instruction}
    ?? TOPIC ASSIGNMENT:
    - Assign a SPECIFIC topic/subject to this question based on what it tests
    - The topic should be 2-4 words maximum
    - Be specific and descriptive (e.g., "Cardiac Medications" not "Medicine")
    - Use consistent naming across related questions
    - CRITICAL: Write the topic in {language} (same language as the quiz)
    - IF EXISTING TOPICS WERE PROVIDED ABOVE, try to match to one of those first!
    - Examples of good topics in English:
      * "Heart Anatomy"
      * "Blood Pressure"
      * "Wound Assessment"
      * "Pain Management"
      * "Fluid Balance"
      * "Infection Control"
    - Examples of good topics in French:
      * "Anatomie Cardiaque"
      * "Pression Art?rielle"
      * "?valuation des Plaies"
      * "Gestion de la Douleur"
      * "?quilibre Hydrique"
      * "Contr?le des Infections"
    - Examples of BAD topics:
      * "General Knowledge" / "Connaissances G?n?rales" (too broad)
      * "Chapter 5" / "Chapitre 5" (not descriptive)
      * "Various Topics" / "Sujets Divers" (meaningless)

    ?? Return ONLY valid JSON (no markdown wrapper):
    {
        "question": "Detailed clinical scenario in {language}",
        "options": [
            "A) First option",
            "B) Second option",
            "C) Third option",
            "D) Fourth option"
        ],
        "answer": "X) The correct option",
        "justification": "<strong>Option X is correct</strong> because [1-2 sentences explaining why this is the BEST evidence-based choice].<br><br><strong>Option A is incorrect</strong> because [1 sentence explaining the clinical flaw].<br><strong>Option B is incorrect</strong> because [1 sentence explaining the clinical flaw].<br><strong>Option C is incorrect</strong> because [1 sentence explaining the clinical flaw].",
        "topic": "Specific Topic Name",
        "metadata": {
            "sourceLanguage": "{language}",
            "topic": "{topic}",
            "category": "nursing",
            "difficulty": "{difficulty}",
            "correctAnswerIndex": 0,
            "sourceDocument": "conversational_generation",
            "keywords": ["relevant", "keywords"]
        }
    }

    ?? Formatting Rules (CRITICAL):
    - Use **bold** for every "Option X is..." statement
    - Maintain parallel structure (all start the same way)
    - Keep each explanation to 1-2 sentences
    - Test application and analysis, not just recall
    - MUST include the "topic" field at the root level of the JSON
    """
)

    rendered_prompt = prompt.format(
        content=content,
        topic=topic,
        difficulty=difficulty,
        question_num=question_num,
        language=language,
        questions_to_avoid=avoid_text,
        answer_instruction=answer_instruction,
        existing_topics_instruction=existing_topics_instruction
    )

    gemini_model = _get_gemini_model(model_env_var="GEMINI_QUIZ_MODEL", default_model="gemini-2.5-flash")
    if gemini_model:
        try:
            response = gemini_model.generate_content(
                rendered_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2000,
                },
            )
            result_text = getattr(response, "text", "") or ""
            cleaned = result_text.strip().strip("```json").strip("```").strip()
            parsed_question = json.loads(cleaned)

            answer = parsed_question.get('answer', '')
            if answer:
                answer_letter = answer[0]
                parsed_question.setdefault('metadata', {})
                parsed_question['metadata']['correctAnswerIndex'] = max(0, min(3, ord(answer_letter.upper()) - ord('A')))
            return parsed_question
        except Exception as e:
            print(f"? Gemini generation failed, falling back to OpenAI: {e}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = await chain.ainvoke({
            "content": content,
            "topic": topic,
            "difficulty": difficulty,
            "question_num": question_num,
            "language": language,
            "questions_to_avoid": avoid_text,
            "answer_instruction": answer_instruction,
            "existing_topics_instruction": existing_topics_instruction
        })
        
        cleaned = result.strip().strip("```json").strip("```").strip()
        parsed_question = json.loads(cleaned)

        answer = parsed_question.get('answer', '')
        if answer:
            answer_letter = answer[0]
            answer_index = ord(answer_letter) - ord('A')
            if 'metadata' in parsed_question:
                parsed_question['metadata']['correctAnswerIndex'] = answer_index

        if 'topic' not in parsed_question or not parsed_question['topic']:
            if 'metadata' in parsed_question and 'topic' in parsed_question['metadata']:
                parsed_question['topic'] = parsed_question['metadata']['topic']
            else:
                parsed_question['topic'] = topic if topic else "General"
            print(f"?? Topic field missing, assigned: {parsed_question['topic']}")
        else:
            print(f"? Topic assigned: {parsed_question['topic']}")

        return parsed_question
        
    except json.JSONDecodeError as e:
        print(f"? Failed to parse question {question_num}: {e}")
        if 'result' in locals():
            print(f"Raw output: {result[:500]}...")
        return None
    except Exception as e:
        print(f"? Error generating question {question_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _extract_previous_questions(session: PersistentSessionContext, limit: int = 15) -> list:
    """
    Extract question text from recent quiz history for deduplication.
    
    Args:
        session: Current session context
        limit: Maximum number of previous questions to return
    
    Returns:
        List of question strings from previous quizzes
    """
    previous_questions = []
    
    if not session.quizzes:
        return []
    
    # Get last 3 quiz sessions
    for quiz_session in session.quizzes[-3:]:
        quiz_data = quiz_session.get('quiz_data', {})
        
        # Handle both dict and list formats
        if isinstance(quiz_data, dict) and 'quiz' in quiz_data:
            questions = quiz_data['quiz']
        else:
            questions = quiz_data
        
        # Extract question text
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict) and 'question' in q:
                    previous_questions.append(q['question'])
    
    # Return most recent questions up to limit
    return previous_questions[-limit:]



    """
    Replace option letter references in justification text.
    
    Example:
        text: "Option A is incorrect because... Option B is correct..."
        mapping: {'A': 'C', 'B': 'A'}
        result: "Option C is incorrect because... Option A is correct..."
    
    Args:
        text: The justification text with option references
        letter_mapping: Dictionary mapping old letters to new letters
        
    Returns:
        Text with updated option letters
    """
    
    # We need to replace in a way that doesn't cause collisions
    # Strategy: Use temporary placeholders first
    
    # Step 1: Replace with temporary placeholders
    temp_mapping = {
        'A': '__TEMP_A__',
        'B': '__TEMP_B__',
        'C': '__TEMP_C__',
        'D': '__TEMP_D__'
    }
    
    result = text
    
    # Replace "Option A", "option A", etc. with temporary placeholders
    for old_letter in ['A', 'B', 'C', 'D']:
        # Match various formats: "Option A", "option A", "Options A", etc.
        patterns = [
            f"Option {old_letter}",
            f"option {old_letter}",
            f"Options {old_letter}",
            f"options {old_letter}",
        ]
        
        for pattern in patterns:
            result = result.replace(pattern, pattern.replace(old_letter, temp_mapping[old_letter]))
    
    # Step 2: Replace temporary placeholders with new letters
    for old_letter, new_letter in letter_mapping.items():
        temp = temp_mapping[old_letter]
        result = result.replace(temp, new_letter)
    
    return result


def _extract_answer_concept(answer: str, topic: str) -> str:
    """
    Extract the key concept from the correct answer.
    
    Examples:
    - "B) Randomized controlled trial" ΓåÆ "Randomized controlled trial"
    - "A) Descriptive design" ΓåÆ "Descriptive design"
    
    Args:
        answer: The correct answer string (e.g., "B) Randomized controlled trial")
        topic: The quiz topic for context
        
    Returns:
        The core concept being tested
    """
    # Remove the letter prefix (A), B), C), D))
    if ') ' in answer:
        concept = answer.split(') ', 1)[1]
    else:
        concept = answer
    
    # Clean up common patterns
    concept = concept.strip()
    
    return concept

async def load_vectorstore_from_firebase(session: PersistentSessionContext) -> Optional[FAISS]:
    """
    Loads vectorstore based on documents uploaded previously, 
    to generate quizzes or answer questions relating to a specific document
    """
    if not session.documents:
        return None
    
    # Check if already cached in session
    if session.vectorstore is not None:
        print(f"Using session-cached vectorstore for {session.chat_id}")
        return session.vectorstore
    
    # Create cache key from session's documents
    current_doc_hash = frozenset([doc.get("filename") for doc in session.documents])
    
    # Check if documents changed since last load
    if hasattr(session, '_last_doc_hash') and session._last_doc_hash == current_doc_hash:
        print(f"Document set unchanged for {session.chat_id}")
    
    # Load from Firebase
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            bucket = storage.bucket()
            blob_path = f"vectorstores/{session.chat_id}/index.faiss"
            
            blob = bucket.blob(blob_path)
            if not blob.exists():
                print(f"Vectorstore not found for {session.chat_id}")
                return None
            
            # Download files
            blob.download_to_filename(os.path.join(tempdir, "index.faiss"))
            bucket.blob(f"vectorstores/{session.chat_id}/index.pkl").download_to_filename(
                os.path.join(tempdir, "index.pkl")
            )
            
            vectorstore = FAISS.load_local(
                tempdir, 
                OpenAIEmbeddings(), 
                allow_dangerous_deserialization=True
            )
            
            # Cache in the session itself
            session.vectorstore = vectorstore
            session.vectorstore_loaded = True
            session._last_doc_hash = current_doc_hash
            
            print(f"Loaded and cached vectorstore in session {session.chat_id}")
            return vectorstore
            
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None
# ============================================================================
# TOOL COLLECTION CLASS (Updated)
# ============================================================================

@tool
async def analyze_last_quiz_and_generate_practice(
    last_quiz_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze the student's last quiz performance and generate a targeted practice quiz.

    This tool should be used when a student expresses intent to practice their weak areas,
    improve on topics they struggled with, or wants more practice based on previous quiz results.

    The tool will:
    1. Analyze quiz performance (score, weak topics, patterns)
    2. Generate an empathetic, performance-appropriate message
    3. Create a targeted quiz focusing on weak areas

    Args:
        last_quiz_data: Complete quiz data including questions, answers, topics, and user selections

    Returns:
        Streaming generator that yields empathetic message chunks and quiz questions

    Example user intents that should trigger this tool:
    - "I want to practice my weak areas"
    - "Help me improve on topics I struggled with"
    - "I need more practice on what I got wrong"
    - "Practice weak topics"
    """
    session = get_session()

    try:
        # Extract quiz data
        questions = last_quiz_data.get('questions', [])

        if not questions:
            yield {
                "status": "error",
                "message": "No quiz data available to analyze. Please complete a quiz first."
            }
            return

        # Analyze performance
        total_questions = len(questions)
        correct_count = sum(1 for q in questions if q.get('userSelection', {}).get('isCorrect', False))
        percentage = round((correct_count / total_questions) * 100) if total_questions > 0 else 0

        # Analyze topic performance
        topic_performance = {}
        for q in questions:
            topic = q.get('topic', 'General')
            if topic not in topic_performance:
                topic_performance[topic] = {'total': 0, 'correct': 0, 'questions': []}

            topic_performance[topic]['total'] += 1
            topic_performance[topic]['questions'].append(q)

            if q.get('userSelection', {}).get('isCorrect', False):
                topic_performance[topic]['correct'] += 1

        # Calculate topic percentages and identify weak topics
        weak_topics = []
        topic_details = []

        for topic, perf in topic_performance.items():
            topic_pct = round((perf['correct'] / perf['total']) * 100) if perf['total'] > 0 else 0
            topic_details.append(f"{topic}: {perf['correct']}/{perf['total']} ({topic_pct}%)")

            if topic_pct < 60:  # Weak topic threshold
                weak_topics.append({
                    'name': topic,
                    'percentage': topic_pct,
                    'correct': perf['correct'],
                    'total': perf['total']
                })

        # Generate empathetic message based on performance
        empathetic_message = await _generate_performance_message(
            percentage=percentage,
            correct=correct_count,
            total=total_questions,
            weak_topics=weak_topics,
            language=session.user_language
        )

        print(f"≡ƒôè Quiz Analysis: {percentage}% ({correct_count}/{total_questions})")
        print(f"≡ƒôî Weak Topics: {', '.join([t['name'] for t in weak_topics]) if weak_topics else 'None'}")
        print(f"≡ƒÆ¼ Empathetic Message: {empathetic_message[:100]}...")

        # PHASE 1: Stream empathetic message
        yield {
            "status": "empathetic_message_start",
            "message": "Understanding your learning needs..."
        }

        words = empathetic_message.split()
        current_text = ""

        for i, word in enumerate(words):
            current_text += word + " "

            # Stream in chunks of 4 words for smooth UX
            if (i + 1) % 4 == 0 or i == len(words) - 1:
                yield {
                    "status": "empathetic_message_chunk",
                    "chunk": current_text.strip(),
                    "progress": int((i + 1) / len(words) * 100)
                }

        yield {
            "status": "empathetic_message_complete",
            "full_message": empathetic_message
        }

        # PHASE 2: Determine quiz parameters based on weak topics
        if weak_topics:
            # Focus on weakest topics
            target_topics = ', '.join([t['name'] for t in weak_topics[:3]])  # Max 3 topics
            difficulty = "easy" if percentage < 50 else "medium"
            num_questions = 5
        else:
            # No weak topics - challenge with harder questions on all topics
            target_topics = ', '.join(topic_performance.keys())
            difficulty = "hard"
            num_questions = 5

        print(f"≡ƒÄ» Generating practice quiz: {num_questions} {difficulty} questions on {target_topics}")

        # PHASE 3: Call generate_quiz_stream to signal quiz generation with empathetic message
        # This will be picked up by the orchestrator which will handle streaming via stream_quiz_questions
        source_preference = "documents" if session.documents else "scratch"

        # Call the generate_quiz_stream tool to get the signal
        quiz_signal = await generate_quiz_stream(
            topic=target_topics,
            difficulty=difficulty,
            num_questions=num_questions,
            source_preference=source_preference,
            empathetic_message=empathetic_message  # Pass the empathetic message we generated
        )

        # Return the signal for orchestrator to handle
        yield quiz_signal

    except Exception as e:
        print(f"Γ¥î Error in analyze_last_quiz_and_generate_practice: {str(e)}")
        yield {
            "status": "error",
            "message": f"Failed to analyze quiz and generate practice: {str(e)}"
        }


async def _generate_performance_message(
    percentage: int,
    correct: int,
    total: int,
    weak_topics: List[Dict[str, Any]],
    language: str = "en-US"
) -> str:
    """
    Generate dynamic, empathetic performance message using LLM.

    Each message is uniquely generated to feel authentic and human,
    avoiding repetitive templates.

    Args:
        percentage: Overall score percentage
        correct: Number of correct answers
        total: Total questions
        weak_topics: List of weak topic dictionaries
        language: User's language preference

    Returns:
        Empathetic message string
    """
    is_french = language == "fr"

    # Build topic analysis
    if weak_topics:
        topic_details = "\n".join([
            f"- {t['name']}: {t['correct']}/{t['total']} ({t['percentage']}%)"
            for t in weak_topics[:3]
        ])
        topic_summary = f"Weak areas identified:\n{topic_details}"
    else:
        topic_summary = "No significant weak areas - strong performance across all topics"

    # Determine performance tier and tone
    if percentage < 50:
        tier = "struggling"
        tone_guidance = """Warm, understanding, normalize the difficulty.
        Sound like a supportive friend who gets it. Acknowledge this genuinely sucks but frame practice as the path forward.
        Keep it conversational - imagine texting a friend who just told you they bombed a test."""
    elif percentage < 70:
        tier = "developing"
        tone_guidance = """Encouraging, recognize their progress naturally.
        Point out what they're getting right before mentioning what needs work.
        Casual, friendly tone - like celebrating small wins with a study buddy."""
    elif percentage < 85:
        tier = "proficient"
        tone_guidance = """Genuine praise, then gentle push toward excellence.
        Celebrate their solid performance authentically, then frame practice as fine-tuning.
        Confident, supportive friend who knows they can master this."""
    else:
        tier = "exceptional"
        tone_guidance = """Celebrate their mastery genuinely, then challenge them.
        Show real excitement about their performance, frame next practice as leveling up.
        Like a coach who's genuinely impressed and wants to push their star player higher."""

    # Build dynamic prompt for LLM
    language_instruction = "in French (fr)" if is_french else "in English"

    prompt = f"""You're a supportive nursing tutor and friend. Your nursing student friend just finished a quiz and you can see their results.

≡ƒôè THEIR PERFORMANCE:
- Score: {correct} out of {total} ({percentage}%)
- Performance level: {tier}
- {topic_summary}

≡ƒÄ» YOUR MISSION:
Write a SHORT (2-3 sentences, MAX 60 words), conversational message {language_instruction} that:

1. **Acknowledges their exact score** ({correct}/{total}) - be specific!
2. **Mentions the topics they struggled with by name** - this shows you're paying attention
3. **Briefly says what you're doing next** - creating a practice quiz for them
4. **{tone_guidance}**

≡ƒÜ½ CRITICAL DON'Ts:
- NO generic phrases ("Let's do this together!", "You've got this!", "I'm here to help")
- NO templates or corporate speak
- NO emojis or excessive punctuation
- NO being overly wordy - keep it tight and real
- DO NOT sound like a robot or chatbot

Γ£à CRITICAL DOs:
- BE SPECIFIC: Use their actual score and actual topic names
- BE CONVERSATIONAL: Write like you're texting a friend
- BE AUTHENTIC: Vary your language - no two messages should ever sound the same
- BE BRIEF: 2-3 sentences max, under 60 words
- USE NATURAL LANGUAGE: Contractions, casual phrasing, real human speech

≡ƒÆí TONE GUIDE: {tone_guidance}

Think: "What would I text my nursing student friend right now?"
NOT: "What's the template for this score range?"

Generate your unique, authentic message {language_instruction}:"""

    try:
        # Use OpenAI for fast, conversational responses with HIGH temperature for maximum variety
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.95)  # Very high temp for natural variation

        result = await llm.ainvoke(prompt)
        message = result.content.strip()

        print(f"Γ£à Generated dynamic empathetic message ({len(message)} chars)")
        return message

    except Exception as e:
        print(f"Γ¥î Error generating dynamic message: {e}")

        # Fallback to conversational varied messages
        if is_french:
            fallback_messages = [
                f"{correct} sur {total} sur {weak_topics[0]['name'] if weak_topics else 'ces sujets'} - c'est vraiment difficile au d├⌐but. Je pr├⌐pare des questions plus simples pour t'aider ├á comprendre.",
                f"R├⌐sultat: {correct}/{total}. {weak_topics[0]['name'] if weak_topics else 'Ces concepts'} sont compliqu├⌐s, je sais. On va pratiquer avec des questions cibl├⌐es.",
                f"Hey, {percentage}% - {weak_topics[0]['name'] if weak_topics else 'ce sujet'} casse la t├¬te ├á tout le monde. Je g├⌐n├¿re un quiz adapt├⌐ pour toi."
            ]
        else:
            fallback_messages = [
                f"{correct} out of {total} on {weak_topics[0]['name'] if weak_topics else 'these topics'} - that stuff is genuinely hard at first. I'm setting up easier questions to help you get it.",
                f"Score: {correct}/{total}. {weak_topics[0]['name'] if weak_topics else 'These concepts'} are tricky, I know. We'll practice with targeted questions.",
                f"Hey, {percentage}% - {weak_topics[0]['name'] if weak_topics else 'this topic'} trips everyone up. I'm generating a quiz tailored for you."
            ]

        # Pick a random fallback to add some variety
        import random
        return random.choice(fallback_messages)


class NursingTools:
    """Collection of tools for the nursing tutor - Updated for LangChain integration"""
    # pass context to the constructor
    def __init__(self, session: PersistentSessionContext):
        self.session = session
        # Set global session context for tools to access
        set_session_context(session)

    # return list of tools I have
    def get_tools(self):
        """Return list of tools for LangChain binding"""
        from tools.flashcard_tools import generate_flashcards_stream
        from tools.audio_tools import generate_audio_content

        return [
            search_documents,
            generate_quiz_stream,
            generate_flashcards_stream,
            summarize_document,
            generate_study_sheet_stream,
            generate_audio_content,
            generate_mindmap_stream
        ]
