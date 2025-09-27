from typing import Dict, Any, Optional
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

#to format llm response into string
from langchain_core.output_parsers import StrOutputParser 

# firebase that stores data needed for context (conversation)
from firebase_admin import firestore

# firebase that stores the vector store files
from firebase_admin import storage


def get_firestore_client():
    return firestore.client()

# Global session access - this will be injected by NursingTutor
_CURRENT_SESSION: Optional[PersistentSessionContext] = None

    
def set_session_context(session: PersistentSessionContext):
    """Set the current session context for tool access"""
    global _CURRENT_SESSION
    session.vectorstore = get_chat_vectorstore(session.chat_id)
    _CURRENT_SESSION = session

def get_session() -> PersistentSessionContext:
    """Get current session context"""
    if _CURRENT_SESSION is None:
        raise RuntimeError("No session context available. This is a system error.")
    return _CURRENT_SESSION

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
            
            # Regular conversation messages
            if message_data.get('role') and message_data.get('content'):
                if isinstance(message_data.get('content'), str):
                    conversation_history.append({
                        'role': message_data['role'],
                        'content': message_data['content']
                    })
            
            # Extract quizzes
            if message_data.get('quizData'):
                quizzes_created.append({
                    'timestamp': message_data.get('timestamp'),
                    'quiz_data': message_data['quizData']
                })
            
            # Extract study sheets
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
    """Generate complete HTML study sheet"""
    
    session = get_session()
    context_study_sheet=  await get_chat_context_from_db(session.chat_id)
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
              
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

        Return ONLY the complete HTML code with no explanations or markdown formatting.
        """
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html_content = response.content[0].text
        
        # Clean any markdown formatting if present
        if html_content.startswith("```html"):
            html_content = html_content.split("```html")[1].split("```")[0]
        elif html_content.startswith("```"):
            html_content = html_content.split("```")[1].split("```")[0]
            
        return html_content.strip()
        
    except Exception as e:
        print(f"Anthropic API error: {e}")
        raise Exception(f"Failed to generate study content: {str(e)}")

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
    Search vector store for relevant chunks to create summary
    """
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            # Use same pattern as your working code
            bucket = storage.bucket()
            
            faiss_blob = bucket.blob(f"FileVectorStore/{chat_id}/{filename}/index.faiss")
            pkl_blob = bucket.blob(f"FileVectorStore/{chat_id}/{filename}/index.pkl")
            
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
            
            # Get ALL chunks like your working code
            docs = vectorstore.similarity_search(
                query="", 
                k=1000, 
                filter={"source": filename}
            )
            
            if not docs:
                return []
            
            # Convert to your chunk format but limit based on detail level
            chunk_limits = {
                "brief": 15000,
                "detailed": 20000,
                "comprehensive": 30000
            }
            
            limit = chunk_limits.get(detail_level, 20000)
            full_text = "\n\n".join([doc.page_content for doc in docs])[:limit]
            
            # Return as chunks for your streaming function
            return [{"content": full_text, "metadata": {"source": filename}}]
            
    except Exception as e:
        print(f"🔥 Vector store search error: {e}")
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
        print(f"EY, COULD NOT FIND vectorstore for {chat_id}, exception",e)
        return None

@tool
async def search_documents(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search uploaded nursing documents for relevant information.
    
    Use this when students ask questions about their uploaded study materials,
    textbooks, or need specific information from their documents.
    
    Args:
        query: What to search for (e.g., "cardiac medications", "NCLEX tips")
        max_results: Maximum number of document chunks to return (default: 3)
    
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
async def generate_quiz(
        topic: str, 
        difficulty: str = "medium", 
        num_questions: int = 4,
        source_preference: str = "auto"
    ) -> Dict[str, Any]:
     
    # standard description to tell llm that the tool is for (langchain format)
    """
    Generate a nursing quiz for the student.
    
    Use this when students request practice questions, want to test their knowledge,
    or need NCLEX-style questions on specific topics.
    
    Args:
        topic: Subject area (e.g., "pharmacology", "cardiac care", "NCLEX prep")
        difficulty: Question difficulty ("easy", "medium", "hard")
        num_questions: Number of questions to generate (1-10, default: 4)
        source_preference: "documents" (from uploads), "scratch" (general), or "auto"
    
    Returns:
        Dictionary with quiz questions, answers, and rationales
    """
    
    print("INSIDE THE QUIZ TOOL")
    
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
        num_questions = max(1, min(10, num_questions))
        
        # Generate quiz
        if source == "documents" and session.vectorstore:
            quiz_data = await _generate_from_documents(
                topic=topic,
                difficulty=normalized_difficulty,
                num_questions=num_questions,
                session=session
            )
        else:
            print("NEED TO GENERATE QUIZ FROM SCRATCH")
            quiz_data = await _generate_from_scratch(
                topic=topic,
                difficulty=normalized_difficulty,
                num_questions=num_questions,
                session=session
            )
        
        # Cache the generated quiz
        session.last_quiz_generated = quiz_data
        
        # Record tool call
        session.tool_calls.append({
            "tool": "generate_quiz",
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": normalized_difficulty,
            "num_questions": num_questions,
            "source": source,
            "status": "generated"
        })
        
        return {
            "status": "success",
            "quiz": quiz_data,
            "metadata": {
                "topic": topic,
                "difficulty": normalized_difficulty,
                "question_count": num_questions,
                "source": source
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Quiz generation failed: {str(e)}"
        }

@tool
def check_student_progress() -> Dict[str, Any]:
    """
    Check the current student's learning session and progress.
    
    Use this to understand what the student has been working on,
    what materials they have available, and their recent activity.
    
    Returns:
        Dictionary with session information and learning progress
    """
    try:
        session = get_session()     
        return {
            "status": "success",
            "session_info": {
                "chat_id": session.chat_id,
                "language": session.user_language,
                "has_documents": bool(session.documents),
                "document_count": len(session.documents) if session.documents else 0,
                "vectorstore_loaded": session.vectorstore_loaded,
                "message_count": len(session.message_history),
                "recent_topics": _extract_recent_topics(session),
                "recent_tool_usage": [
                    {k: v for k, v in call.items() if k != 'context'}
                    for call in session.tool_calls[-5:]
                ] if session.tool_calls else [],
                "has_cached_materials": session.last_retrieval is not None,
                "last_quiz_available": session.last_quiz_generated is not None
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Progress check failed: {str(e)}"
        }

# @tool
# def summarize_document() ->
# ============================================================================
# HELPER FUNCTIONS (keep your existing implementation)
# ============================================================================

async def _generate_from_documents(topic: str, 
                                   difficulty: str,
                                   num_questions: int,
                                   session: PersistentSessionContext):
    # TODO: Implement your existing document-based quiz generation
    # This should use the vectorstore to find relevant content and create questions
      # build the vecotorstore with the .faiss and .pkl files
    vectorstore = session.vectorstore
    
     # Get all chunks for the file
    docs = vectorstore.similarity_search(
        query="", 
        k=1000,
    )
    
    if not docs:
        raise HTTPException(status_code=404, detail="FAST API: No content found for this file.")
    
    full_text = "\n\n".join([doc.page_content for doc in docs])[:12000]
    
    prompt = PromptTemplate(
        input_variables=["content", "num_questions", "quiz_type", "language", "filename"],
        template="""
        You are a {language}-speaking quiz generator designed to evaluate understanding of nursing educational documents.

        🎯 Your task is to generate **{num_questions} high-quality {quiz_type} questions** that verify the user's **true understanding** of the following content.

        Do **NOT** copy examples or facts directly. Instead:

        ✅ Create questions that:
        - Apply concepts in **new situations**
        - Require **logic**, **cause-effect reasoning**, or **critical thinking**
        - Involve **scenarios**, **"what if" cases**, or **comparisons**
        - Use clear and correct {language}

        ❌ Do NOT:
        - Ask questions based on literal examples from the text
        - Repeat data, dates, or values
        - Ask questions that could be answered by scanning a line

        🧠 Think like a nursing instructor creating a comprehension test.

        📤 Return ONLY valid JSON in the following structure:
        [
        {{
        "question": "Question text in {language}",
        "options": [
            "A) Option 1",
            "B) Option 2", 
            "C) Option 3",
            "D) Option 4"
        ],
        "answer": "B) Option 2",
        "justification": "Explanation in {language}",
        "metadata": {{
            "sourceLanguage": "{language}",
            "topic": "main_topic_identifier",
            "category": "nursing_category_based_on_content",
            "difficulty": {difficulty},
            "correctAnswerIndex": 1,
            "sourceDocument": "{filename}",
            "keywords": ["keyword1", "keyword2", "keyword3"]
            }}
        }}
        ]

        📌 Metadata Guidelines:
        - topic: Concise identifier (e.g., "medication_administration", "patient_assessment", "infection_control")
        - category: Analyze the document content and choose the most appropriate nursing category (e.g., "fundamentals", "medical_surgical", "pediatrics", "mental_health", "pharmacology", "anatomy_physiology", "pathophysiology", "nursing_process", "ethics", "leadership", etc.)
        - difficulty: easy (basic nursing concepts), medium (application in practice), hard (critical thinking/complex scenarios)
        - keywords: 3-6 relevant nursing/medical terms for search and clustering
        - correctAnswerIndex: Must match the position of correct answer in options array (0-based indexing)
        
        📌 Clinical Accuracy Guidelines:
        - Base questions on evidence-based nursing practice and current medical standards
        - Ensure all scenarios reflect realistic patient populations and conditions
        - For rare conditions, clearly indicate prevalence (e.g., "Though rare in men, breast cancer...")
        - Avoid outdated medical practices or misconceptions
        - Focus on nursing interventions and assessments rather than medical diagnosis
        - Use age-appropriate scenarios for the condition being discussed

        📌 Question Quality Requirements:
        - Questions should test nursing knowledge, not obscure medical facts
        - Scenarios should reflect situations nurses commonly encounter
        - Include diverse patient populations (age, gender, ethnicity) when clinically relevant
        - Ensure all answer options are plausible and realistic
        - Focus on nursing process: assessment, diagnosis, planning, implementation, evaluation

        📄 Document content:
        {content}
        """
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run(
        content=full_text,
        num_questions=num_questions,
        difficulty =difficulty,
        quiz_type="multiple choice",
        filename=session.documents,
        language=session.user_language
    )
    
    
    try:
        cleaned = result.strip().strip("```json").strip("```")
        parsed_quiz = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(e)}")
                       
    return {"quiz": parsed_quiz}

async def _generate_from_scratch(topic: str,
                                 difficulty: str,
                                 num_questions: int,
                                 session: PersistentSessionContext):
    try:
        print(f"starting QUIZ FROM SCRATCH")
        prompt = PromptTemplate(
            input_variables=["topic", "difficulty", "num_questions", "quiz_type", "language"],
            template="""
            You are a quiz generator creating educational questions from scratch.
            
            🎯 Create **{num_questions} high-quality {quiz_type} questions** on the topic: **{topic}**
            📊 Difficulty level: **{difficulty}**

            Requirements:
            ✅ Questions explanation and the content produced should be in the language :{language}
            ✅ Questions should test understanding, not just memorization
            ✅ Create realistic, educational scenarios when appropriate
            ✅ Ensure questions are appropriate for the {difficulty} difficulty level
            ✅ Cover different aspects of {topic}
            
            Important:
            The content of the quiz should be in the user language: {language}
            Number each question, so we can refer to it in the conversation

            📤 Return ONLY valid JSON in the following structure:
            [
            {{
            "question": "1) Question text",
            "options": [
                "A) Option 1",
                "B) Option 2", 
                "C) Option 3",
                "D) Option 4"
            ],
            "answer": "B) Option 2",
            "justification": "Explanation that justifies the answer",
            "metadata": {{
                "sourceLanguage": "{language},
                "topic": "{topic}",
                "category": "educational_category_based_on_topic",
                "difficulty": "{difficulty}",
                "correctAnswerIndex": 1,
                "sourceDocument": "generated_from_scratch",
                "keywords": ["keyword1", "keyword2", "keyword3"]
            }}
            }}
            ]
            """
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
        output_parser = StrOutputParser()

        # Create the chain using LCEL (LangChain Expression Language)
        chain = prompt | llm | output_parser

        # Execute
        result = await chain.ainvoke({
            "topic": topic,
            "difficulty": difficulty, 
            "quiz_type": "mcq",
            "num_questions": num_questions,
            "language": session.user_language
        })
        
        cleaned = result.strip().strip("```json").strip("```")
        parsed_quiz = json.loads(cleaned)
        print(f"this is the quiz{parsed_quiz}")
        return {"quiz": parsed_quiz}

    except Exception as e:
        print("🔥🔥🔥 ERROR IN SCRATCH QUIZ:", e)
        raise Exception(f"Scratch quiz generation failed: {str(e)}")

def _extract_recent_topics(session: PersistentSessionContext) -> list:
    """Extract topics from recent messages and tool calls"""
    topics = []
    
    # Get topics from recent tool calls
    for call in session.tool_calls[-5:]:
        if call.get("tool") == "generate_quiz" and "topic" in call:
            topics.append(call["topic"])
    
    return list(set(topics))  # Remove duplicates

@tool
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
    current_doc_hash = frozenset([doc.filename for doc in session.documents])
    
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
        return [
            search_documents,
            generate_quiz,
            check_student_progress,
            summarize_document,
            generate_study_sheet
        ]