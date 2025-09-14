
from typing import Dict,Any,Optional,Tuple,List
from datetime import datetime
from langchain_community.vectorstores import FAISS
from collections.abc import AsyncGenerator
from models.requests import StatelessChatRequest,ScratchQuizRequest,SummaryRequest
from models.session import PersistentSessionContext
import json
from langchain.prompts import PromptTemplate

class NursingTools:
    """Collection of tools for the nursing tutor"""
    
    @staticmethod
    async def retrieve_context(
        query: str, 
        session: PersistentSessionContext,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Search documents for relevant information.
        Caches vectorstore after first load.
        """
        try:
            # Load vectorstore if not already loaded
            if session.vectorstore is None and session.documents:
                # This would connect to your existing get_vectorstore function
                session.vectorstore = await load_vectorstore_from_firebase(
                    session.chat_id,
                    session.documents
                )
                session.vectorstore_loaded = True
            
            if session.vectorstore:
                # Search for relevant content
                docs = session.vectorstore.similarity_search(query, k=k)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Cache the retrieval
                session.last_retrieval = context
                
                # Record tool call
                session.tool_calls.append({
                    "tool": "retrieve_context",
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "chunks_found": len(docs)
                })
                
                return {
                    "status": "success",
                    "context": context,
                    "num_chunks": len(docs)
                }
            
            return {
                "status": "no_documents",
                "message": "No documents available for search"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    @staticmethod
    async def generate_quiz(
        session: PersistentSessionContext,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        num_questions: Optional[int] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate quiz with accumulating parameters.
        Parameters persist across multiple calls until quiz is generated.
        """
        
        # Update session parameters (accumulate, don't replace)
        if topic is not None:
            session.quiz_params["topic"] = topic
            
        if difficulty is not None:
            # Normalize difficulty input
            difficulty_map = {
                "facile": "easy", "easy": "easy",
                "moyen": "medium", "medium": "medium", "normal": "medium",
                "difficile": "hard", "hard": "hard", "challenging": "hard"
            }
            normalized = difficulty.lower()
            session.quiz_params["difficulty"] = difficulty_map.get(normalized, difficulty)
            
        if num_questions is not None:
            session.quiz_params["num_questions"] = num_questions
            
        if source is not None:
            session.quiz_params["source"] = source
        
        # Determine source automatically if not specified
        if session.quiz_params["source"] is None:
            session.quiz_params["source"] = "documents" if session.documents else "scratch"
        
        # Check what's missing
        missing = []
        if not session.quiz_params["topic"]:
            missing.append("topic")
        if not session.quiz_params["difficulty"]:
            missing.append("difficulty")
        
        # Return incomplete status if parameters missing
        if missing:
            # Generate helpful message in user's language
            messages = {
                'french': {
                    'topic': "Sur quel sujet voulez-vous être évalué?",
                    'difficulty': "Quel niveau de difficulté préférez-vous? (facile/moyen/difficile)"
                },
                'english': {
                    'topic': "What topic would you like to be quizzed on?",
                    'difficulty': "What difficulty level would you prefer? (easy/medium/hard)"
                }
            }
            
            user_msgs = messages.get(session.user_language, messages['english'])
            
            return {
                "status": "incomplete",
                "missing": missing,
                "collected": {k: v for k, v in session.quiz_params.items() if v is not None},
                "next_question": user_msgs.get(missing[0], f"Please provide: {missing[0]}")
            }
        
        # All parameters collected - generate quiz
        try:
            if session.quiz_params["source"] == "documents" and session.vectorstore:
                # Use your existing document-based quiz generation
                quiz_data = await generate_quiz_from_documents_internal(
                    vectorstore=session.vectorstore,
                    params=session.quiz_params,
                    language=session.user_language
                )
            else:
                # Use your existing scratch quiz generation
                quiz_data = await generate_quiz_from_scratch_internal(
                    topic=session.quiz_params["topic"],
                    difficulty=session.quiz_params["difficulty"],
                    num_questions=session.quiz_params["num_questions"],
                    language=session.user_language
                )
            
            # Cache the generated quiz
            session.last_quiz_generated = quiz_data
            
            # Record tool call
            session.tool_calls.append({
                "tool": "generate_quiz",
                "timestamp": datetime.now().isoformat(),
                "params_used": dict(session.quiz_params),
                "status": "generated"
            })
            
            # Clear quiz params for next quiz
            session.quiz_params = {
                "topic": None,
                "difficulty": None,
                "num_questions": 4,
                "quiz_type": "mcq",
                "source": None
            }
            
            return {
                "status": "success",
                "quiz": quiz_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
@staticmethod
def check_session_state(session: PersistentSessionContext) -> Dict[str, Any]:
        """
        Diagnostic tool to check current session state.
        Helps AI understand what's in memory.
        """
        return {
            "chat_id": session.chat_id,
            "user_language": session.user_language,
            "has_documents": bool(session.documents),
            "vectorstore_loaded": session.vectorstore_loaded,
            "pending_quiz_params": {
                k: v for k, v in session.quiz_params.items() 
                if v is not None
            },
            "recent_messages": session.message_history[-5:] if session.message_history else [],
            "recent_tool_calls": [
                {k: v for k, v in call.items() if k != 'context'}  # Exclude large context
                for call in session.tool_calls[-3:]
            ] if session.tool_calls else [],
            "has_cached_retrieval": session.last_retrieval is not None,
            "has_last_quiz": session.last_quiz_generated is not None
        }
   
async def handle_quiz_generation_from_documents(
    session: PersistentSessionContext,
    request: StatelessChatRequest
) -> AsyncGenerator[str, None]:
    """
    Generate quiz from uploaded documents
    """
    if not request.documents:
        yield json.dumps({
            "status": "error", 
            "message": "No documents available. Upload documents or request a quiz from scratch."
        }) + "\n"
        return
    
    yield json.dumps({"status": "processing", "message": "Generating quiz from your documents..."}) + "\n"
    
    try:
        # Use most recent document
        last_doc = request.documents[-1]
        
        quiz_request = QuizRequest(
            chat_id=request.chat_id,
            filename=last_doc.filename,
            num_questions=session.quiz_params.get("num_questions", 4),
            quiz_type=session.quiz_params.get("quiz_type", "mcq"),
            language=request.language
        )
        
        quiz_result = await generate_quiz_based_on_documents(quiz_request)
        yield json.dumps(quiz_result) + "\n"
        
    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"

async def handle_quiz_generation_from_scratch(
    session: PersistentSessionContext,
    request: StatelessChatRequest
) -> AsyncGenerator[str, None]:
    """
    Generate quiz without documents (from scratch)
    """
    yield json.dumps({"status": "processing", "message": "Creating custom quiz..."}) + "\n"
    
    try:
        scratch_request = ScratchQuizRequest(
            chat_id=request.chat_id,
            topic=session.quiz_params.get("topic", "general knowledge"),
            difficulty=session.quiz_params.get("difficulty", "medium"),
            num_questions=session.quiz_params.get("num_questions", 4),
            quiz_type=session.quiz_params.get("quiz_type", "mcq"),
            language=request.language
        )
        
        quiz_result = await generate_quiz_from_scratch(scratch_request, request)
        yield json.dumps(quiz_result) + "\n"
        
    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"

async def generate_quiz_from_scratch(request: ScratchQuizRequest,user_request: StatelessChatRequest):
    try:
        print(f"starting to print quiz!!!")
        prompt = PromptTemplate(
            input_variables=["topic", "difficulty", "num_questions", "quiz_type", "input"],
            template="""
            You are a quiz generator creating educational questions from scratch.
            
            🎯 Create **{num_questions} high-quality {quiz_type} questions** on the topic: **{topic}**
            📊 Difficulty level: **{difficulty}**

            Requirements:
            ✅ Questions explanation and the content produced should be in the same language as the input of the user:{input}
            ✅ Questions should test understanding, not just memorization
            ✅ Create realistic, educational scenarios when appropriate
            ✅ Ensure questions are appropriate for the {difficulty} difficulty level
            ✅ Cover different aspects of {topic}
            
            Important:
            The content of the quiz should be in the user language based on his input : {input}

            📤 Return ONLY valid JSON in the following structure:
            [
            {{
            "question": "Question text",
            "options": [
                "A) Option 1",
                "B) Option 2", 
                "C) Option 3",
                "D) Option 4"
            ],
            "answer": "B) Option 2",
            "justification": "Explanation that justifies the answer",
            "metadata": {{
                "sourceLanguage": "the language of the input of the user",
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

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)
        chain = LLMChain(llm=llm, prompt=prompt)

        result = chain.run(
            topic=request.topic,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            quiz_type="multiple choice" if request.quiz_type == "mcq" else "true/false",
            input=user_request.input
        )
        
        cleaned = result.strip().strip("```json").strip("```")
        parsed_quiz = json.loads(cleaned)
        print(f"this is the quiz{parsed_quiz}")
        return {"quiz": parsed_quiz}

    except Exception as e:
        print("🔥🔥🔥 ERROR IN SCRATCH QUIZ:", e)
        raise Exception(f"Scratch quiz generation failed: {str(e)}")
     
async def load_vectorstore_from_firebase(chat_id: str, documents: List) -> Optional[FAISS]:
    """
    Loads or retrieves a cached vector store for a given set of documents in a chat.
    The cache key is based on the chat ID and the document filenames.
    """
    if not documents:
        return None

    # Create a unique cache key from the chat_id and a frozenset of document filenames.
    # frozenset is used to ensure the order of documents does not change the key.
    document_filenames: frozenset = frozenset([doc.filename for doc in documents])
    cache_key: Tuple[str, frozenset] = (chat_id, document_filenames)

    session_state = check_session_state()

    # Check the in-memory cache first
    if cache_key in session_state:
        print(f"Using cached vector store for chat_id: {chat_id}")
        return cached_vectorstore[cache_key]

    # If not in cache, download and load from Google Cloud Storage
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            bucket = storage.bucket()
            blob_path = f"vectorstores/{chat_id}/index.faiss"
            
            # Check if vector store exists
            blob = bucket.blob(blob_path)
            if not blob.exists():
                print(f"Vector store not found for chat {chat_id}. Cache will not be populated.")
                return None
            
            # Download the FAISS and PKL files
            blob.download_to_filename(os.path.join(tempdir, "index.faiss"))
            bucket.blob(f"vectorstores/{chat_id}/index.pkl").download_to_filename(os.path.join(tempdir, "index.pkl"))
            
            vectorstore = FAISS.load_local(tempdir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            
            # Cache the vector store for subsequent calls with the same key
            cached_vectorstore[cache_key] = vectorstore
            print(f"Loaded and cached new vector store for chat_id: {chat_id}")
            return vectorstore
            
    except Exception as e:
        print(f"Error loading vector store for chat {chat_id}: {e}")
        return None

