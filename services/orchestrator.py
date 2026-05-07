from langchain_openai import ChatOpenAI
from tools.quiztools import NursingTools, set_session_context
from models.session import PersistentSessionContext
from typing import AsyncGenerator, Optional, Tuple
from datetime import datetime
from tools.quiztools import search_documents,summarize_document,get_chat_context_from_db
import json
import re

class NursingTutor:
    """
    Main nursing tutor orchestrator with fixed tool integration
    """

    # ============================================================================
    # FAST ROUTING PATTERNS
    # ============================================================================
    # These patterns enable instant tool selection without an LLM call,
    # reducing first-byte latency by 1-2 seconds for common requests.
    # ============================================================================
    QUIZ_PATTERNS = re.compile(
        r'\b(quiz|test|exam|practice\s+question|nclex|give\s+me\s+question|'
        r'make\s+me\s+question|create\s+question|generate\s+question|'
        r'quiz\s+me|test\s+me|pratique|questionnaire)\b',
        re.IGNORECASE
    )
    FLASHCARD_PATTERNS = re.compile(
        r'\b(flashcard|flash\s+card|carte|cartes)\b',
        re.IGNORECASE
    )
    STUDYSHEET_PATTERNS = re.compile(
        r'\b(study\s+sheet|study\s+guide|fiche|résumé|cheat\s+sheet)\b',
        re.IGNORECASE
    )
    SUMMARIZE_PATTERNS = re.compile(
        r'\b(summarize|summary|résume|résumer)\b',
        re.IGNORECASE
    )
    SEARCH_PATTERNS = re.compile(
        r'\b(search|find|look\s+for|cherche|trouve)\b',
        re.IGNORECASE
    )

    def __init__(self, chat_id: str):
        self.session = PersistentSessionContext(chat_id)
        self.tools_instance = NursingTools(self.session)

        # Get properly decorated tools
        tools = self.tools_instance.get_tools()

        import os
        api_key = os.getenv("OPENAI_API_KEY")

        # Main content generation model (high quality)
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.5,
            streaming=True,
            openai_api_key=api_key
        )

        # Fast routing model for ambiguous cases (lower latency)
        # Used only when pattern matching fails
        self.routing_llm = ChatOpenAI(
            model="gpt-4.1-mini",  # ~2-3x faster than gpt-4.1
            temperature=0,
            streaming=False,
            openai_api_key=api_key
        )

        self.llm_with_tools = self.llm.bind_tools(tools)
        self.routing_llm_with_tools = self.routing_llm.bind_tools(tools)

    # Patterns that indicate NO tool is needed (direct conversation)
    CONVERSATIONAL_PATTERNS = re.compile(
        r'^(hi|hello|hey|bonjour|salut|thanks|thank you|merci|ok|okay|sure|yes|no|'
        r'what is|what are|who is|why|how does|explain|tell me about|'
        r'can you explain|help me understand|i don\'t understand|'
        r'what do you mean|could you clarify)\b',
        re.IGNORECASE
    )

    def _fast_route_check(self, user_input: str) -> Optional[str]:
        """
        Fast pattern-based routing check.

        Returns:
            - Tool name string if a tool pattern matches
            - "NO_TOOL" if it's clearly conversational (skip routing)
            - None if ambiguous (needs LLM routing)

        This skips the LLM routing call entirely for obvious requests,
        reducing first-byte latency by 1-2 seconds.
        """
        input_lower = user_input.lower().strip()

        # Check for quiz patterns (most common)
        if self.QUIZ_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected quiz pattern in '{user_input[:50]}...'")
            return "generate_quiz_stream"

        # Check for flashcard patterns
        if self.FLASHCARD_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected flashcard pattern")
            return "generate_flashcards_stream"

        # Check for study sheet patterns
        if self.STUDYSHEET_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected study sheet pattern")
            return "generate_study_sheet_stream"

        # Check for summarize patterns
        if self.SUMMARIZE_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected summarize pattern")
            return "summarize_document"

        # Check for conversational patterns (no tool needed)
        # Short messages are usually conversational
        if len(input_lower) < 50 and self.CONVERSATIONAL_PATTERNS.match(input_lower):
            print(f"⚡ FAST ROUTE: Detected conversational pattern - skipping tool routing")
            return "NO_TOOL"

        # Short confirmation messages (yes, okay, sure, etc.)
        if len(input_lower) < 20 and input_lower in ["yes", "ok", "okay", "sure", "no", "oui", "non", "d'accord"]:
            print(f"⚡ FAST ROUTE: Detected confirmation - skipping tool routing")
            return "NO_TOOL"

        # No pattern match - will need LLM routing
        return None
    
    async def process_message(
        self,
        user_input: str,
        language: str = "english",
        pre_fetched_context: dict = None
    ):
        """
        Process student message with proper streaming for simple responses,
        non-streaming for tool calls.

        Args:
            user_input: The user's message text
            language: Detected language of the input (e.g., 'english', 'french')
            pre_fetched_context: Optional pre-fetched context from main.py to avoid
                                 duplicate Firebase queries. If provided, we skip
                                 the get_chat_context_from_db() call here.
                                 Expected format: {
                                     'conversation': [...],
                                     'quizzes': [...],
                                     'study_sheets': [...]
                                 }
        """
        try:
            # Update session language
            self.session.user_language = language

            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZATION: Use pre-fetched context if available
            # ═══════════════════════════════════════════════════════════════════
            # If main.py already fetched the context (to avoid duplicate Firebase calls),
            # use that instead of fetching again. This saves ~300-800ms per message.
            # ═══════════════════════════════════════════════════════════════════
            if pre_fetched_context is not None:
                # Use the context that was already fetched by main.py
                full_context_from_db = pre_fetched_context
                print("📦 Using pre-fetched context (skipped duplicate Firebase query)")
            else:
                # Fallback: fetch context if not provided (for backwards compatibility)
                full_context_from_db = await get_chat_context_from_db(self.session.chat_id)
                print("🔄 Fetched context from Firebase (no pre-fetched context provided)")

            # Add user message to history
            try:
                if(full_context_from_db["conversation"]):
                    #print("CONTEXT PREV CONVO",full_context_from_db["conversation"])
                    self.session.message_history = full_context_from_db["conversation"][-15:]
            except Exception as e:
                print("error during conversation context creation",e)

            # add quizzes history to context
            try:
                if(full_context_from_db["quizzes"]):
                    #print("CONTEXT PREV QUIZZES",full_context_from_db["quizzes"])
                    self.session.quizzes = full_context_from_db["quizzes"]
            except Exception as e:
                print("error during quizzes context creation",e)
                
            
            # self.session.message_history.append({
            #     "role": "user",
            #     "content": user_input,
            #     "timestamp": datetime.now().isoformat()
            # })

                # Prepare messages for LLM
            print("About to create messages to feed llm")


            # Ensure session context is available to tools
            set_session_context(self.session)

            # ═══════════════════════════════════════════════════════════════════
            # STEP: CHECK FOR CONTINUATION REQUEST
            # ═══════════════════════════════════════════════════════════════════
            # This checks if the user's message is a short "continuation" request
            # like "more", "again", "another" after completing a quiz/flashcard.
            # If so, we transform the message to make it explicit for the LLM.
            # ═══════════════════════════════════════════════════════════════════

            # Check if this is a continuation request
            continuation_info = self._is_continuation_request(user_input)

            # If it's a continuation, transform the message to be explicit
            if continuation_info['is_continuation']:
                print(f"🔄 CONTINUATION DETECTED!")
                print(f"   Original message: '{user_input}'")
                print(f"   Action type: {continuation_info['action_type']}")
                print(f"   Topic: {continuation_info['topic']}")

                # Transform the message to make intent explicit
                user_input = self._transform_continuation_message(user_input, continuation_info)

                print(f"   Transformed to: '{user_input[:100]}...'")
            else:
                print(f"📝 Normal message (not a continuation): '{user_input[:50]}...'")

            # ═══════════════════════════════════════════════════════════════════
            # END CONTINUATION CHECK
            # ═══════════════════════════════════════════════════════════════════

            # Create nursing-specific system prompt
            system_prompt = self._create_system_prompt()

            print("SYSTEM PROMPT", system_prompt)

            print("System Prompt created")

            messages = []
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    *self.session.message_history,
                    {"role": "user", "content": user_input}
                ]
            except Exception as e:
                print("Error when building message",e)

            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZED ROUTING: Fast pattern matching → Fast LLM → Full LLM
            # ═══════════════════════════════════════════════════════════════════
            # Previously: Every message did a full gpt-4.1 call for routing (1-2s)
            # Now:
            #   1. Fast pattern matching (0ms) - catches 60%+ of tool requests
            #   2. "NO_TOOL" fast path - skips routing for conversational messages
            #   3. gpt-4.1-mini routing (~300-500ms) - for ambiguous cases
            #   4. gpt-4.1 only for actual content generation
            # ═══════════════════════════════════════════════════════════════════

            fast_route_tool = self._fast_route_check(user_input)

            if fast_route_tool == "NO_TOOL":
                # Ultra-fast path: Skip routing entirely, stream directly
                print(f"⚡ ULTRA-FAST: Conversational message detected, streaming immediately")
                response_content = ""
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        yield json.dumps({
                            "answer_chunk": chunk.content
                        }) + "\n"

                # Update history and complete
                self.session.message_history.append({
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now().isoformat()
                })
                yield json.dumps({"status": "complete"}) + "\n"
                return  # Exit early, skip the tool routing path

            elif fast_route_tool:
                # Fast path: Pattern matched, use gpt-4.1-mini to get tool args
                print(f"⚡ FAST ROUTING: Using {fast_route_tool} (pattern matched)")
                response = await self.routing_llm_with_tools.ainvoke(messages)
            else:
                # Ambiguous: Use gpt-4.1-mini for routing decision (faster than gpt-4.1)
                print("🔀 SMART ROUTING: Using gpt-4.1-mini for tool decision...")
                response = await self.routing_llm_with_tools.ainvoke(messages)
            
            # Check if tools were called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls_made = response.tool_calls
                
                # Notify about tool execution
                for tool_call in tool_calls_made:
                    yield json.dumps({
                        "status": "tool_executing",
                        "tool_name": tool_call.get("name"),
                        "message": f"Executing {tool_call.get('name')}..."
                    }) + "\n"
                
                # MANUALLY EXECUTE TOOLS
                tool_results = []
                for tool_call in tool_calls_made:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    print(f"🔥 Manually executing tool: {tool_name} with args: {tool_args}")
                    
                    try:            
                        if tool_name == "search_documents":
                            result = await search_documents.ainvoke(tool_args)
                            tool_results.append(result)
                            
                        elif tool_name == "check_student_progress":
                            from tools.quiztools import check_student_progress
                            result = await check_student_progress.ainvoke(tool_args)
                            tool_results.append(result)
                            
                        elif tool_name == "summarize_document": 
                            print("Tool call summarize_document triggered")                               
                            # Get chunks from vector store using your tool
                            result = await summarize_document.ainvoke(tool_args)
                            
                            if result.get("status") == "ready_for_streaming":
                                # Now stream using your existing method
                                async for chunk in self.stream_document_summary(
                                    result["relevant_chunks"], 
                                    result["detail_level"], 
                                    result["filename"],
                                    language
                                ):
                                    yield json.dumps({
                                        "answer_chunk": chunk
                                    }) + "\n"  
                            else:
                                # Handle error from tool
                                yield json.dumps({
                                    "answer_chunk": result.get("error", "Summarization failed")
                                }) + "\n"
                                                        
                        elif tool_name == "respond_to_student":
                            response_content = ""            
                            async for chunk in self.llm_with_tools.astream(messages):
                                if hasattr(chunk, 'content') and chunk.content:
                                    response_content += chunk.content
                                    yield json.dumps({
                                        "answer_chunk": chunk.content
                                    }) + "\n"
                        elif tool_name == "generate_study_sheet_stream":
                            print("🎓 Study sheet tool triggered")
                                                          
                            topic = tool_args.get("topic")
                              
                            # status expected on the front-end to open the side panel and show the study sheet
                            yield json.dumps({
                                "status": "study_sheet_trigger",
                                "topic": topic,
                                "message": f"Creating study sheet about {topic}..."
                            }) + "\n"
                            
                            if not topic:
                                yield json.dumps({
                                    "status": "error", 
                                    "message": "No topic specified for study sheet"
                                }) + "\n"
                                return
                            
                            print(f"📚 Generating study sheet for: {topic}")

                            # Use new simple study sheet generator (fast, streamed text)
                            from services.studysheet_simple import SimpleStudySheetGenerator
                            generator = SimpleStudySheetGenerator(self.session)

                            # Stream the study sheet
                            async for chunk in generator.generate_study_sheet_stream(topic, language):
                                yield chunk

                            return
                                                       
                        elif tool_name == "generate_flashcards_stream":
                            print("📇 Flashcard tool called - checking for streaming")
                            print(f"📋 Tool arguments received from LLM: {tool_args}")

                            # Import flashcard streaming tools
                            from tools.flashcard_tools import generate_flashcards_stream, stream_flashcards
                            result = await generate_flashcards_stream.ainvoke(tool_args)

                            # Check if tool signaled streaming intent
                            if result.get("status") == "flashcard_streaming_initiated":
                                print("🌊 Starting flashcard streaming from orchestrator")

                                metadata = result.get("metadata", {})

                                # Track flashcards for final save
                                all_flashcards = []

                                # Stream flashcards one by one
                                async for chunk in stream_flashcards(
                                    topic=metadata.get("topic"),
                                    num_cards=metadata.get("num_cards"),
                                    source=metadata.get("source"),
                                    session=self.session,
                                    chat_id=self.session.chat_id
                                ):
                                    # Check status type
                                    status = chunk.get("status")

                                    if status == "flashcard_generating":
                                        # Initial generation signal
                                        yield json.dumps({
                                            "status": "flashcard_generating",
                                            "message": chunk.get("message"),
                                            "current": chunk.get("current"),
                                            "total": chunk.get("total")
                                        }) + "\n"

                                    elif status == "generating":
                                        # Progress update (optional, can be skipped)
                                        pass

                                    elif status == "flashcard_ready":
                                        # Individual flashcard ready
                                        flashcard = chunk.get("flashcard")
                                        all_flashcards.append(flashcard)

                                        yield json.dumps({
                                            "status": "flashcard_ready",
                                            "flashcard": flashcard,
                                            "index": chunk.get("index"),
                                            "total_so_far": chunk.get("total_so_far")
                                        }) + "\n"

                                    elif status == "flashcard_complete":
                                        # All flashcards generated
                                        yield json.dumps({
                                            "status": "flashcard_complete",
                                            "flashcard_data": all_flashcards,
                                            "total_generated": chunk.get("total_generated")
                                        }) + "\n"
                            else:
                                # Tool error
                                yield json.dumps({
                                    "status": "error",
                                    "message": result.get("message", "Flashcard generation failed")
                                }) + "\n"

                            return

                        if tool_name == "generate_mindmap_stream":
                            print("🧠 Mindmap tool called")

                            from tools.quiztools import generate_mindmap_stream
                            result = await generate_mindmap_stream.ainvoke(tool_args)

                            if result.get("status") == "mindmap_streaming_initiated":
                                print("🌊 Starting mindmap streaming from orchestrator")

                                metadata = result.get("metadata", {})

                                from services.mindmap_generator import stream_mindmap_data

                                async for chunk in stream_mindmap_data(
                                    topic=metadata.get("topic", ""),
                                    depth=metadata.get("depth", "medium"),
                                    session=self.session,
                                    chat_id=self.session.chat_id
                                ):
                                    status = chunk.get("status")

                                    if status == "mindmap_generating":
                                        yield json.dumps({
                                            "status": "mindmap_generating",
                                            "message": chunk.get("message")
                                        }) + "\n"

                                    elif status == "mindmap_complete":
                                        yield json.dumps({
                                            "status": "mindmap_complete",
                                            "mindmap_data": chunk.get("mindmap_data")
                                        }) + "\n"

                                    elif status == "error":
                                        yield json.dumps({
                                            "status": "error",
                                            "message": chunk.get("message")
                                        }) + "\n"
                            else:
                                # Tool error (e.g., no documents)
                                yield json.dumps({
                                    "status": "error",
                                    "message": result.get("message", "Mindmap generation failed")
                                }) + "\n"

                            return

                        if tool_name == "generate_quiz_stream":
                            print("🎯 Quiz tool called - checking for streaming")
                            print(f"📋 Tool arguments received from LLM: {tool_args}")
                            print(f"🔍 Empathetic message in args: {tool_args.get('empathetic_message', 'NOT FOUND')}")

                            def transform_question_for_frontend(q: dict) -> dict:
                                """Transform question to StudyQuizCard-compatible format.

                                StudyQuizCard expects:
                                - correctIndex: number (0-indexed)
                                - rationale: string (HTML)

                                Backend provides:
                                - answer: string like "A) Option text" or just "A"
                                - justification: string (HTML rationale)
                                - metadata.correctAnswerIndex: number (if available)
                                """
                                transformed = dict(q)  # Copy original

                                # 1. Extract correctIndex
                                correct_index = -1

                                # Strategy 1: Use metadata.correctAnswerIndex if available
                                if q.get('metadata') and isinstance(q['metadata'], dict):
                                    idx = q['metadata'].get('correctAnswerIndex')
                                    if idx is not None and isinstance(idx, int):
                                        correct_index = idx

                                # Strategy 2: Parse from answer field
                                if correct_index == -1:
                                    answer = q.get('answer', '')
                                    if isinstance(answer, str) and len(answer) > 0:
                                        # Extract first letter (handles "A", "A)", "A. text", etc.)
                                        first_char = answer.strip()[0].upper()
                                        if first_char in 'ABCDEF':
                                            correct_index = ord(first_char) - ord('A')

                                # Strategy 3: If answer is a number
                                if correct_index == -1:
                                    answer = q.get('answer')
                                    if isinstance(answer, int):
                                        correct_index = answer
                                    elif isinstance(answer, str) and answer.isdigit():
                                        correct_index = int(answer)

                                transformed['correctIndex'] = correct_index

                                # 2. Copy rationale fields. After the deferred-
                                # rationale rollout, new generations no longer
                                # include `justification` — they ship a one-sentence
                                # `correct_blurb` instead, and the frontend fetches
                                # the full per-option rationale on demand from
                                # /quiz_rationale. Old saved quizzes still have
                                # `justification`, so we forward whichever exists.
                                if 'justification' in q:
                                    transformed['rationale'] = q['justification']
                                if 'correct_blurb' in q:
                                    transformed['correctBlurb'] = q['correct_blurb']

                                print(f"🔄 Transformed question: correctIndex={correct_index}, has rationale={bool(transformed.get('rationale'))}, has blurb={bool(transformed.get('correctBlurb'))}")
                                return transformed

                            # creates parameters we will need for the quizz, and start streaming
                            from tools.quiztools import generate_quiz_stream
                            result = await generate_quiz_stream.ainvoke(tool_args)

                            # Check if tool signaled streaming intent
                            if result.get("status") == "quiz_streaming_initiated":

                                print("🌊 Starting quiz streaming from orchestrator")

                                metadata = result.get("metadata", {})
                                empathetic_message = metadata.get("empathetic_message")

                                # Import streaming function WITH Question Bank integration
                                # This checks the bank for instant delivery, then generates remaining via LLM
                                # New questions are saved to the bank in the background for future reuse
                                from services.quiz_with_bank import stream_quiz_with_bank as stream_quiz_questions

                                # Track questions for final save
                                all_questions = []

                                # Stream questions one by one (with optional empathetic message)
                                async for chunk in stream_quiz_questions(
                                    topic=metadata.get("topic"),
                                    difficulty=metadata.get("difficulty"),
                                    num_questions=metadata.get("num_questions"),
                                    source=metadata.get("source"),
                                    session=self.session,
                                    empathetic_message=empathetic_message,
                                    chat_id=self.session.chat_id,
                                    question_types=metadata.get("question_types", ["mcq"]),
                                    quiz_mode=metadata.get("quiz_mode", "knowledge"),
                                    learning_objective=metadata.get("learning_objective", "general"),
                                    user_prompt=metadata.get("user_prompt")
                                ):
                                    # Handle empathetic message streaming
                                    if chunk.get("status") == "empathetic_message_start":
                                        print("💬 Empathetic message streaming started")
                                        yield json.dumps({
                                            "status": "empathetic_message_start",
                                            "type": "quiz",
                                            "message": chunk.get("message")
                                        }) + "\n"

                                    elif chunk.get("status") == "empathetic_message_chunk":
                                        # Stream empathetic message chunks
                                        yield json.dumps({
                                            "status": "empathetic_message_chunk",
                                            "type": "quiz",
                                            "chunk": chunk.get("chunk"),
                                            "progress": chunk.get("progress")
                                        }) + "\n"

                                    elif chunk.get("status") == "empathetic_message_complete":
                                        print("✅ Empathetic message complete")
                                        yield json.dumps({
                                            "status": "empathetic_message_complete",
                                            "type": "quiz",
                                            "full_message": chunk.get("full_message")
                                        }) + "\n"

                                    elif chunk.get("status") == "generating":

                                        value ={ "status": "quiz_generating",
                                            "current": chunk.get("current"),
                                            "type":"quiz",
                                            "total": chunk.get("total"),
                                            "message": f"Génération question {chunk.get('current')} sur {chunk.get('total')}..."}

                                        print("GENERATING", value)
                                        # Send progress update
                                        yield json.dumps(value) + "\n"

                                    elif chunk.get("status") == "question_ready":
                                        # Send complete question to frontend
                                        # Transform to StudyQuizCard-compatible format
                                        raw_question = chunk.get("question")
                                        question = transform_question_for_frontend(raw_question)
                                        all_questions.append(question)

                                        value = { "status": "quiz_question",
                                            "question": question,
                                            "type":"quiz",
                                            "index": chunk.get("index"),
                                            "total_so_far": len(all_questions)}

                                        print("READY",value)
                                        yield json.dumps(value) + "\n"

                                    elif chunk.get("status") == "quiz_complete":
                                        # Send completion signal with all questions
                                        yield json.dumps({
                                            "status": "quiz_complete",
                                            "type":"quiz",
                                            "quiz_data": all_questions,
                                            "total_generated": chunk.get("total_generated")
                                        }) + "\n"

                                # Persist generated quiz in session for deduplication on follow-up prompts
                                try:
                                    self.session.quizzes.append({
                                        "quiz_data": all_questions,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                except Exception as e:
                                    print(f"Failed to cache quiz in session for dedupe: {e}")

                                # Quiz is done - don't generate suggestions after quiz
                                return

                        elif tool_name == "generate_audio_content":
                            print("🎙️ Audio content tool triggered")
                            print(f"📋 Tool arguments: {tool_args}")

                            from tools.audio_tools import generate_audio_content
                            result = await generate_audio_content.ainvoke(tool_args)

                            if result.get("status") == "audio_options":
                                # User needs to select duration - show options
                                print("🎙️ Showing audio options to user")
                                yield json.dumps({
                                    "status": "audio_options",
                                    "topic": result.get("topic"),
                                    "intent": result.get("intent"),
                                    "style_name": result.get("style_name"),
                                    "style_description": result.get("style_description"),
                                    "durations": result.get("durations"),
                                    "default_duration": result.get("default_duration"),
                                    "message": result.get("message")
                                }) + "\n"
                                return

                            elif result.get("status") == "audio_generation_initiated":
                                # Duration selected - generate audio
                                print("🎙️ Starting audio generation")
                                metadata = result.get("metadata", {})

                                yield json.dumps({
                                    "status": "audio_generating",
                                    "message": f"Creating {metadata.get('style_name', 'audio')} about {metadata.get('topic')}...",
                                    "phase": "starting"
                                }) + "\n"

                                # Import and use audio generator
                                from services.audio_generator import AudioGenerator
                                generator = AudioGenerator(self.session)

                                # Get progress data if this is a progress report
                                progress_data = metadata.get("progress_data")

                                # Stream audio generation
                                async for chunk in generator.generate_audio_stream(
                                    topic=metadata.get("topic"),
                                    intent=metadata.get("intent"),
                                    duration=metadata.get("duration"),
                                    language=metadata.get("language", language),
                                    progress_data=progress_data
                                ):
                                    yield chunk

                                return

                            else:
                                # Error
                                yield json.dumps({
                                    "status": "error",
                                    "message": result.get("message", "Audio generation failed")
                                }) + "\n"
                                return

                    except Exception as e:
                        print(f"🔥 Tool execution error: {e}")
                        tool_results.append({"error": str(e)})
                        import traceback
                        traceback.print_exc()  
                    
                
                # Format and stream the tool results
                if tool_results:
                    print("ORCHESTRATOR USING A TOOL")
                    for result in tool_results:
                        if isinstance(result, dict):
                            if "quiz" in result:
                                # Handle quiz results
                                quiz_data = result["quiz"]
                                
                                if isinstance(quiz_data, dict) and "quiz" in quiz_data:
                                    questions = quiz_data["quiz"]
                                else:
                                    questions = quiz_data
                                
                                # Send quiz as structured JSON
                                yield json.dumps({
                                    "quiz_data": questions,
                                    "status": "quiz_generated"
                                }) + "\n"
                                
                            elif "html_content" in result:
                                print("getting html content in study sheet")
                                html = result["html_content"]
                                print("this is the html:", html)
                                yield json.dumps({
                                    "html": html,
                                    "status":"studysheet_generated"
                                })+ "\n"
                                
                            elif "context" in result:
                                # Handle document search results - STREAM THIS
                                context = result['context']
                                words = context.split()
                            
                                print("WORDS FOUND:", words)
                                # Ask LLM to answer based on the search results
                                prompt = f"""Based on this information {words} from the user's documents, 
                                            answer their question:\n\n{user_input}
                                            using the language {language} use spacing,fonts,line breaks, emojis 
                                            to make the content clear and easy to read"""
                                
                                searchresultmessages = [{"role": "user", "content": prompt}]
                                
                                print("search result streaming starting")
                                
                                print("🔍 Now trying streaming...")
                                try:
                                    # Add timeout to prevent infinite hanging
                                    import asyncio
                                    
                                    async def stream_with_timeout():
                                        async for chunk in self.llm.astream(searchresultmessages):
                                            print(f"🔍 GOT CHUNK: {chunk}")
                                            if hasattr(chunk, 'content') and chunk.content:
                                                print(f"🔍 YIELDING: {chunk.content}")
                                                yield json.dumps({
                                                    "answer_chunk": chunk.content
                                                }) + "\n"
                                    
                                    # Try with 30 second timeout
                                    async for chunk in stream_with_timeout():
                                        yield chunk
                                        
                                except asyncio.TimeoutError:
                                    print("🔍 STREAMING TIMED OUT!")
                                    yield json.dumps({"answer_chunk": "Streaming timed out"}) + "\n"
                                except Exception as e:
                                    print(f"🔍 STREAMING EXCEPTION: {e}")
                                    import traceback
                                    traceback.print_exc()
                                                                
                            elif "error" in result:
                                # Handle errors
                                yield json.dumps({
                                    "answer_chunk": f"Error: {result['error']}"
                                }) + "\n"
            else:
                print("===NO TOOLS USED, STRAIGHT RESPONSE=====")
                
                response_content = ""
            
                # Stream the response chunk by chunk
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        print(chunk.content)
                        # Yield each chunk properly formatted
                        yield json.dumps({
                            "answer_chunk":chunk.content
                        }) + "\n"
            
            # Add assistant response to history
            self.session.message_history.append({
                "role": "assistant",
                "content": response_content if 'response_content' in locals() else response.content,
                "timestamp": datetime.now().isoformat()
            })

            # ═══════════════════════════════════════════════════════════════════
            # DISABLED: Prompt suggestions generation
            # ═══════════════════════════════════════════════════════════════════
            # Commenting out to save tokens - analytics showed low usage.
            # This was calling gpt-4.1-mini after every response to generate
            # 4-5 suggested follow-up prompts, costing ~500-1000 tokens per message.
            # Can be re-enabled if needed by uncommenting the code below.
            # ═══════════════════════════════════════════════════════════════════
            # suggestions = await self._generate_dynamic_suggestions()
            #
            # print(f"PROMPTS SUGGESTIONS CREATED: {suggestions}")
            #
            # # send them to the user
            # if suggestions:
            #     yield json.dumps({
            #         "status": "suggested_prompts",
            #         "suggestions": suggestions
            #     }) + "\n"

            # Final response status
            yield json.dumps({
                "status": "complete"
            }) + "\n"
            
        except Exception as e:
            print("ERROR OCCURED DURING PROCESS MESSAGE",e)
            yield json.dumps({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            }) + "\n"

    
    def _create_system_prompt(self) -> str:
        """
        Create nursing-specific system prompt.

        OPTIMIZATION: This prompt has been condensed from ~400 lines to ~150 lines
        while preserving all essential routing logic. Key optimizations:
        1. Removed redundant examples and duplicated instructions
        2. Made quiz data conditional (only include when relevant)
        3. Removed inline conversation history (already in messages array)
        4. Condensed verbose guidelines into concise rules
        """
        # Only include quiz data if user might be asking about practice/weak areas
        quiz_context = self._get_quiz_context_if_needed()

        return f"""You are an AI nursing tutor helping students with clinical skills, quizzes, flashcards, and study materials.

CORE TOOLS (use these, never write content manually):
• generate_quiz_stream: For ANY quiz/question request (max 15 questions)
  - ALWAYS pass user_prompt=<exact user message verbatim> — used for accurate mode detection
  - quiz_mode="knowledge" (DEFAULT) for factual questions
  - quiz_mode="nclex" ONLY when user explicitly asks for NCLEX/clinical scenarios/judgment
  - source_preference="documents" when student has uploaded files
• generate_flashcards_stream: For flashcard requests (max 15 cards)
• generate_study_sheet_stream: When user explicitly asks for study sheet/guide
• search_documents: Search student's uploaded materials
• summarize_document: For document summaries
• generate_audio_content: For audio content (teach, summarize, deep_dive, simplify, progress)

CRITICAL RULES:
1. NEVER write questions as text - ALWAYS use generate_quiz_stream tool
2. If user has documents and asks for quiz → source_preference="documents"
3. If user asks for >15 items → inform them of limit, ask if 15 is okay
4. If user says "yes/okay/sure" → ACT IMMEDIATELY, don't ask again
5. Preserve topic language - extract topics in user's language
6. Respond in the SAME language as the user's current message

QUIZ MODE DEFAULTS:
• Default: quiz_mode="knowledge" (factual recall questions)
• Use quiz_mode="nclex" when user says: "NCLEX", "NCLEX-style", "clinical scenarios", "patient scenarios", "clinical judgment", "testing judgment"
• ALWAYS include user_prompt=<verbatim user message> in every generate_quiz_stream call

LEARNING OBJECTIVE DETECTION (pass learning_objective= in generate_quiz_stream):
Infer the student's intent from their message and set learning_objective accordingly:
• "exam_prep"   → user says: "exam", "test tomorrow", "NCLEX exam", "prepare for", "ready for my exam", "cramming"
• "weak_areas"  → user says: "struggling with", "don't understand", "weak on", "keep getting wrong", "confused about", "help me with"
• "first_review"→ user says: "just started", "first time", "new to", "learning for the first time", "beginner", "intro"
• "deep_dive"   → user says: "deep dive", "in depth", "really understand", "mechanisms", "pathophysiology", "why does"
• "quick_check" → user says: "quick", "just a few", "fast review", "check myself", "rapid", "briefly"
• "general"     → (default) no clear intent signal detected
RULE: If ambiguous, default to "general". Never ask the user to clarify their intent — infer it.

SMART BEHAVIOR:
• When user enters bare topic → search_documents first, don't auto-generate content
• For "practice weak areas" after quiz → analyze quiz data, use empathetic_message
• For meta-questions about quizzes ("why was that hard?") → answer normally, don't generate new quiz

SESSION CONTEXT:
• Documents: {"YES - " + str(len(self.session.documents)) + " files" if self.session.documents else "none"}
• Last file: {self.session.documents[-1]["filename"] if self.session.documents else "none"}
• Language: {self.session.user_language or "auto-detect"}

{self._get_last_activity_summary()}

{quiz_context}

FILE INSIGHTS:
{self._format_file_insights()}

EMPATHETIC MESSAGE RULES (only for post-quiz practice):
• MAX 2 sentences, ~50 words, casual/friendly tone
• Include actual score and specific weak topics
• Avoid robotic phrases: "I understand", "You've got this", "Let's tackle this together"
• Tone by performance: <50% warm/understanding, 50-70% encouraging, 70-85% praise+push, 85%+ celebrate+challenge"""

    def _get_quiz_context_if_needed(self) -> str:
        """
        Return quiz context only when it might be needed.

        OPTIMIZATION: Previously, full quiz JSON was included on EVERY message,
        wasting tokens. Now we only include it when the context suggests
        the user might want to practice weak areas or review their quiz.
        """
        # Check if we have any quizzes at all
        if not self.session.quizzes or len(self.session.quizzes) == 0:
            return ""

        # Get a summary instead of full JSON (much smaller)
        quiz_summary = self._format_last_quiz_summary()

        if quiz_summary == "No quiz completed yet":
            return ""

        return f"""LAST QUIZ SUMMARY (for practice recommendations):
{quiz_summary}

Note: For detailed quiz analysis, the full quiz data is available via _format_last_quiz_for_extraction()."""

    async def load_file_insights_from_firebase(self):
        """
        Load file insights from upload_loading messages stored in Firebase.
        These insights were saved when files were uploaded.
        """
        try:
            from firebase_admin import firestore
            db = firestore.client()
            
            chat_id = self.session.chat_id
            
            # Get all messages for this chat
            messages_ref = db.collection("chats").document(chat_id).collection("messages")
            
            # Query for upload_loading messages that have insights
            query = messages_ref.where("type", "==", "upload_loading")
            docs = query.stream()
            
            # Initialize file_insights dict
            if not hasattr(self.session, 'file_insights'):
                self.session.file_insights = {}
            
            insights_loaded = 0
            
            for doc in docs:
                data = doc.to_dict()
                insights_list = data.get('insights', [])
                
                if insights_list:
                    for insight in insights_list:
                        filename = insight.get('filename')
                        if filename:
                            self.session.file_insights[filename] = {
                                'topics': insight.get('topics', []),
                                'concepts': insight.get('concepts', []),
                                'document_type': insight.get('documentType', 'unknown')
                            }
                            insights_loaded += 1
            
            if insights_loaded > 0:
                print(f"✅ Loaded {insights_loaded} file insights from Firebase")
                print(f"   Files: {list(self.session.file_insights.keys())}")
            else:
                print(f"📝 No file insights found in Firebase for {chat_id}")
                
        except Exception as e:
            print(f"⚠️ Failed to load file insights from Firebase: {e}")
            import traceback
            traceback.print_exc()
       
    def _format_last_quiz_for_extraction(self) -> str:
        """
        Format last quiz data for LLM extraction.

        OPTIMIZATION: Uses pre-fetched session.quizzes instead of making a
        synchronous Firebase query on every message. The quiz data was already
        loaded by get_chat_context_from_db() and stored in self.session.quizzes.

        This eliminates ~300-800ms of blocking I/O per message.
        """
        try:
            # Use pre-fetched quizzes from session (loaded by get_chat_context_from_db)
            if not self.session.quizzes or len(self.session.quizzes) == 0:
                print(f"📝 No quizzes in session for {self.session.chat_id}")
                return "null (no quiz completed yet)"

            # Get the most recent quiz (last item in the list)
            last_quiz_entry = self.session.quizzes[-1]
            print(f"✅ Using pre-fetched quiz from session (total quizzes: {len(self.session.quizzes)})")

            # Extract quiz_data from the entry (format from get_chat_context_from_db)
            quiz_data = last_quiz_entry.get('quiz_data')

            if quiz_data is None:
                print("📝 Quiz entry found but has no quiz_data")
                return "null (last quiz had no data)"

            # Handle both formats:
            # 1. Array directly: [{ question: ..., options: ..., answer: ..., userSelection: {...} }]
            # 2. Object with questions key: { questions: [...] }
            questions = None
            if isinstance(quiz_data, list):
                # Frontend format: quizData is array directly
                questions = quiz_data
                print(f"🔍 Quiz data is array (length: {len(questions)})")
            elif isinstance(quiz_data, dict):
                # Could be { questions: [...] } or direct question object
                if 'questions' in quiz_data:
                    questions = quiz_data.get('questions', [])
                    print(f"🔍 Quiz data is dict with questions key (length: {len(questions)})")
                else:
                    # Might be the questions array wrapped differently
                    questions = quiz_data.get('quiz', []) or []
                    print(f"🔍 Quiz data is dict, checking 'quiz' key (length: {len(questions)})")

            if not questions:
                print("📝 Quiz found but has no questions")
                return "null (last quiz had no questions)"

            print(f"✅ Loaded last quiz from session: {len(questions)} questions")

            # Print first question for verification
            if questions and len(questions) > 0:
                first_q = questions[0]
                q_text = first_q.get('question', 'N/A') if isinstance(first_q, dict) else str(first_q)
                print(f"🔍 First question preview: {q_text[:100]}...")

            # Clean questions data to remove Firebase-specific types (DatetimeWithNanoseconds, etc.)
            def clean_firebase_data(obj):
                """Recursively clean Firebase objects to make them JSON serializable"""
                if isinstance(obj, dict):
                    return {k: clean_firebase_data(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_firebase_data(item) for item in obj]
                elif hasattr(obj, 'isoformat'):  # DatetimeWithNanoseconds, datetime, etc.
                    return obj.isoformat()
                else:
                    return obj

            cleaned_questions = clean_firebase_data(questions)

            # Return complete quiz data as JSON string for LLM to extract
            quiz_json = json.dumps({"questions": cleaned_questions}, indent=2)
            print(f"📤 Returning quiz JSON (length: {len(quiz_json)} chars)")
            return quiz_json

        except Exception as e:
            print(f"❌ Error formatting quiz from session: {e}")
            import traceback
            traceback.print_exc()
            return "null (error formatting quiz data)"

    def _format_last_quiz_summary(self) -> str:
        """Format last quiz summary for LLM analysis - lightweight version"""
        if not self.session.quizzes or len(self.session.quizzes) == 0:
            return "No quiz completed yet"

        try:
            # Get only the last quiz
            last_quiz = self.session.quizzes[-1]

            # Handle different quiz entry formats
            if isinstance(last_quiz, dict):
                quiz_data = last_quiz.get('quiz_data', {})
            else:
                # Unexpected format
                return "Last quiz format not recognized"

            # Handle both quiz_data formats:
            # 1. List directly: [{ question: ..., options: ... }]
            # 2. Dict with questions key: { questions: [...] }
            if isinstance(quiz_data, list):
                questions = quiz_data
            elif isinstance(quiz_data, dict):
                questions = quiz_data.get('questions', [])
            else:
                questions = []

            if not questions:
                return "Last quiz had no questions"

            # Analyze performance
            total = len(questions)
            correct = sum(1 for q in questions if isinstance(q, dict) and q.get('userSelection', {}).get('isCorrect', False))
            incorrect = total - correct
            percentage = round((correct / total) * 100) if total > 0 else 0

            # Extract topics and performance
            topic_performance = {}
            for q in questions:
                if not isinstance(q, dict):
                    continue
                topic = q.get('topic', 'General')
                if topic not in topic_performance:
                    topic_performance[topic] = {'total': 0, 'correct': 0}
                topic_performance[topic]['total'] += 1
                if q.get('userSelection', {}).get('isCorrect', False):
                    topic_performance[topic]['correct'] += 1

            # Format topic breakdown
            topic_breakdown = []
            weak_topics = []
            for topic, perf in topic_performance.items():
                topic_pct = round((perf['correct'] / perf['total']) * 100) if perf['total'] > 0 else 0
                topic_breakdown.append(f"{topic}: {perf['correct']}/{perf['total']} ({topic_pct}%)")
                if topic_pct < 60:
                    weak_topics.append(topic)

            summary = f"Score: {correct}/{total} ({percentage}%)"
            if topic_breakdown:
                summary += f" | Topics: {', '.join(topic_breakdown)}"
            if weak_topics:
                summary += f" | Weak areas: {', '.join(weak_topics)}"

            return summary

        except Exception as e:
            print(f"⚠️ Error in _format_last_quiz_summary: {e}")
            return "Error analyzing quiz data"

    def _format_file_insights(self) -> str:
        """Format file insights for inclusion in system prompt"""
        file_insights = getattr(self.session, 'file_insights', {})
        
        if not file_insights:
            return "No file insights available yet."
        
        formatted = []
        for filename, insights in file_insights.items():
            if not insights:
                continue
                
            # Get short filename (remove chat ID prefix)
            short_name = filename.split('_uploads_')[-1] if '_uploads_' in filename else filename
            
            topics = insights.get('topics', [])
            concepts = insights.get('concepts', [])
            doc_type = insights.get('document_type', 'unknown')
            
            insight_str = f"• {short_name} ({doc_type})"
            if topics:
                insight_str += f"\n  Topics: {', '.join(topics[:3])}"
            if concepts:
                insight_str += f"\n  Key concepts: {', '.join(concepts[:5])}"
            
            formatted.append(insight_str)
        
        if not formatted:
            return "No file insights available yet."

        return "\n".join(formatted)

    # =========================================================================
    # CONTEXT-AWARE INTENT DETECTION
    # These methods help the LLM understand what just happened in the conversation
    # so it can better interpret short messages like "more", "again", etc.
    # =========================================================================

    def _get_last_activity_summary(self) -> str:
        """
        Create a summary of the user's last activity.

        This helps the LLM understand WHAT JUST HAPPENED so it can interpret
        short messages like "more", "again", "another" correctly.

        Returns:
            A formatted string describing the last activity, or "No recent activity"

        Example output:
            "Type: QUIZ_COMPLETED
             Topic: allergy management
             Score: 100% (1/1 correct)
             Status: User just finished this quiz"
        """
        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Check if there was a recent quiz
            # ─────────────────────────────────────────────────────────────────
            if self.session.quizzes and len(self.session.quizzes) > 0:

                # Get the most recent quiz
                last_quiz = self.session.quizzes[-1]

                # Handle different quiz entry formats
                if not isinstance(last_quiz, dict):
                    return "No recent quiz or flashcard activity"

                # Extract quiz data (handle different formats)
                quiz_data = last_quiz.get('quiz_data', {})

                # Get questions list - handle both formats
                if isinstance(quiz_data, list):
                    # Format: quizData is the array directly
                    questions = quiz_data
                elif isinstance(quiz_data, dict):
                    # Format: quizData has 'questions' key
                    questions = quiz_data.get('questions', [])
                else:
                    questions = []

                # If we have questions, analyze them
                if questions and len(questions) > 0:

                    # ─────────────────────────────────────────────────────────
                    # STEP 2: Calculate score from quiz
                    # ─────────────────────────────────────────────────────────
                    total_questions = len(questions)
                    correct_answers = 0
                    topics_covered = set()

                    for question in questions:
                        # Skip non-dict items
                        if not isinstance(question, dict):
                            continue

                        # Check if user answered correctly
                        user_selection = question.get('userSelection', {})
                        if isinstance(user_selection, dict) and user_selection.get('isCorrect', False):
                            correct_answers += 1

                        # Collect topics
                        topic = question.get('topic', '')
                        if topic:
                            topics_covered.add(topic)

                    # Calculate percentage
                    if total_questions > 0:
                        percentage = round((correct_answers / total_questions) * 100)
                    else:
                        percentage = 0

                    # ─────────────────────────────────────────────────────────
                    # STEP 3: Build the summary string
                    # ─────────────────────────────────────────────────────────
                    topics_str = ', '.join(topics_covered) if topics_covered else 'General'

                    summary = f"""
Type: QUIZ_COMPLETED
Topic(s): {topics_str}
Score: {percentage}% ({correct_answers}/{total_questions} correct)
Status: User just finished this quiz

IMPORTANT: If user now says "more", "again", "another", "continue", "next":
→ They want ANOTHER QUIZ on the same topic ({topics_str})
→ Use generate_quiz_stream tool immediately!
"""
                    return summary.strip()

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Check for other activities (flashcards, study sheets)
            # ─────────────────────────────────────────────────────────────────

            # Check for flashcards (if you track them in session)
            if hasattr(self.session, 'last_flashcards') and self.session.last_flashcards:
                return """
Type: FLASHCARDS_CREATED
Status: User just reviewed flashcards

IMPORTANT: If user now says "more", "again", "another":
→ They want MORE FLASHCARDS on the same topic
→ Use generate_flashcards_stream tool immediately!
""".strip()

            # ─────────────────────────────────────────────────────────────────
            # STEP 5: No recent activity found
            # ─────────────────────────────────────────────────────────────────
            return "No recent quiz or flashcard activity"

        except Exception as e:
            print(f"⚠️ Error getting last activity summary: {e}")
            return "Unable to determine last activity"

    def _is_continuation_request(self, user_input: str) -> dict:
        """
        Check if the user's message is a "continuation" request.

        A continuation request is when the user says something short like
        "more", "again", "another" - meaning they want more of whatever
        just happened (quiz, flashcards, etc.)

        Args:
            user_input: The user's message

        Returns:
            A dictionary with:
            - is_continuation: True/False
            - action_type: 'quiz', 'flashcard', or None
            - topic: The topic to continue with, or None

        Example:
            Input: "more"
            Output: {'is_continuation': True, 'action_type': 'quiz', 'topic': 'allergy management'}
        """
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Define continuation words (English and French)
        # ─────────────────────────────────────────────────────────────────────
        CONTINUATION_WORDS_EN = [
            'more',           # "more" - most common
            'again',          # "again" - repeat
            'another',        # "another one"
            'another one',    # explicit
            'one more',       # "one more"
            'continue',       # "continue"
            'keep going',     # "keep going"
            'next',           # "next"
            'go on',          # "go on"
            'yes',            # "yes" (after quiz offer)
            'sure',           # "sure"
            'okay',           # "okay"
            'ok',             # "ok"
            'yep',            # "yep"
            'yeah',           # "yeah"
            'do it',          # "do it"
            'lets go',        # "let's go"
            "let's go",       # with apostrophe
            'challenge me',   # after seeing "Challenge Me More" button
        ]

        CONTINUATION_WORDS_FR = [
            'encore',         # "encore" - again/more
            'plus',           # "plus" - more
            'un autre',       # "un autre" - another one
            'une autre',      # "une autre" - another one (feminine)
            'continuer',      # "continuer" - continue
            'suite',          # "suite" - next
            'suivant',        # "suivant" - next
            'oui',            # "oui" - yes
            'ouais',          # "ouais" - yeah
            'd accord',       # "d'accord" - okay
            "d'accord",       # with apostrophe
            'ok',             # same in French
            'vas-y',          # "vas-y" - go ahead
            'allez',          # "allez" - let's go
        ]

        # Combine all continuation words
        ALL_CONTINUATION_WORDS = CONTINUATION_WORDS_EN + CONTINUATION_WORDS_FR

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Normalize the user input
        # ─────────────────────────────────────────────────────────────────────
        normalized_input = user_input.lower().strip()

        # Remove punctuation for comparison
        import re
        normalized_input_clean = re.sub(r'[^\w\s]', '', normalized_input)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Check if message is short AND contains continuation word
        # ─────────────────────────────────────────────────────────────────────

        # Only treat as continuation if message is short (max 5 words)
        word_count = len(normalized_input_clean.split())

        if word_count > 5:
            # Message too long - probably not a simple continuation
            return {'is_continuation': False, 'action_type': None, 'topic': None}

        # Check if any continuation word matches
        is_continuation = False
        for word in ALL_CONTINUATION_WORDS:
            if word in normalized_input_clean or normalized_input_clean == word:
                is_continuation = True
                break

        if not is_continuation:
            return {'is_continuation': False, 'action_type': None, 'topic': None}

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Determine what action to continue (quiz, flashcard, etc.)
        # ─────────────────────────────────────────────────────────────────────
        action_type = None
        topic = None

        # Check if there was a recent quiz
        if self.session.quizzes and len(self.session.quizzes) > 0:
            action_type = 'quiz'

            # Get the topic from the last quiz
            last_quiz = self.session.quizzes[-1]

            # Handle different entry formats
            if isinstance(last_quiz, dict):
                quiz_data = last_quiz.get('quiz_data', {})
            else:
                quiz_data = {}

            # Extract questions - handle both formats
            if isinstance(quiz_data, list):
                questions = quiz_data
            elif isinstance(quiz_data, dict):
                questions = quiz_data.get('questions', [])
            else:
                questions = []

            # Get topics from questions
            if questions:
                topics = set()
                for q in questions:
                    if isinstance(q, dict):
                        t = q.get('topic', '')
                        if t:
                            topics.add(t)
                if topics:
                    topic = ', '.join(topics)

        # Check for flashcards (if tracked)
        elif hasattr(self.session, 'last_flashcards') and self.session.last_flashcards:
            action_type = 'flashcard'
            topic = getattr(self.session, 'last_flashcard_topic', None)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Return the result
        # ─────────────────────────────────────────────────────────────────────
        return {
            'is_continuation': True,
            'action_type': action_type,
            'topic': topic
        }

    def _transform_continuation_message(self, user_input: str, continuation_info: dict) -> str:
        """
        Transform a short continuation message into an explicit request.

        This helps the LLM understand exactly what the user wants.

        Args:
            user_input: Original message like "more"
            continuation_info: Result from _is_continuation_request()

        Returns:
            Transformed message with explicit intent

        Example:
            Input: "more", {'action_type': 'quiz', 'topic': 'allergies'}
            Output: "[CONTINUATION REQUEST: User wants another QUIZ on topic: allergies]
                     Original message: more"
        """
        action_type = continuation_info.get('action_type')
        topic = continuation_info.get('topic')

        if action_type == 'quiz':
            if topic:
                return f"""[CONTINUATION REQUEST: User wants ANOTHER QUIZ on the same topic.
Topic: {topic}
Action: Call generate_quiz_stream tool immediately with topic="{topic}"
Do NOT ask for confirmation - just generate the quiz!]

Original message: {user_input}"""
            else:
                return f"""[CONTINUATION REQUEST: User wants ANOTHER QUIZ.
Action: Call generate_quiz_stream tool immediately
Do NOT ask for confirmation - just generate the quiz!]

Original message: {user_input}"""

        elif action_type == 'flashcard':
            if topic:
                return f"""[CONTINUATION REQUEST: User wants MORE FLASHCARDS on the same topic.
Topic: {topic}
Action: Call generate_flashcards_stream tool immediately with topic="{topic}"
Do NOT ask for confirmation - just generate the flashcards!]

Original message: {user_input}"""
            else:
                return f"""[CONTINUATION REQUEST: User wants MORE FLASHCARDS.
Action: Call generate_flashcards_stream tool immediately
Do NOT ask for confirmation - just generate the flashcards!]

Original message: {user_input}"""

        # No specific action detected - return original
        return user_input

    # =========================================================================
    # END OF CONTEXT-AWARE INTENT DETECTION
    # =========================================================================

    # Streaming function that works with chunks
    @staticmethod
    async def stream_document_summary(
        relevant_chunks: list, 
        detail_level: str,
        filename: str,
        language: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream summary generation from relevant vector store chunks
        """
        try:
            # Combine relevant chunks into context
            context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
            
            # Limit context size for LLM - allow more for comprehensive requests
            limits = {
                "brief": 20000,
                "detailed": 40000,
                "comprehensive": 80000
            }
            max_chars = limits.get(detail_level, 40000)
            if len(context) > max_chars:
                context = context[:max_chars] + "..."
            
            detail_instructions = {
                "brief": "Create a concise 3-4 paragraph summary focusing on main concepts only",
                "detailed": "Create a comprehensive summary with 6-10 paragraphs covering key concepts, clinical applications, and nursing considerations", 
                "comprehensive": "Create an in-depth analysis with detailed explanations, examples, and extensive nursing applications; include pharmacology, pathophysiology, and nursing implications when relevant"
            }
            
            summary_prompt = f"""
                You are an AI assistant that creates accurate summaries of documents. 

                CRITICAL: Summarize the ACTUAL content provided, not hypothetical content.

                If the document is about nursing/medical topics, focus on:
                - Key nursing concepts and terminology
                - Clinical applications and patient care
                - NCLEX-relevant information

                If the document is about other topics , summarize what it actually contains:
                - Main topics and themes
                - Key information and details
                - Relevant highlights

                Instructions: {detail_instructions.get(detail_level, detail_instructions['detailed'])}

                Document Excerpts from {filename}: 
                {context}
                
                Important: Use emojis, space, bold, etc to organize the text and make it easy to digest. 
                Adapt it for dyslexic people to be easy to read and understand
                
                Important: Use must write the content in the language {language}

                Create an accurate summary of the ACTUAL content above (do not create hypothetical nursing content):
                """
            
            # Summarization is mini's strong suit; comprehensive summaries can be ~30k input tokens.
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, streaming=True)
            
            async for chunk in llm.astream([{"role": "user", "content": summary_prompt}]):
                if hasattr(chunk, 'content') and chunk.content:
                     # ✅ Just the text!
                     yield chunk.content 

                    
        except Exception as e:
            yield f"Error generating summary: {str(e)}"
            
    async def _generate_dynamic_suggestions(self) -> list:
        """
        Generate pedagogically intelligent, high-impact suggestions that:
        1. Are outcome-oriented and goal-focused
        2. Only reference available resources
        3. Form a coherent learning pathway
        """
        try:
            # ========================================
            # PHASE 1: Build Rich Context
            # ========================================
            recent_msgs = self.session.message_history[-8:]
            
            # Get conversation context
            context_snippet = "\n".join([
                f"{m['role']}: {m['content'][:150]}" 
                for m in recent_msgs if 'content' in m
            ])[-1000:]
            
            # Analyze tool usage
            last_tools = [t.get("tool") for t in getattr(self.session, "tool_calls", [])[-3:]]
            
            # Assess available resources
            has_documents = bool(self.session.documents)
            document_names = [doc.get("filename", "") for doc in (self.session.documents or [])]
            
            has_quiz_history = bool(getattr(self.session, "quizzes", []))
            quiz_performance = self._analyze_quiz_performance() if has_quiz_history else None
            
            recent_topics = self._extract_recent_quiz_topics() if has_quiz_history else []
            
            # Extract current topic from last message
            last_msg = recent_msgs[-1].get("content", "") if recent_msgs else ""
            current_topic = self._extract_current_topic(last_msg, recent_topics)
            
            # ========================================
            # PHASE 2: Create High-Impact System Prompt
            # ========================================
            
            # Build resource-aware tool list
            available_tools = self._build_available_tools_list(has_documents)
            
            system_prompt = f"""You are a nursing education AI helping students achieve specific learning outcomes.

            Your role: Suggest 4-5 HIGH-IMPACT next actions that:
            - Are goal-oriented and outcome-focused
            - Lead to measurable learning gains
            - Form a coherent progression toward mastery
            - Are immediately actionable

            {available_tools}

            **CRITICAL RULES:**

            1. **Be Specific and Outcome-Oriented**
            ❌ BAD: "Search my notes for information"
            ✅ GOOD: "What are the 3 main causes of acute respiratory distress?"
            
            ❌ BAD: "Quiz me on cardiac care"
            ✅ GOOD: "Test my ability to identify arrhythmias from ECG patterns"
            
            ❌ BAD: "Create a study guide"
            ✅ GOOD: "Build a step-by-step guide for performing wound assessments"

            2. **Focus on Learning Goals**
            - Identify what the student needs to MASTER
            - Target specific skills, not vague topics
            - Include clinical application when relevant

            3. **Make It Action-Oriented**
            Each suggestion should clearly state:
            - What skill/knowledge will be gained
            - What specific outcome to expect
            - How it advances their competency

            4. **Pedagogical Strategy**
            - After explanation → Test understanding with specific scenarios
            - After quiz → Review weak areas with targeted questions
            - After reading → Apply knowledge to clinical situations
            - Mix recall and application questions

            **Response Format:**
            Return ONLY a JSON array of 4-5 strings in {self.session.user_language}.
            Each must be specific, actionable, and outcome-focused.

            Examples of HIGH-IMPACT suggestions:

            English:
            [
            "Test my ability to calculate dopamine drip rates for different patient weights",
            "What are the priority nursing interventions for a patient in septic shock?",
            "Quiz me on identifying heart failure vs COPD based on assessment findings",
            "Walk me through the step-by-step process of inserting a Foley catheter"
            ]

            French:
            [
            "Teste ma capacité à calculer les débits de perfusion de dopamine selon le poids",
            "Quelles sont les interventions infirmières prioritaires en cas de choc septique?",
            "Fais-moi un quiz pour différencier l'insuffisance cardiaque de la MPOC",
            "Guide-moi étape par étape dans l'insertion d'une sonde vésicale"
            ]

            Spanish:
            [
            "Prueba mi capacidad para calcular tasas de goteo de dopamina según peso",
            "¿Cuáles son las intervenciones prioritarias en shock séptico?",
            "Hazme un quiz para diferenciar insuficiencia cardíaca de EPOC",
            "Guíame paso a paso en la inserción de sonda Foley"
            ]
            """

            # ========================================
            # PHASE 3: Build Context Message
            # ========================================
            
            context_parts = [f"**Recent Conversation:**\n{context_snippet}"]
            
            if current_topic:
                context_parts.append(f"\n**Current Topic:** {current_topic}")
            
            if last_tools:
                context_parts.append(f"\n**Recent Actions:** {', '.join(last_tools)}")
            
            if has_documents:
                context_parts.append(f"\n**Student Has Uploaded:** {', '.join(document_names[:3])}")
            
            if quiz_performance:
                context_parts.append(f"\n**Recent Quiz Results:** {quiz_performance}")
            
            if recent_topics:
                context_parts.append(f"\n**Recently Studied:** {', '.join(recent_topics)}")
            
            context_message = "\n".join(context_parts)
            
            # ========================================
            # PHASE 4: Generate Suggestions
            # ========================================
            
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
            
            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""{context_message}

    Generate 4-5 HIGH-IMPACT, outcome-oriented suggestions that will most effectively advance this student's learning.

    Focus on:
    - Specific skills or knowledge gaps to address
    - Clinical application of concepts just discussed
    - Progressive complexity based on their performance
    - Actionable next steps with clear learning outcomes

    Remember: Be specific, goal-oriented, and immediately useful."""}
            ])
            
            # ========================================
            # PHASE 5: Parse and Return
            # ========================================
            
            import json
            
            try:
                suggestions = json.loads(response.content)
            except json.JSONDecodeError:
                print(f"⚠️ JSON parse failed, attempting cleanup")
                cleaned = response.content.strip().strip("```json").strip("```").strip()
                try:
                    suggestions = json.loads(cleaned)
                except:
                    print(f"⚠️ JSON parsing failed, extracting lines")
                    lines = [line.strip().strip('"-,[]') for line in response.content.split('\n') if line.strip()]
                    suggestions = [line for line in lines if len(line) > 15 and not line.startswith('{')][:5]
            
            if not isinstance(suggestions, list):
                print(f"⚠️ Response not a list: {suggestions}")
                return []
            
            # Cleanup: remove empty/short strings
            cleaned_suggestions = [
                s.strip() for s in suggestions 
                if isinstance(s, str) and len(s.strip()) > 15  # Longer minimum for quality
            ][:5]
            
            # Store for context
            self.session.last_suggestions = cleaned_suggestions
            
            return cleaned_suggestions
            
        except Exception as e:
            print(f"⚠️ Suggestion generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []


    def _build_available_tools_list(self, has_documents: bool) -> str:
        """
        Build a resource-aware list of available tools.
        Only mention document-related tools if student has uploaded files.
        """
        
        if has_documents:
            return """**Available Tools (use ONLY these):**

    1. **search_documents** - Find specific information in uploaded materials
    Use for: "What does my textbook say about [specific clinical question]?"
    Example: "How do my notes explain the pathophysiology of diabetic ketoacidosis?"

    2. **generate_quiz** - Test knowledge with specific questions
    Use for: "Test my ability to [specific skill/knowledge]"
    Example: "Quiz me on differentiating types of shock based on hemodynamic parameters"

    3. **summarize_document** - Extract key information from uploaded files
    Use for: "What are the main takeaways about [topic] from my document?"
    Example: "Summarize the priority interventions for stroke patients from my notes"

    4. **generate_study_sheet** - Create comprehensive learning guides
    Use for: "Build a step-by-step guide for [clinical skill/concept]"
    Example: "Create a systematic approach to respiratory assessment"

    5. **ask_questions** - Get detailed explanations on specific topics
    Use for: "Explain [specific concept] with clinical examples"
    Example: "What are the key differences between Type 1 and Type 2 respiratory failure?"
    """
        else:
            return """**Available Tools (use ONLY these):**

    1. **generate_quiz** - Test knowledge with specific questions
    Use for: "Test my ability to [specific skill/knowledge]"
    Example: "Quiz me on differentiating types of shock based on hemodynamic parameters"

    2. **generate_study_sheet** - Create comprehensive learning guides
    Use for: "Build a step-by-step guide for [clinical skill/concept]"
    Example: "Create a systematic approach to respiratory assessment"

    3. **ask_questions** - Get detailed explanations on specific topics
    Use for: "Explain [specific concept] with clinical examples"
    Example: "What are the key differences between Type 1 and Type 2 respiratory failure?"

    NOTE: Student has NOT uploaded any documents yet. DO NOT suggest searching or summarizing files.
    """


    def _extract_current_topic(self, last_msg: str, recent_topics: list) -> str:
        """
        Extract the main topic being discussed.
        Priority: last message > recent quiz topics
        """
        
        # Try to extract from last message
        msg_lower = last_msg.lower()
        
        # Medical keywords that might indicate topic
        medical_terms = [
            "cardiac", "respiratory", "renal", "neuro", "diabetes", "sepsis",
            "shock", "heart failure", "COPD", "asthma", "pneumonia",
            "medications", "pharmacology", "dosage", "IV", "catheter"
        ]
        
        for term in medical_terms:
            if term in msg_lower:
                return term
        
        # Fallback to recent quiz topics
        if recent_topics:
            return recent_topics[0]
        
        # Extract from question patterns
        if "what" in msg_lower or "how" in msg_lower or "explain" in msg_lower:
            # Try to grab the noun phrase after the question word
            words = last_msg.split()
            if len(words) > 3:
                return " ".join(words[-5:])  # Last few words often contain the topic
        
        return None


    def _analyze_quiz_performance(self) -> str:
        """Analyze recent quiz results to inform suggestions."""
        try:
            recent_quizzes = getattr(self.session, "quizzes", [])[-3:]
            
            if not recent_quizzes:
                return None
            
            total_questions = 0
            total_correct = 0
            weak_topics = []
            
            for quiz in recent_quizzes:
                quiz_data = quiz.get("quiz_data", {})
                
                if isinstance(quiz_data, dict):
                    questions = quiz_data.get("quiz", [])
                else:
                    questions = quiz_data
                
                for q in questions:
                    if not isinstance(q, dict):
                        continue
                    
                    total_questions += 1
                    
                    user_sel = q.get("userSelection", {})
                    if user_sel.get("isCorrect"):
                        total_correct += 1
                    else:
                        topic = q.get("metadata", {}).get("topic", "")
                        if topic and topic not in weak_topics:
                            weak_topics.append(topic)
            
            if total_questions == 0:
                return None
            
            accuracy = int((total_correct / total_questions) * 100)
            summary = f"{accuracy}% accuracy ({total_correct}/{total_questions})"
            
            if weak_topics:
                summary += f", weak areas: {', '.join(weak_topics[:2])}"
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Quiz analysis failed: {e}")
            return None


    def _extract_recent_quiz_topics(self) -> list:
        """Extract topics from recent quizzes."""
        try:
            recent_quizzes = getattr(self.session, "quizzes", [])[-2:]
            topics = set()
            
            for quiz in recent_quizzes:
                quiz_data = quiz.get("quiz_data", {})
                
                if isinstance(quiz_data, dict):
                    questions = quiz_data.get("quiz", [])
                else:
                    questions = quiz_data
                
                for q in questions:
                    if isinstance(q, dict):
                        topic = q.get("metadata", {}).get("topic", "")
                        if topic:
                            topics.add(topic)
            
            return list(topics)[:3]
            
        except Exception as e:
            print(f"⚠️ Topic extraction failed: {e}")
            return []
    
    
async def generate_post_upload_suggestions(session, file_insights: dict) -> list:
        """
        Generate intelligent, file-content-aware suggestions after upload.
        Follows same high-impact principles as _generate_dynamic_suggestions.
        
        Args:
            session: NursingTutor session with user_language
            file_insights: Dict of {filename: {topics, concepts, document_type}}
        
        Returns:
            List of 4-5 outcome-oriented suggestions
        """
        try:
            if not file_insights:
                return []
            
            # Aggregate insights from all files
            all_topics = []
            all_concepts = []
            doc_types = []
            filenames = list(file_insights.keys())
            
            for filename, insights in file_insights.items():
                if insights:
                    all_topics.extend(insights.get("topics", []))
                    all_concepts.extend(insights.get("concepts", []))
                    doc_type = insights.get("document_type", "")
                    if doc_type:
                        doc_types.append(doc_type)
            
            # Deduplicate
            unique_topics = list(set(all_topics))[:5]
            unique_concepts = list(set(all_concepts))[:10]
            
            # Build context
            context_parts = [
                f"**Files Just Uploaded:** {', '.join(filenames)}",
                f"**Main Topics:** {', '.join(unique_topics)}",
                f"**Key Concepts:** {', '.join(unique_concepts[:5])}"
            ]
            
            if doc_types:
                context_parts.append(f"**Document Types:** {', '.join(set(doc_types))}")
            
            context = "\n".join(context_parts)
            
            # Use same system prompt structure as _generate_dynamic_suggestions
            
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
            
            system_prompt = f"""You are a nursing education AI helping students achieve specific learning outcomes.

            The student just uploaded study materials. Generate 4-5 HIGH-IMPACT suggestions based on the file content.

            **Available Tools:**
            1. search_documents - Find specific information in uploaded materials
            2. generate_quiz - Test knowledge with specific questions
            3. summarize_document - Extract key information
            4. generate_study_sheet - Create comprehensive guides
            5. ask_questions - Get detailed explanations

            **CRITICAL RULES:**

            1. **Be Specific and Outcome-Oriented**
            ❌ BAD: "Search my notes for information"
            ✅ GOOD: "What are the 3 main causes of acute respiratory distress in my document?"
            
            ❌ BAD: "Quiz me on cardiac care"
            ✅ GOOD: "Test my ability to identify the 6 arrhythmia types covered in Chapter 3"
            
            ❌ BAD: "Summarize my document"
            ✅ GOOD: "What are the priority interventions for stroke patients from my notes?"

            2. **Reference Specific File Content**
            - Mention specific topics/concepts from the uploaded files
            - Target clinical skills or procedures mentioned in documents
            - Make suggestions immediately actionable

            3. **Focus on Learning Goals**
            - What skill/knowledge will be gained?
            - What specific outcome to expect?
            - How does it advance their competency?

            **Response Format:**
            Return ONLY a JSON array of 4-5 strings in {session.session.user_language}.

            Examples (English):
            [
            "Quiz me on the 5 types of shock covered in this document",
            "What are the key differences between the cardiac medications listed?",
            "Test my understanding of the ECG interpretation guidelines from my notes",
            "Summarize the priority nursing interventions for sepsis from my file"
            ]

            Examples (French):
            [
            "Fais-moi un quiz sur les 5 types de choc abordés dans ce document",
            "Quelles sont les différences clés entre les médicaments cardiaques listés?",
            "Teste ma compréhension des directives d'interprétation ECG de mes notes",
            "Résume les interventions infirmières prioritaires pour la septicémie"
            ]
            """

            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nGenerate 4-5 specific, outcome-oriented suggestions based on this uploaded content."}
            ])
            
            # Parse
            try:
                suggestions = json.loads(response.content.strip().strip("```json").strip("```"))
            except:
                # Fallback parsing
                print(f"⚠️ JSON parse failed, attempting line extraction")
                lines = [line.strip().strip('"-,[]') for line in response.content.split('\n') if line.strip()]
                suggestions = [line for line in lines if len(line) > 15 and not line.startswith('{')][:5]
            
            if isinstance(suggestions, list):
                return suggestions[:5]
            
            return []
            
        except Exception as e:
            print(f"⚠️ Post-upload suggestion generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []
