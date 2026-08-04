from langchain_openai import ChatOpenAI
from tools.quiztools import NursingTools, set_session_context
from models.session import PersistentSessionContext
from typing import AsyncGenerator, Optional, Tuple
from datetime import datetime
from tools.quiztools import search_documents,summarize_document,get_chat_context_from_db
from services.intent_analyzer import analyze_intent, should_skip_analysis
from services.urgency_detector import detect_urgency
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

        # Get properly decorated tools — keep a reference so we can rebind
        # with `tool_choice` later when fast routing has a confident pick.
        self.tools = self.tools_instance.get_tools()

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

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.routing_llm_with_tools = self.routing_llm.bind_tools(self.tools)

    # Patterns that indicate NO tool is needed (direct conversation)
    CONVERSATIONAL_PATTERNS = re.compile(
        r'^(hi|hello|hey|bonjour|salut|thanks|thank you|merci|ok|okay|sure|yes|no|'
        r'what is|what are|who is|why|how does|explain|tell me about|'
        r'can you explain|help me understand|i don\'t understand|'
        r'what do you mean|could you clarify)\b',
        re.IGNORECASE
    )

    # User is asking for an answer / complaining about non-response. Kept
    # deliberately broad: this only fires when the LAST assistant turn was a
    # generated quiz, and the fallback is a plain written answer — the safe
    # outcome either way.
    ANSWER_COMPLAINT_PATTERNS = re.compile(
        r'(answer\s+(my|the|this)\s+question|did\s*n[o\']t\s+answer|not\s+answer'
        r'|don\'?t\s+want\s+(a\s+)?(quiz|question|generated)|no\s+quiz'
        r'|i\s+(need|want)\s+(an?\s+)?answer|give\s+me\s+the\s+answer'
        r'|what(\'s|\s+is)\s+the\s+answer|answer\s*,?\s*please|please\s+answer'
        r'|why\s+(are\s+)?you\s+not\s+(respond|answer)|are\s+you\s+there'
        r'|r[ée]ponds?\s+([àa]\s+)?ma\s+question|je\s+veux\s+(la\s+|une\s+)?r[ée]ponse)',
        re.IGNORECASE
    )

    def _is_post_quiz_complaint(self, user_input: str) -> bool:
        """
        True when the previous assistant turn was a generated quiz and the
        user's new message is either a complaint asking for an answer or a
        near-repeat of what they already sent (their retry reflex when they
        feel ignored).
        """
        try:
            history = self.session.message_history or []
            last_assistant = next(
                (m for m in reversed(history)
                 if m.get('role') == 'assistant' and m.get('content')),
                None,
            )
            if not last_assistant:
                return False
            # Marker written by get_chat_context_from_db for quiz messages.
            if not str(last_assistant['content']).startswith('[App generated a'):
                return False

            if self.ANSWER_COMPLAINT_PATTERNS.search(user_input):
                return True

            # Near-repeat of the previous user message = retry, not a fresh ask.
            # The frontend persists the user's message before calling us, so an
            # entry AFTER the quiz marker may be an echo of the current message.
            # The message that provoked the quiz is the last user entry BEFORE
            # the marker — that's what a retry repeats.
            norm = lambda s: ' '.join(str(s).lower().split())
            a = norm(user_input)
            last_assistant_idx = max(
                i for i, m in enumerate(history)
                if m.get('role') == 'assistant' and m.get('content')
            )
            before_quiz = [
                norm(m['content']) for m in history[:last_assistant_idx]
                if m.get('role') == 'user' and isinstance(m.get('content'), str)
            ]
            if a and before_quiz:
                b = before_quiz[-1]
                if a == b or (len(a) > 40 and (a in b or b in a)):
                    return True
            return False
        except Exception as e:
            print(f"quiz-repeat guard failed open: {e}")
            return False

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

        # Order matters: more specific patterns first. The QUIZ regex matches
        # broad single words ("test", "exam", "nclex") that commonly appear
        # inside flashcard/study-sheet requests ("make flashcards to TEST me",
        # "give me a study sheet for the EXAM"). Checking QUIZ first would
        # mis-route those, so we check the specific patterns first and let
        # QUIZ catch only the residual cases.

        # Flashcards (specific keyword, no overlap with quiz vocab)
        if self.FLASHCARD_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected flashcard pattern")
            return "generate_flashcards_stream"

        # Study sheet (multi-word, very specific)
        if self.STUDYSHEET_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected study sheet pattern")
            return "generate_study_sheet_stream"

        # Summarize (specific verbs, but "résumé" also lives in STUDYSHEET —
        # checked after so French study-sheet phrasing wins where intended)
        if self.SUMMARIZE_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected summarize pattern")
            return "summarize_document"

        # Quiz / test / exam — broadest content pattern, checked last
        if self.QUIZ_PATTERNS.search(user_input):
            print(f"⚡ FAST ROUTE: Detected quiz pattern in '{user_input[:50]}...'")
            return "generate_quiz_stream"

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

            # ═══════════════════════════════════════════════════════════════════
            # STEP: EXAM-PRESSURE CHECK
            # ═══════════════════════════════════════════════════════════════════
            # Reads deadline / zero-baseline / distress signals out of the message
            # (services/urgency_detector.py). Durable facts are kept for the rest
            # of the session so later turns still know the exam is in 2 days —
            # without that, the deadline is forgotten on the very next message.
            #
            # A student in crisis is acknowledged and given a plan BEFORE any tool
            # can fire: throwing a quiz at someone who just said they know nothing
            # is the behaviour this exists to stop. Returning here also skips the
            # analyzer and research calls, which is the right trade on a turn we
            # have already decided will not generate content.
            # ═══════════════════════════════════════════════════════════════════
            urgency = self._merge_urgency(detect_urgency(user_input))
            self.session.urgency_context = urgency
            if urgency["signals"]:
                print(f"🚨 Urgency signals {urgency['signals']} → crisis={urgency['is_crisis']}")

            if urgency["is_crisis"] and not self.session.crisis_acknowledged:
                self.session.crisis_acknowledged = True

                acknowledgement = ""
                async for piece in self._stream_crisis_support(user_input, urgency, language):
                    acknowledgement += piece
                    yield json.dumps({"answer_chunk": piece}) + "\n"

                self.session.message_history.append({
                    "role": "assistant",
                    "content": acknowledgement,
                    "timestamp": datetime.now().isoformat()
                })

                suggestions = await self._generate_dynamic_suggestions()
                if suggestions:
                    yield json.dumps({
                        "status": "suggested_prompts",
                        "suggestions": suggestions
                    }) + "\n"

                yield json.dumps({"status": "complete"}) + "\n"
                return

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
            # STEP: INTENT ANALYSIS (Claude "think before generating")
            # ═══════════════════════════════════════════════════════════════════
            # Runs Claude Sonnet 4.6 with adaptive thinking + web_search before
            # the existing fast-route logic. Produces a structured classification
            # of intent, quality bar, document use, and (when applicable) a
            # one-sentence honesty preamble we stream before generation.
            #
            # Skipped for continuations and very short conversational messages
            # to keep the hot path fast.
            # ═══════════════════════════════════════════════════════════════════
            classification = None
            tool_args_overrides = {}
            honesty_preamble = None
            analyzer_recommended_tool = None

            if not should_skip_analysis(user_input, continuation_info.get('is_continuation', False)):
                try:
                    uploaded_docs = [
                        d.get("filename", "") for d in (self.session.documents or [])
                        if isinstance(d, dict)
                    ]
                    file_insights = getattr(self.session, 'file_insights', {}) or {}
                    classification = await analyze_intent(
                        user_message=user_input,
                        recent_history=self.session.message_history[-5:],
                        uploaded_docs=uploaded_docs,
                        file_insights=file_insights,
                    )
                    print(
                        f"🧠 ANALYZER: intent={classification.get('intent')} "
                        f"tool={classification.get('recommended_tool')} "
                        f"quality={classification.get('quality_standard')} "
                        f"board={classification.get('exam_board')} "
                        f"honesty={classification.get('honesty_required')} "
                        f"fallback={classification.get('_fallback', False)}"
                    )
                    print(f"   reasoning: {classification.get('reasoning', '')}")

                    # Honor the analyzer's recommendation only when it didn't
                    # fall back to the safe default. On fallback we let the
                    # existing fast-route / LLM-routing path decide.
                    if not classification.get('_fallback'):
                        analyzer_recommended_tool = classification.get('recommended_tool')
                        tool_args_overrides = classification.get('tool_args_overrides') or {}
                        if classification.get('honesty_required'):
                            honesty_preamble = classification.get('honesty_message')
                except Exception as e:
                    print(f"ERROR running intent analyzer (continuing without): {e}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP: EXAM RESEARCH (web search for school-specific exams)
            # ═══════════════════════════════════════════════════════════════════
            # Fires only when the analyzer set needs_research=true (the user
            # named a specific school OR a country-specific exam board).
            # Runs Claude Sonnet 4.6 with web_search, returns a structured
            # brief + citations, caches in Firestore for 24h.
            #
            # The citations are streamed to the frontend as a
            # `web_sources_found` event BEFORE the quiz starts, so the user
            # sees the sources panel build up first, then the quiz.
            #
            # The brief itself is passed to the quiz generator as additional
            # context so questions are grounded in real-world material, not
            # just generic LLM knowledge.
            # ═══════════════════════════════════════════════════════════════════
            research_brief = None
            if (
                classification
                and not classification.get('_fallback')
                and classification.get('needs_research')
                and classification.get('recommended_tool') in {
                    "generate_quiz_stream",
                    "generate_flashcards_stream",
                }
            ):
                try:
                    # Topic: prefer the analyzer's tool_args_overrides topic
                    # if it set one, else fall back to whatever the user said.
                    research_topic = (
                        tool_args_overrides.get('topic')
                        or user_input[:200]
                    )

                    print(
                        f"🔬 RESEARCH PHASE starting: "
                        f"school={classification.get('school_name')} "
                        f"board={classification.get('exam_board')} "
                        f"query={classification.get('research_query')}"
                    )

                    # Tell the frontend research is underway so it can show a
                    # premium "Searching the web..." skeleton in the chat.
                    yield json.dumps({
                        "status": "web_research_started",
                        "school": classification.get('school_name'),
                        "exam_board": classification.get('exam_board'),
                        "query": classification.get('research_query'),
                    }) + "\n"

                    from services.exam_research import (
                        gather_exam_research,
                        format_brief_as_context,
                    )
                    research_brief = await gather_exam_research(
                        topic=research_topic,
                        exam_board=classification.get('exam_board'),
                        school=classification.get('school_name'),
                        research_query=classification.get('research_query'),
                        language=language,
                    )

                    if research_brief:
                        print(
                            f"🔬 RESEARCH DONE: "
                            f"{len(research_brief.get('citations', []))} citations, "
                            f"cached={research_brief.get('cached', False)}, "
                            f"real_papers={research_brief.get('found_real_papers', False)}"
                        )
                        # Stream the sources panel event for the frontend.
                        yield json.dumps({
                            "status": "web_sources_found",
                            "school": classification.get('school_name'),
                            "exam_board": classification.get('exam_board'),
                            "exam_summary": research_brief.get('exam_summary', ''),
                            "found_real_papers": research_brief.get('found_real_papers', False),
                            "honesty_note": research_brief.get('honesty_note'),
                            "cached": research_brief.get('cached', False),
                            "citations": research_brief.get('citations', []),
                        }) + "\n"

                        # The brief itself is passed directly to
                        # stream_quiz_with_bank as additional_context at
                        # the dispatch site below — we do NOT merge it
                        # into tool_args_overrides because the @tool
                        # signature for generate_quiz_stream doesn't
                        # accept that field (LangChain would drop it).
                    else:
                        # Search produced nothing usable. Tell the frontend so
                        # it can show the honesty fallback message and we
                        # generate at the typical standard for the board.
                        print("🔬 RESEARCH PHASE returned no brief - falling back to generic generation")
                        yield json.dumps({
                            "status": "web_research_failed",
                            "school": classification.get('school_name'),
                            "exam_board": classification.get('exam_board'),
                            "message": (
                                "I couldn't find specific online material "
                                "for that exam. I'll generate practice "
                                "questions at the typical standard for "
                                "this kind of exam instead."
                            ),
                        }) + "\n"
                except Exception as e:
                    print(f"ERROR running exam research (continuing without): {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZED ROUTING: Analyzer → Fast pattern → Fast LLM → Full LLM
            # ═══════════════════════════════════════════════════════════════════
            # Previously: Every message did a full gpt-4.1 call for routing (1-2s)
            # Now:
            #   0. Intent analyzer (Claude Sonnet 4.6, ~400-2000ms) — primary route
            #   1. Fast pattern matching (0ms) — fallback when analyzer skipped
            #   2. "NO_TOOL" fast path - skips routing for conversational messages
            #   3. gpt-4.1-mini routing (~300-500ms) - for ambiguous cases
            #   4. gpt-4.1 only for actual content generation
            # ═══════════════════════════════════════════════════════════════════

            fast_route_tool = self._fast_route_check(user_input)

            # ─────────────────────────────────────────────────────────────────
            # If the analyzer picked an explicit content-generating tool, use
            # its pick instead of the regex-based fast-route. The analyzer
            # has seen the full message + recent history + uploaded-doc
            # context, so its decision is strictly better-informed.
            # ─────────────────────────────────────────────────────────────────
            ANALYZER_DISPATCHABLE = {
                "generate_quiz_stream",
                "generate_flashcards_stream",
                "generate_study_sheet_stream",
                "extract_questions_from_doc",
                "search_documents",
                "summarize_document",
            }
            if analyzer_recommended_tool in ANALYZER_DISPATCHABLE:
                fast_route_tool = analyzer_recommended_tool
                print(f"🧠 ANALYZER OVERRIDE: routing to {fast_route_tool}")
            elif analyzer_recommended_tool == "respond_directly":
                # Analyzer says no tool needed. Honor it unless fast-route
                # found a content pattern (which means the user really did
                # ask for something specific).
                if not fast_route_tool or fast_route_tool == "NO_TOOL":
                    fast_route_tool = "NO_TOOL"

            # ─────────────────────────────────────────────────────────────────
            # DETERMINISTIC GUARD: never answer a complaint with another quiz.
            # The analyzer is an LLM and can misfire; this is a hard stop.
            # If the last assistant turn was a generated quiz and the user is
            # complaining or re-sending the same message, force a direct
            # written answer.
            # ─────────────────────────────────────────────────────────────────
            GENERATION_TOOLS = {
                "generate_quiz_stream",
                "generate_flashcards_stream",
                "extract_questions_from_doc",
            }
            if fast_route_tool in GENERATION_TOOLS and self._is_post_quiz_complaint(user_input):
                print("🛑 QUIZ-REPEAT GUARD: last turn was a quiz and the user "
                      "is asking for an answer — forcing direct response.")
                fast_route_tool = "NO_TOOL"

            # ─────────────────────────────────────────────────────────────────
            # Doc-extraction has its own streaming path — not a langchain tool.
            # Handle it inline before the existing LLM dispatch branch.
            # ─────────────────────────────────────────────────────────────────
            if fast_route_tool == "extract_questions_from_doc":
                if honesty_preamble:
                    yield json.dumps({"answer_chunk": honesty_preamble + "\n\n"}) + "\n"

                from services.doc_extraction import stream_extracted_questions
                all_questions = []
                async for chunk in stream_extracted_questions(
                    session=self.session,
                    chat_id=self.session.chat_id,
                    user_prompt=user_input,
                    language=language,
                ):
                    status = chunk.get("status")
                    if status == "quiz_generating":
                        yield json.dumps({
                            "status": "quiz_generating",
                            "type": "quiz",
                            "current": chunk.get("current", 0),
                            "total": chunk.get("total", "?"),
                            "message": chunk.get("message", ""),
                        }) + "\n"
                    elif status == "question_ready":
                        question = chunk.get("question") or {}
                        all_questions.append(question)
                        yield json.dumps({
                            "status": "quiz_question",
                            "type": "quiz",
                            "question": question,
                            "index": chunk.get("index", len(all_questions) - 1),
                            "total_so_far": len(all_questions),
                        }) + "\n"
                    elif status == "quiz_complete":
                        yield json.dumps({
                            "status": "quiz_complete",
                            "type": "quiz",
                            "quiz_data": all_questions,
                            "total_generated": chunk.get("total_generated", len(all_questions)),
                        }) + "\n"
                    elif status == "error":
                        yield json.dumps({
                            "status": "error",
                            "message": chunk.get("message", "Extraction failed"),
                        }) + "\n"

                # Cache extracted quiz in session so follow-ups know what just happened.
                try:
                    self.session.quizzes.append({
                        "quiz_data": all_questions,
                        "timestamp": datetime.now().isoformat(),
                        "source": "document_extraction",
                    })
                except Exception as e:
                    print(f"Failed to cache extracted quiz: {e}")

                yield json.dumps({"status": "complete"}) + "\n"
                return

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

                # An empty stream means the LLM call silently failed. Tell the
                # user instead of completing with nothing — a "complete" with
                # zero content renders as the app ignoring the message.
                if not response_content.strip():
                    print("⚠️ Empty response from LLM on ultra-fast path")
                    yield json.dumps({
                        "status": "error",
                        "message": "The response came back empty. Please try again.",
                    }) + "\n"
                    return

                # Update history and complete
                self.session.message_history.append({
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now().isoformat()
                })
                yield json.dumps({"status": "complete"}) + "\n"
                return  # Exit early, skip the tool routing path

            elif fast_route_tool:
                # Fast path: Pattern matched. FORCE the LLM to call this tool —
                # without tool_choice, the small routing model sometimes
                # second-guesses the pattern and picks a different tool
                # (the "asked for flashcards, got a quiz" bug). The LLM still
                # extracts arguments from the user message; we only constrain
                # WHICH tool gets called.
                print(f"⚡ FAST ROUTING: Forcing {fast_route_tool} (pattern matched)")
                forced_llm = self.routing_llm.bind_tools(
                    self.tools,
                    tool_choice=fast_route_tool,
                )
                response = await forced_llm.ainvoke(messages)
            else:
                # Ambiguous: Use gpt-4.1-mini for routing decision (faster than gpt-4.1)
                print("🔀 SMART ROUTING: Using gpt-4.1-mini for tool decision...")
                response = await self.routing_llm_with_tools.ainvoke(messages)
            
            # Check if tools were called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls_made = response.tool_calls

                # ─────────────────────────────────────────────────────────
                # ANALYZER OVERRIDES: merge tool_args_overrides into each
                # content-generating tool call. The analyzer's call on
                # quiz_mode/difficulty/learning_objective is better-informed
                # than the LLM's because the analyzer thinks specifically
                # about the quality bar (council-standard, NCLEX, etc.).
                # ─────────────────────────────────────────────────────────
                CONTENT_GEN_TOOLS = {"generate_quiz_stream", "generate_flashcards_stream"}
                if tool_args_overrides:
                    for tc in tool_calls_made:
                        if tc.get("name") in CONTENT_GEN_TOOLS:
                            merged = dict(tc.get("args") or {})
                            for k, v in tool_args_overrides.items():
                                if v is None:
                                    continue
                                merged[k] = v
                            tc["args"] = merged
                            print(f"📐 ANALYZER OVERRIDES applied to {tc.get('name')}: {tool_args_overrides}")

                # ─────────────────────────────────────────────────────────
                # HONESTY PREAMBLE: stream once before content generation
                # starts, so the user sees the disclosure ("These are AI-
                # generated practice questions at council standard") BEFORE
                # the quiz/flashcard stream begins.
                # ─────────────────────────────────────────────────────────
                will_generate_content = any(
                    tc.get("name") in CONTENT_GEN_TOOLS for tc in tool_calls_made
                )
                if honesty_preamble and will_generate_content:
                    print(f"💬 Streaming honesty preamble: {honesty_preamble}")
                    yield json.dumps({"answer_chunk": honesty_preamble + "\n\n"}) + "\n"

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

                                # Persist generated flashcards in session so
                                # follow-up turns know flashcards (not quizzes)
                                # were the most recent activity. Without this,
                                # _get_last_activity_summary keeps reporting
                                # an old quiz as "what just happened" and the
                                # LLM keeps producing quizzes.
                                try:
                                    self.session.last_flashcards = all_flashcards
                                    self.session.last_flashcard_topic = metadata.get("topic")
                                    self.session.last_flashcard_timestamp = datetime.now().isoformat()
                                except Exception as e:
                                    print(f"Failed to cache flashcards in session: {e}")
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

                                # Format the research brief (if we have one) into
                                # additional context for the quiz generator. This
                                # is what makes the questions specific to the
                                # researched exam instead of generic.
                                research_context_text = None
                                if research_brief:
                                    from services.exam_research import format_brief_as_context
                                    research_context_text = format_brief_as_context(research_brief)

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
                                    user_prompt=metadata.get("user_prompt"),
                                    additional_context=research_context_text,
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
                        # Yield each chunk properly formatted
                        yield json.dumps({
                            "answer_chunk":chunk.content
                        }) + "\n"

                # Empty stream = silent LLM failure. Surface it instead of
                # completing with nothing (the frontend would otherwise save
                # an empty assistant message).
                if not response_content.strip():
                    print("⚠️ Empty response from LLM on straight-response path")
                    yield json.dumps({
                        "status": "error",
                        "message": "The response came back empty. Please try again.",
                    }) + "\n"
                    return

            # Add assistant response to history (skip empty content — an empty
            # assistant turn poisons the context for follow-up messages)
            final_content = response_content if 'response_content' in locals() else response.content
            if final_content and str(final_content).strip():
                self.session.message_history.append({
                    "role": "assistant",
                    "content": final_content,
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
            import traceback
            traceback.print_exc()
            # "status" is what the frontend's stream_chunk handler routes to
            # onStatusUpdate; "type" kept for any other consumers.
            yield json.dumps({
                "status": "error",
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            }) + "\n"

    
    def _describe_deadline(self, days) -> str:
        """Plain-language deadline used in both the support prompt and the
        system prompt, so the tutor never invents a different timeline."""
        if days is None:
            return "no exam date named yet"
        if days == 0:
            return "the exam is TODAY"
        if days == 1:
            return "the exam is tomorrow"
        return f"the exam is in {days} days"

    def _merge_urgency(self, fresh: dict) -> dict:
        """
        Carry durable facts forward across turns.

        detect_urgency reads a single message, but a student states their deadline
        once and then spends the next turns answering questions about topics.
        Without this, "exam in 2 days" is forgotten on the very next message and
        generation silently goes back to being untimed — which is the whole
        problem this feature exists to fix.

        Only facts persist. `is_crisis` and `has_explicit_request` describe THIS
        message and are always taken fresh. Panic is treated as a mood rather than
        a fact, so it is not carried either.
        """
        previous = getattr(self.session, "urgency_context", None) or {}
        merged = dict(fresh)

        for key in ("days_to_exam", "confidence_baseline"):
            if merged.get(key) is None and previous.get(key) is not None:
                merged[key] = previous[key]

        return merged

    def _format_urgency_context(self) -> str:
        """Render the detected situation for the system prompt."""
        ctx = getattr(self.session, "urgency_context", None) or {}

        # An explicit request on its own ("quiz me on X") is not pressure — only a
        # deadline, a zero baseline or distress is. Without one of those this block
        # stays quiet rather than nudging the model toward a tone nobody asked for.
        has_pressure = any((
            ctx.get("days_to_exam") is not None,
            ctx.get("confidence_baseline"),
            ctx.get("emotional_state"),
        ))
        if not has_pressure:
            return "- No exam-pressure signals in this message. Respond normally."

        lines = [f"- Deadline: {self._describe_deadline(ctx.get('days_to_exam'))}."]

        if ctx.get("confidence_baseline") == "very_low":
            lines.append("- The student says they are starting from zero / have not studied.")
        if ctx.get("emotional_state") == "panicking":
            lines.append("- The student sounds stressed or scared. Take it seriously; do not perform sympathy.")
        if ctx.get("has_explicit_request"):
            lines.append("- They already said what they want. Give them that, do not redirect them.")
        if self.session.crisis_acknowledged:
            lines.append(
                "- You have ALREADY acknowledged their situation earlier in this "
                "conversation. Do not open with sympathy again — move on to helping."
            )

        return "\n".join(lines)

    async def _stream_crisis_support(
        self,
        user_input: str,
        urgency: dict,
        language: str
    ) -> AsyncGenerator[str, None]:
        """
        First reply to a student who is out of time and out of depth.

        Deliberately runs on self.llm (no tools bound) so this turn cannot end in
        a quiz. The goal is that the student feels understood and leaves the turn
        knowing what the plan is — generation happens on the next turn, once we
        know what is actually on their exam.
        """
        deadline = self._describe_deadline(urgency.get("days_to_exam"))
        knows_nothing = urgency.get("confidence_baseline") == "very_low"
        distressed = urgency.get("emotional_state") == "panicking"

        situation = [f"Deadline: {deadline}."]
        if knows_nothing:
            situation.append("They say they know nothing / have not started studying.")
        if distressed:
            situation.append("They sound panicked or badly stressed.")

        prompt = f"""You are a nursing tutor. A student just told you they are in trouble.

THEIR SITUATION:
{chr(10).join('- ' + s for s in situation)}

Write the reply that makes them feel understood and gives them a way forward.

STRUCTURE (three short paragraphs, under 120 words total):

1. Name what you heard, concretely. Use their real numbers — "{deadline}". If they
   said they know nothing, say that back without softening it and without making
   them feel stupid about it.

2. Say what is realistically achievable in that time, honestly. In two days nobody
   covers a whole course, and pretending otherwise is a lie they will find out about
   during the exam. What IS achievable: the highest-yield material, and enough
   familiarity with the question format that they stop losing marks to confusion.
   Give them that version of the plan in one or two sentences.

3. Ask exactly ONE question so you can build the real plan: what subjects are on
   this exam, and which one worries them most.

HARD RULES:
- Write in {language}. Match their language exactly.
- Separate the three paragraphs with a blank line. One dense block is harder to
  read for someone who is already overwhelmed.
- Do NOT generate questions, flashcards or a study sheet in this reply. You are
  planning with them first. Do not list your features or capabilities either.
- Do NOT promise they will be fine or that they can learn everything in time.
- Ask ONE question, not three. They are stressed, not patient.
- Sound like a person who has helped a lot of students through this, not like an
  app. Short sentences. No emoji in this particular reply.

BANNED — these read as a script and destroy the effect:
"I understand it can be challenging", "It's completely normal to struggle",
"Many nursing students experience this", "You've got this", "Let's tackle this
together", "Don't give up", "Take a deep breath", "I'm here to help you succeed".

The student's message: {user_input}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]

        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content

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

ANSWERING vs QUIZZING (the #1 routing mistake — read carefully):
• When the user pastes THEIR OWN homework, assignment, case study, or exam
  question and wants it answered ("outline...", "identify two...", "list...",
  an MCQ with options, "can you answer...", "i need an answer"), ANSWER IT
  DIRECTLY as text. That is not a quiz request. Rule 1 applies to generating
  NEW practice questions, not to answering the user's question.
• If a quiz was just generated and the user complains ("answer my question",
  "I don't want quizzes", repeats their message), apologize in one short
  sentence and answer their original question in full. NEVER generate
  another quiz in that situation.
• Only call generate_quiz_stream when the user asks to BE TESTED ("quiz me",
  "test me", "practice questions", "make me an exam").

APP CAPABILITIES (answer accurately when asked — do not deny features):
• Users CAN upload files with the paperclip button and CAN paste images
  (screenshots, photos) directly into the message box.
• Supported uploads: PDF, Word, PowerPoint, Excel, TXT/MD, and images
  (JPG, PNG, BMP, TIFF, WebP, HEIC). Image text is extracted with OCR —
  typed/printed text works well; handwriting and photos of objects (e.g. a
  wound photo) may extract little or nothing. If a user needs visual
  assessment of a photo, say honestly that the app currently reads text
  from images, not the image itself.

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

STUDENT SITUATION (detected from their message — trust this over your own guess):
{self._format_urgency_context()}

UNDER EXAM PRESSURE (applies only when the block above shows a deadline, a zero baseline, or stress):
• Acknowledge the deadline once, in ONE sentence, then generate. Don't re-open with it every turn.
• Scope to the time they actually have and SAY what you cut: "2 days, so I'm skipping dosage calc and putting all 15 on cardiac drugs." Cutting silently is what makes the app feel like it isn't listening.
• Never imply they can cover a whole course in 2 days. What the time buys is high-yield material plus enough question-format familiarity to stop losing easy marks.
• If they've named their subjects but have no plan yet, give the plan first (topics, order, what you're leaving out), then ask if they want to start there.
• If the block says you already acknowledged their situation, skip the sympathy and go straight to the work.

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
            # STEP 1: Find the MOST RECENT activity by timestamp.
            # Previously we always checked quizzes first regardless of recency,
            # which made the LLM keep proposing quizzes even after the user
            # had moved on to flashcards. We now compare timestamps and
            # surface whichever activity actually happened last.
            # ─────────────────────────────────────────────────────────────────
            quiz_ts = None
            last_quiz = None
            if self.session.quizzes and len(self.session.quizzes) > 0:
                candidate = self.session.quizzes[-1]
                if isinstance(candidate, dict):
                    last_quiz = candidate
                    quiz_ts = candidate.get('timestamp')

            flash_ts = getattr(self.session, 'last_flashcard_timestamp', None)
            has_flashcards = bool(getattr(self.session, 'last_flashcards', None))

            # Pick the latest of the two. ISO-8601 strings sort
            # lexicographically, so direct string compare is correct.
            quiz_is_latest = False
            flash_is_latest = False
            if last_quiz and has_flashcards:
                if (quiz_ts or "") >= (flash_ts or ""):
                    quiz_is_latest = True
                else:
                    flash_is_latest = True
            elif last_quiz:
                quiz_is_latest = True
            elif has_flashcards:
                flash_is_latest = True

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Build summary for the latest activity
            # ─────────────────────────────────────────────────────────────────
            if quiz_is_latest and last_quiz is not None:

                # Extract quiz data (handle different formats)
                quiz_data = last_quiz.get('quiz_data', {})

                # Get questions list - handle both formats
                if isinstance(quiz_data, list):
                    questions = quiz_data
                elif isinstance(quiz_data, dict):
                    questions = quiz_data.get('questions', [])
                else:
                    questions = []

                if questions and len(questions) > 0:
                    total_questions = len(questions)
                    correct_answers = 0
                    topics_covered = set()

                    for question in questions:
                        if not isinstance(question, dict):
                            continue
                        user_selection = question.get('userSelection', {})
                        if isinstance(user_selection, dict) and user_selection.get('isCorrect', False):
                            correct_answers += 1
                        topic = question.get('topic', '')
                        if topic:
                            topics_covered.add(topic)

                    if total_questions > 0:
                        percentage = round((correct_answers / total_questions) * 100)
                    else:
                        percentage = 0

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

            if flash_is_latest:
                topic = getattr(self.session, 'last_flashcard_topic', None) or 'General'
                return f"""
Type: FLASHCARDS_CREATED
Topic: {topic}
Status: User just reviewed flashcards

IMPORTANT: If user now says "more", "again", "another":
→ They want MORE FLASHCARDS on the same topic ({topic})
→ Use generate_flashcards_stream tool immediately!
""".strip()

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: No recent activity found
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
        # STEP 4: Determine what action to continue.
        # Pick the MOST RECENT activity by timestamp — a chat may contain both
        # quizzes and flashcards, and "more" / "again" should always continue
        # whichever the user just interacted with, not whichever happens to
        # be checked first.
        # ─────────────────────────────────────────────────────────────────────
        action_type = None
        topic = None

        quiz_ts = None
        last_quiz = None
        if self.session.quizzes and len(self.session.quizzes) > 0:
            candidate = self.session.quizzes[-1]
            if isinstance(candidate, dict):
                last_quiz = candidate
                quiz_ts = candidate.get('timestamp')

        flash_ts = getattr(self.session, 'last_flashcard_timestamp', None)
        has_flashcards = bool(getattr(self.session, 'last_flashcards', None))

        prefer_quiz = False
        prefer_flashcards = False
        if last_quiz and has_flashcards:
            if (quiz_ts or "") >= (flash_ts or ""):
                prefer_quiz = True
            else:
                prefer_flashcards = True
        elif last_quiz:
            prefer_quiz = True
        elif has_flashcards:
            prefer_flashcards = True

        if prefer_quiz and last_quiz is not None:
            action_type = 'quiz'
            quiz_data = last_quiz.get('quiz_data', {})
            if isinstance(quiz_data, list):
                questions = quiz_data
            elif isinstance(quiz_data, dict):
                questions = quiz_data.get('questions', [])
            else:
                questions = []

            if questions:
                topics = set()
                for q in questions:
                    if isinstance(q, dict):
                        t = q.get('topic', '')
                        if t:
                            topics.add(t)
                if topics:
                    topic = ', '.join(topics)

        elif prefer_flashcards:
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
