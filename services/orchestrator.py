from langchain_openai import ChatOpenAI
from tools.quiztools import NursingTools, set_session_context
from models.session import PersistentSessionContext
from typing import AsyncGenerator
from datetime import datetime
from tools.quiztools import search_documents,summarize_document,get_chat_context_from_db
import json

class NursingTutor:
    """
    Main nursing tutor orchestrator with fixed tool integration
    """
    
    def __init__(self, chat_id: str):
        self.session = PersistentSessionContext(chat_id)
        self.tools_instance = NursingTools(self.session)
        
        # Get properly decorated tools
        tools = self.tools_instance.get_tools()
        
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        # Configure LLM with tools
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.5,
            streaming=True,
            openai_api_key=api_key  
        )
        
        self.llm_with_tools = self.llm.bind_tools(tools)
    
    async def process_message(
        self, 
        user_input: str, 
        language: str = "english"
    ):
        """
        Process student message with proper streaming for simple responses,
        non-streaming for tool calls
        """
        try:
            # Update session language
            self.session.user_language = language
                  
            full_context_from_db = await get_chat_context_from_db(self.session.chat_id)
            
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
                        
            print("DECIDED TO USE TOOLS")
            # NON-STREAMING for tool calls (your existing logic)
            response = await self.llm_with_tools.ainvoke(messages)
            
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
                            
                            from services.studysheetstreamer import StudySheetStreamer
                            streamer = StudySheetStreamer(self.session)
                            
                            # Get context
                            context = await streamer.get_document_context(topic)
                            
                            # Generate outline
                            yield json.dumps({
                                "status": "study_sheet_analyzing",
                                "message": "Analyzing documents..."
                            }) + "\n"
                            
                            sections = await streamer.generate_dynamic_outline(topic, context, language)
                            
                            yield json.dumps({
                                "status": "study_sheet_plan_ready",
                                "sections": sections
                            }) + "\n"
                            
                            # Create skeleton
                            skeleton_html = streamer.create_collapsible_skeleton(topic, sections, language)
                            
                            yield json.dumps({
                                "status": "study_sheet_html_skeleton",
                                "html_content": skeleton_html
                            }) + "\n"
                            
                            # Generate each section COMPLETELY (no word-by-word)
                            current_html = skeleton_html
                            current_progress = 15
                            section_weight = 70 / len(sections)
                            
                            for i, section in enumerate(sections):
                                # Section start
                                yield json.dumps({
                                    "status": "study_sheet_section_start",
                                    "section_id": section["id"],
                                    "section_title": section["title"],
                                    "message": section["message"],
                                    "progress": current_progress
                                }) + "\n"
                                
                                
                                # Get section-specific context
                                section_context = await streamer.get_section_specific_context(
                                    section_title=section["title"],
                                    section_scope=section.get("scope", ""),
                                    topic=topic
                                )
                                
                                # Fallback if needed
                                if not section_context:
                                    print(f"⚠️ Using fallback context for: {section['title']}")
                                    section_context = context[:8000]
                                    
                                # Generate COMPLETE section (no streaming)
                                section_html = await streamer.generate_rich_section_html(section, topic, context, language)
                                
                                # Replace placeholder
                                placeholder = f"{{{{CONTENT_{section['id']}}}}}"
                                current_html = current_html.replace(placeholder, section_html)
                                
                                # Update badge
                                old_badge = f'<span class="badge badge-loading" id="badge-{section["id"]}">'
                                new_badge = f'<span class="badge badge-loaded" id="badge-{section["id"]}">✓</span>'
                                current_html = current_html.replace(old_badge, new_badge)
                                
                                current_progress += section_weight
                                
                                # Send update with complete section
                                yield json.dumps({
                                    "status": "study_sheet_content_update",
                                    "html_content": current_html,
                                    "progress": min(current_progress, 99)
                                }) + "\n"
                                
                                # Section complete
                                yield json.dumps({
                                    "status": "study_sheet_section_complete",
                                    "section_id": section["id"],
                                    "progress": current_progress
                                }) + "\n"
                            
                            # Final completion
                            yield json.dumps({
                                "status": "study_sheet_complete",
                                "html_content": current_html,
                                "progress": 100
                            }) + "\n"
                            
                            return
                                                       
                        if tool_name == "generate_quiz_stream":
                            print("🎯 Quiz tool called - checking for streaming")
                            # creates parameters we will need for the quizz, and start streaming
                            from tools.quiztools import generate_quiz_stream
                            result = await generate_quiz_stream.ainvoke(tool_args)
                            
                            # Check if tool signaled streaming intent
                            if result.get("status") == "quiz_streaming_initiated":
                                
                                print("🌊 Starting quiz streaming from orchestrator")
                                
                                metadata = result.get("metadata", {})
                                
                                # Import streaming function
                                from tools.quiztools import stream_quiz_questions
                                
                                # Track questions for final save
                                all_questions = []
                                
                                # Stream questions one by one
                                async for chunk in stream_quiz_questions(
                                    topic=metadata.get("topic"),
                                    difficulty=metadata.get("difficulty"),
                                    num_questions=metadata.get("num_questions"),
                                    source=metadata.get("source"),
                                    session=self.session
                                ):
                                    if chunk.get("status") == "generating":
                                        
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
                                        question = chunk.get("question")
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
            
            # create prompt suggestions based on the context
            suggestions = await self._generate_dynamic_suggestions()
            
            print( f"PROMPTS SUGGESTIONS CREATED: {suggestions}")
            
            # send them to the user
            if suggestions:
                yield json.dumps({
                    "status": "suggested_prompts",
                    "suggestions": suggestions
                }) + "\n"

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
        """Create nursing-specific system prompt"""
        return f"""You are an AI nursing tutor helping students develop clinical skills.
    
        Carefully analyze the context and determine the user intent before taking action
        
        Your role:
        - Help students learn nursing concepts
        - Generate practice questions and quizzes  
        - Search their uploaded study materials
        - Provide clear explanations with rationales
        - Support NCLEX-style critical thinking

        Available tools:
        - search_documents: Search student's uploaded materials
        - generate_quiz_stream: Create practice questions on any topic, this has a maximum limit of 15 questions, 
                                if the user asks for more inform him/her of the limit and confirm before proceeding the trigger the tool
        - generate_study_sheet_stream: When the user explicitly asks for a study sheet or a guide  OR when they ask for previous/old study sheets, basically, if the intent is to create or modify a study sheet
        - summarize_document: When they want document summaries
        
        CRITICAL LANGUAGE RULE:
        - When extracting the 'topic' parameter for any tool, you MUST preserve the topic 
        in the SAME LANGUAGE as the user's message
        
        Guidelines:
        - Always provide rationales for answers (WHY, not just WHAT)
        - Use clinical scenarios when appropriate
        - Focus on critical thinking and clinical judgment
        - Be encouraging but academically rigorous
        - Respond in {self.session.user_language} unless asked otherwise
        - If the user complains about an issue with their experience ask
             them what kind of change they want to see and let them know you will inform the team
           
        FORMATTING REQUIREMENTS:
        - Use clear section headers with relevant emojis (🫁 for respiratory, 🩺 for assessment, ⚠️ for critical info) if required
        - Use SINGLE line breaks between list items for better readability
        - Use bullet points (•) for lists with ONE line break between each item if it applies
        - Structure numbered steps with ONE line break between each step if it applies
        - Include **bold text** for important medical terms and concepts

        - Add TWO line breaks only between major sections
        - Keep content tight and scannable - avoid excessive white space
        - Make the content pretty aligned and easy to consume to the eye
        - Avoid putting text above the header if there is one

        Enhance the formatting while keeping all the original content and meaning intact.
        ---
        # TOOL-USE INFERENCE RULES
        ---

        1. **Intent-First Approach:** Only call a tool when the user's message contains 
        clear intent for that specific action. Never infer tool use just because 
        parameters are available.

        2. **Study Sheet Triggers (must contain action verb + intent):**
        - "Make/create/generate/build a study sheet for [topic]"
        - "I need a study guide on [topic]"
        - "Based on [quiz/summary], make me a study sheet"
        
        3. **Default Behavior for Bare Topics:**
        - If user enters just a topic name (e.g., "skeletal system"):
            * First, use search_documents to find relevant materials
            * If no documents exist, provide a brief educational response
            * Never automatically create a study sheet
        
        4. **Ambiguity Handling:**
        - If uncertain, ask: "Would you like me to (a) search your materials, 
            (b) create a study sheet, or (c) quiz you on this topic?"

        Current session:
        - Conversation so far {self.session.message_history if self.session.message_history else "no conversation yet"}
        - Student has {"documents uploaded" if self.session.documents else "no documents"}
        - Quizzes previously done {self.session.quizzes if self.session.quizzes else "no quiz done yet"}
        - file name of the last file uploaded {self.session.documents[-1]["filename"]if self.session.documents else "no documents uploaded yet"}, if you are unsure about which file the user is talking about always use this one
        - file name of the last file you had an interaction with {self.session.name_last_document_used if self.session.name_last_document_used else " no file yet"}
        - Language preference: {self.session.user_language}"""

                  
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
            
            # Limit context size for LLM
            if len(context) > 15000:
                context = context[:15000] + "..."
            
            detail_instructions = {
                "brief": "Create a concise 3-4 paragraph summary focusing on main concepts only",
                "detailed": "Create a comprehensive summary with 6-8 paragraphs covering key concepts, clinical applications, and nursing considerations", 
                "comprehensive": "Create an in-depth analysis with detailed explanations, examples, and extensive nursing applications"
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
            
            # REAL STREAMING with vector store content
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3,streaming=True )
            
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
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
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
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
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