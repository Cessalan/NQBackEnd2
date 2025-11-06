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
                                    
                                return
                                
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

        1.  **Required Argument Focus:** When a tool requires a parameter (like 'topic' for 'generate_study_sheet_stream'), you MUST find the value for that parameter, base on the context (conversation, quizzes, summaries) provided.

        2.  **Context Extraction:** If the user requests a study guide immediately following a detailed discussion, a summary, or a displayed chunk of information, you MUST use the **primary subject** of that immediately preceding content as the **required 'topic' argument**.

        3.  **Fallback:** ONLY if the conversation is new or the topic is completely ambiguous (e.g., "Hi, what tools do you have?"), ask the user for the topic."

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
        """Generate contextual suggestions; skip if not needed."""
        try:
            # ✅ OPTIMIZATION: Only run after tool use or every 3 messages
            if len(self.session.message_history) % 3 != 0:
                return []

            recent_msgs = self.session.message_history[-6:]
            context_snippet = "\n".join(
                [f"{m['role']}: {m['content'][:200]}" for m in recent_msgs if 'content' in m]
            )  # ✅ Limit per-message length

            last_tools = [t.get("tool") for t in getattr(self.session, "tool_calls", [])[-3:]]
            last_tool_used = last_tools[-1] if last_tools else "none"

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

            system_prompt = (
                "Generate 3–5 short, actionable next-step prompts for a nursing student. "
                "Focus on deepening understanding, exploring related topics, or self-assessment. "
                f"Respond in {self.session.user_language}. Return ONLY a JSON array of strings."
            )

            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context_snippet}\n\nLAST TOOL: {last_tool_used}"}
            ])

            # ✅ ROBUST PARSING
            import json
            suggestions = json.loads(response.content)
            if isinstance(suggestions, list):
                # ✅ DEDUPLICATION (cache last suggestions in session)
                prev = getattr(self.session, "last_suggestions", [])
                new_suggestions = [s for s in suggestions[:5] if s not in prev]
                self.session.last_suggestions = new_suggestions
                return new_suggestions

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error in suggestions: {e}")
            # Fallback: split by lines
            suggestions = [s.strip("-• ") for s in response.content.split("\n") if s.strip()]
            return suggestions[:5]
        except Exception as e:
            print(f"⚠️ Suggestions failed: {e}")
            return []

        return []
