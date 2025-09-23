from langchain_openai import ChatOpenAI
from tools.quiztools import NursingTools, set_session_context
from models.session import PersistentSessionContext
import json
from collections.abc import AsyncGenerator
from typing import AsyncGenerator
from datetime import datetime
from tools.quiztools import generate_quiz,search_documents,summarize_document

class NursingTutor:
    """
    Main nursing tutor orchestrator with fixed tool integration
    """
    
    def __init__(self, chat_id: str):
        self.session = PersistentSessionContext(chat_id)
        self.tools_instance = NursingTools(self.session)
        
        # Get properly decorated tools
        tools = self.tools_instance.get_tools()
        
        # Configure LLM with tools
        self.llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0.3,
            streaming=True  # Enable streaming for better UX
        )
        
        self.llm_with_tools = self.llm.bind_tools(tools)
    
    async def process_message(
        self, 
        user_input: str, 
        chat_history: list = None,
        language: str = "english"
    ):
        """
        Process student message with proper streaming for simple responses,
        non-streaming for tool calls
        """
        try:
            # Update session language
            self.session.user_language = language
            
            # Ensure session context is available to tools
            set_session_context(self.session)
            
            # Add user message to history
            if chat_history:
                self.session.message_history = chat_history
            
            self.session.message_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Create nursing-specific system prompt
            system_prompt = self._create_system_prompt()
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                *self._format_chat_history(),
                {"role": "user", "content": user_input}
            ]
            
            print("CHECKING IF SHOULD USE TOOLS")
            # CHECK: Will this likely use tools?
            will_use_tools = self._should_use_tools(user_input)
            
            if will_use_tools:
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
                            if tool_name == "generate_quiz":                           
                                result = await generate_quiz.ainvoke(tool_args)
                                tool_results.append(result)
                                
                            elif tool_name == "search_documents":
                                result = await search_documents(
                                    query=tool_args.get("query", ""),
                                    max_results=tool_args.get("max_results", 3)
                                )
                                tool_results.append(result)
                                
                            elif tool_name == "check_student_progress":
                                from tools.quiztools import check_student_progress
                                result = check_student_progress()
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
                                
                            elif tool_name == "generate_study_sheet":
                                from tools.quiztools import generate_study_sheet
                                result = await generate_study_sheet.ainvoke(tool_args)
                                tool_results.append(result)   
                                                       
                        except Exception as e:
                            print(f"🔥 Tool execution error: {e}")
                            tool_results.append({"error": str(e)})
                        
                    
                    # Format and stream the tool results
                    if tool_results:
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
                                    html = result["html_content"]
                                    
                                    yield json.dumps({
                                        "html": html,
                                        "status":"studysheet_generated"
                                    })+ "\n"
                                    
                                elif "context" in result:
                                    # Handle document search results - STREAM THIS
                                    context = result['context']
                                    words = context.split()
                                    current_chunk = ""
                                    
                                    for word in words:
                                        current_chunk += word + " "
                                        if len(current_chunk) > 50:  # Longer chunks for better flow
                                            yield json.dumps({
                                                "answer_chunk": current_chunk.strip()
                                            }) + "\n"
                                            current_chunk = ""
                                    
                                    # Send remaining content
                                    if current_chunk.strip():
                                        yield json.dumps({
                                            "answer_chunk": current_chunk.strip()
                                        }) + "\n"
                                    
                                elif "error" in result:
                                    # Handle errors
                                    yield json.dumps({
                                        "answer_chunk": f"Error: {result['error']}"
                                    }) + "\n"
                else:
                    # No tools called, send complete response
                    yield json.dumps({
                        "answer_chunk": response.content
                    }) + "\n"
                    
            else:
                print("DECIDED TO NOT USE TOOLS")
                # REAL STREAMING for simple responses
                response_content = ""
                
                async for chunk in self.llm_with_tools.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        yield json.dumps({
                            "answer_chunk": chunk.content
                        }) + "\n"
            
            # Add assistant response to history
            self.session.message_history.append({
                "role": "assistant",
                "content": response_content if 'response_content' in locals() else response.content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Final response status
            yield json.dumps({
                "status": "complete"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            }) + "\n"

    def _should_use_tools(self, user_input: str) -> bool:
        """
        Simple heuristic to detect if user input will trigger tools
        """
        tool_keywords = [
            "quiz", "test", "practice", "generate", 
            "search", "find", "progress", "score",
            "create", "make", "show me", "summary","resume","summarize",
            "study sheet", "feuille d'etude"
        ]
        return any(keyword in user_input.lower() for keyword in tool_keywords)
    
    def _create_system_prompt(self) -> str:
        """Create nursing-specific system prompt"""
        return f"""You are an AI nursing tutor helping students develop clinical skills.

        Your role:
        - Help students learn nursing concepts, pharmacology, pathophysiology
        - Generate practice questions and quizzes  
        - Search their uploaded study materials
        - Provide clear explanations with rationales
        - Support NCLEX-style critical thinking

        Available tools:
        - search_documents: Search student's uploaded materials
        - generate_quiz: Create practice questions on any topic
        - check_student_progress: See what student has been working on

        Guidelines:
        - Always provide rationales for answers (WHY, not just WHAT)
        - Use clinical scenarios when appropriate
        - Focus on critical thinking and clinical judgment
        - Be encouraging but academically rigorous
        - Respond in {self.session.user_language}
        - use emojis , spacing, line breaks to make the content clearer and easier to read, adapt it for dyslexia

        Current session:
        - Student has {"documents uploaded" if self.session.documents else "no documents"}
        - Language preference: {self.session.user_language}"""

    def _format_chat_history(self) -> list:
        """Format chat history for LLM"""
        formatted = []
        for msg in self.session.message_history[-10:]:  # Last 10 messages
            # Handle both dict and Message object formats
            if isinstance(msg, dict):
                # Dictionary format (most common)
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # LangChain Message object
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                # Fallback - try to convert to string
                formatted.append({
                    "role": "user",
                    "content": str(msg)
                })
        return formatted
                    
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
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
            
            async for chunk in llm.astream([{"role": "user", "content": summary_prompt}]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            yield f"Error generating summary: {str(e)}"