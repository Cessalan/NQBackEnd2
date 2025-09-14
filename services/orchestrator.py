from langchain_openai import ChatOpenAI
from tools.quiztools import NursingTools, set_session_context
from models.session import PersistentSessionContext
import json
from collections.abc import AsyncGenerator
import datetime

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
    ) -> AsyncGenerator[str, None]:
        """
        Process student message with proper tool calling and streaming
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
            
            # Stream response with tool calls
            response_content = ""
            tool_calls_made = []
            
            async for chunk in self.llm_with_tools.astream(messages):
                # Handle different chunk types
                if hasattr(chunk, 'content') and chunk.content:
                    response_content += chunk.content
                    yield json.dumps({
                        "type": "content",
                        "content": chunk.content
                    }) + "\n"
                
                elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_calls_made.append(tool_call)
                        yield json.dumps({
                            "type": "tool_call",
                            "tool_name": tool_call.get("name"),
                            "status": "executing"
                        }) + "\n"
            
            # Add assistant response to history
            self.session.message_history.append({
                "role": "assistant",
                "content": response_content,
                "tool_calls": tool_calls_made,
                "timestamp": datetime.now().isoformat()
            })
            
            # Final response status
            yield json.dumps({
                "type": "complete",
                "tools_used": len(tool_calls_made)
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            }) + "\n"
    
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

        Current session:
        - Student has {"documents uploaded" if self.session.documents else "no documents"}
        - Language preference: {self.session.user_language}
        """

    def _format_chat_history(self) -> list:
        """Format chat history for LLM"""
        formatted = []
        for msg in self.session.message_history[-10:]:  # Last 10 messages
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted