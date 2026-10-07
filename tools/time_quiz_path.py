"""Time the steps a chat quiz request runs before the quiz can open.

Read-only: makes the same model calls the orchestrator makes, writes nothing.
Usage: venv/Scripts/python tools/time_quiz_path.py [runs]
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

MESSAGES = [
    "quiz me on the primary survey",
    "Generate a knowledge test quiz about Examen primaire, Examen secondaire. Use direct factual questions, not clinical scenarios.",
    "create ne a quiz to prepare me for my exam",
]
INSIGHTS = {"Évaluation (C)ABCDE.pdf": {"topics": ["Préparation et triage", "Examen primaire", "Examen secondaire",
                                                  "Réévaluation des adjuvants ciblés"]}}


async def time_analyzer(message):
    from services.intent_analyzer import analyze_intent
    start = time.perf_counter()
    result = await analyze_intent(user_message=message, recent_history=[], uploaded_docs=list(INSIGHTS),
                                  file_insights=INSIGHTS, saved_practice=None)
    return time.perf_counter() - start, result.get('intent'), result.get('_fallback', False)


async def time_routing(message):
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    @tool
    def generate_quiz_stream(topic: str, num_questions: int = 5, difficulty: str = "medium", user_prompt: str = "") -> dict:
        """Generate a quiz on a topic."""
        return {}

    llm = ChatOpenAI(model="gpt-4.1-mini").bind_tools([generate_quiz_stream], tool_choice="generate_quiz_stream")
    start = time.perf_counter()
    await llm.ainvoke([{"role": "system", "content": "Route the student's request to a tool."},
                       {"role": "user", "content": message}])
    return time.perf_counter() - start


async def main(runs):
    for message in MESSAGES:
        for run in range(runs):
            analyzer, intent, fallback = await time_analyzer(message)
            routing = await time_routing(message)
            print(f"{analyzer:6.2f}s analyzer ({intent}{', FALLBACK' if fallback else ''})  "
                  f"{routing:5.2f}s routing   | {message[:60]}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
