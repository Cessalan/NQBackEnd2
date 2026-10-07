"""Compare the intent analyzer on Sonnet 4.6 (shipped) against gpt-6-luna fast.

Same system prompt, same context text, same classify_intent schema. Luna is
tried two ways: a forced function call on the Responses API, and JSON-object
output on Chat Completions with the schema in the prompt (how
core/material_model.py already calls Luna). Read-only.
Usage: venv/Scripts/python tools/time_analyzer_luna.py
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from services import intent_analyzer as ia

INSIGHTS = {"Evaluation (C)ABCDE.pdf": {"topics": ["Preparation et triage", "Examen primaire", "Examen secondaire"]}}
MESSAGES = [
    "create ne a quiz to prepare me for my exam",
    "quiz me on the primary survey",
    "make me flashcards on MIST and SAMPLE",
    "UK NMC past paper style questions on sepsis",
    "no more select all that apply, just normal questions on circulation",
    "I have my exam tomorrow and I'm panicking, I don't understand anything",
    "what's the difference between MIST and SAMPLE?",
]
KEYS = ('intent', 'recommended_tool', 'quality_signal', 'exam_board', 'needs_research', 'tool_args_overrides')
NL = "\n"


async def luna(message, effort, mode):
    context = ia._build_context_text([], list(INSIGHTS), INSIGHTS, None)
    schema = ia.CLASSIFY_INTENT_TOOL["input_schema"]
    user = context + NL + NL + "User just said: " + message
    if mode == "responses":
        tool = {"type": "function", "name": ia.CLASSIFY_INTENT_TOOL["name"],
                "description": ia.CLASSIFY_INTENT_TOOL["description"], "parameters": schema}
        llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort=effort, service_tier="fast", use_responses_api=True,
                         request_timeout=60).bind_tools([tool], tool_choice=ia.CLASSIFY_INTENT_TOOL["name"])
        system = ia.ANALYSIS_SYSTEM_PROMPT
    else:
        llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort=effort, service_tier="fast", use_responses_api=False,
                         request_timeout=60).bind(response_format={"type": "json_object"})
        system = (ia.ANALYSIS_SYSTEM_PROMPT + NL + NL + "There is no web search in this call. Return ONLY a JSON "
                  "object matching this classify_intent schema:" + NL + json.dumps(schema, ensure_ascii=False))
    start = time.perf_counter()
    response = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    seconds = time.perf_counter() - start
    if mode == "responses":
        calls = response.tool_calls or []
        return seconds, (calls[0]["args"] if calls else {"_fallback": True})
    text = response.content if isinstance(response.content, str) else "".join(
        part.get("text", "") for part in response.content if isinstance(part, dict))
    return seconds, json.loads(text)


async def sonnet(message):
    start = time.perf_counter()
    result = await ia.analyze_intent(user_message=message, uploaded_docs=list(INSIGHTS), file_insights=INSIGHTS)
    return time.perf_counter() - start, result


def short(result):
    picked = {k: result.get(k) for k in KEYS if result.get(k) not in (None, {}, [], '')}
    return json.dumps(picked, ensure_ascii=False)[:230]


async def main():
    for message in MESSAGES:
        print(NL + "### " + message)
        seconds, result = await sonnet(message)
        print(f"  {seconds:5.2f}s sonnet-4-6 adaptive          {short(result)}")
        for mode, effort in (("responses", "none"), ("responses", "low"), ("json", "none"), ("json", "low")):
            try:
                seconds, result = await luna(message, effort, mode)
                print(f"  {seconds:5.2f}s luna fast {mode:9s} {effort:4s} {short(result)}")
            except Exception as error:
                print(f"  luna fast {mode} {effort}: FAILED {type(error).__name__}: {str(error)[:160]}")


asyncio.run(main())
