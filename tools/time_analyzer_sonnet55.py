"""Intent analyzer on claude-sonnet-4-6 (shipped) vs claude-sonnet-5-5: latency, verdict, cost.

Same prompt, tools and context as services/intent_analyzer.py; only the model
and thinking/effort settings change. Read-only.
Usage: venv/Scripts/python tools/time_analyzer_sonnet55.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

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
# $ per million tokens: input, cache read, cache write (5 min), output.
PRICES = {"claude-sonnet-4-6": (3.00, 0.30, 3.75, 15.00), "claude-sonnet-5-5": (2.00, 0.20, 2.50, 10.00)}
VARIANTS = [
    ("sonnet-4-6 adaptive, no web", "claude-sonnet-4-6", {"type": "adaptive"}, None),
    ("sonnet-5-5 adaptive high, no web", "claude-sonnet-5-5", {"type": "adaptive"}, None),
    ("sonnet-5-5 adaptive low, no web", "claude-sonnet-5-5", {"type": "adaptive"}, "low"),
]


async def run(model, thinking, effort, message):
    client = ia._get_client()
    original = client.messages.create
    seen = {}

    async def patched(**kwargs):
        kwargs["model"] = model
        kwargs["thinking"] = thinking
        # No web search: quizzes never need it (owner, 2026-10-07).
        kwargs["tools"] = [t for t in kwargs["tools"] if t.get("name") != "web_search"]
        if effort:
            # This SDK version has no output_config keyword; send it in the body.
            kwargs["extra_body"] = {"output_config": {"effort": effort}}
        response = await original(**kwargs)
        seen["usage"] = response.usage
        seen["stop"] = response.stop_reason
        return response

    client.messages.create = patched
    try:
        start = time.perf_counter()
        result = await ia.analyze_intent(user_message=message, uploaded_docs=list(INSIGHTS), file_insights=INSIGHTS)
        return time.perf_counter() - start, result, seen
    finally:
        client.messages.create = original


def cost(model, usage):
    if usage is None:
        return 0.0
    p_in, p_read, p_write, p_out = PRICES[model]
    return (usage.input_tokens * p_in + (usage.cache_read_input_tokens or 0) * p_read
            + (usage.cache_creation_input_tokens or 0) * p_write + usage.output_tokens * p_out) / 1e6


async def main():
    totals = {label: [0.0, 0.0, 0] for label, *_ in VARIANTS}
    for message in MESSAGES:
        print("\n### " + message)
        for label, model, thinking, effort in VARIANTS:
            seconds, result, seen = await run(model, thinking, effort, message)
            usage = seen.get("usage")
            dollars = cost(model, usage)
            out = usage.output_tokens if usage else 0
            print(f"  {seconds:5.2f}s ${dollars:.4f} out={out:4d} {label:34s} intent={result.get('intent')} "
                  f"tool={result.get('recommended_tool')} board={result.get('exam_board')} fallback={result.get('_fallback')}")
            totals[label][0] += seconds
            totals[label][1] += dollars
            totals[label][2] += 1
    print("\n### averages")
    for label, (seconds, dollars, n) in totals.items():
        print(f"  {label:34s} {seconds / n:5.2f}s  ${dollars / n:.4f} per call  (~${dollars / n * 1000:.2f} per 1,000 messages)")


asyncio.run(main())
