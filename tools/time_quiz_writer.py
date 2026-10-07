"""Time the quiz writer (concept pick + 5 parallel MCQs) on different models.

Read-only: makes the generator's real calls, writes nothing.
Usage: venv/Scripts/python tools/time_quiz_writer.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import langchain_openai
from services import quiz_with_bank
from tools import quiztools

CONTENT = """Document content:
Primary survey (C)ABCDE. C - Catastrophic haemorrhage: control massive external bleeding first with direct
pressure, haemostatic dressing or tourniquet before assessing the airway. A - Airway with cervical spine
protection: look, listen, feel; jaw thrust if spinal injury suspected; suction blood or vomit. B - Breathing:
respiratory rate, SpO2, chest rise symmetry, tracheal position; suspect tension pneumothorax with absent breath
sounds, hypotension and distended neck veins. C - Circulation: central pulse, capillary refill under 2 seconds,
skin colour, two large-bore IV lines, warmed fluids; no central pulse means start CPR immediately.
D - Disability: GCS, pupils, blood glucose. E - Exposure: remove clothing, inspect, prevent hypothermia.
Secondary survey. MIST handover: Mechanism, Injuries, Signs, Treatment. SAMPLE history: Signs and symptoms,
Allergies, Medications, Past history, Last meal, Events. Reassess ABCDE after every intervention."""

REAL = langchain_openai.ChatOpenAI


def factory(model, tier=None):
    def make(*args, **kwargs):
        kwargs['model'] = model
        if model.startswith('gpt-6'):
            # Reasoning model: no sampling parameters; low effort for speed.
            kwargs.pop('temperature', None)
            kwargs['reasoning_effort'] = 'low'
            if tier:
                kwargs['service_tier'] = tier
        return REAL(**kwargs)
    return make


async def one_quiz():
    start = time.perf_counter()
    concepts, modes = await quiz_with_bank._extract_concepts_for_mode_plan(
        content_context=CONTENT, topic='Primary and secondary survey', mode_sequence=['knowledge'] * 5,
        language='english', learning_objective='exam_prep')
    picked = time.perf_counter()
    questions = await asyncio.gather(*(quiztools._generate_single_question(
        content=CONTENT, topic=c, difficulty='medium', question_num=i + 1, language='english',
        quiz_mode='knowledge', learning_objective='exam_prep') for i, c in enumerate(concepts)))
    done = time.perf_counter()
    ok = sum(1 for q in questions if isinstance(q, dict) and q.get('question'))
    return picked - start, done - picked, ok, len(concepts), (questions[0] or {}).get('question', '')[:90]


async def main():
    variants = [('gpt-4.1-mini (shipped)', 'gpt-4.1-mini', None),
                ('gpt-6-luna default tier', 'gpt-6-luna', 'default'),
                ('gpt-6-luna fast tier', 'gpt-6-luna', 'fast')]
    for label, model, tier in variants:
        for run in range(2):
            quiz_with_bank.ChatOpenAI = factory(model, tier)
            quiztools.ChatOpenAI = factory(model, tier)
            try:
                pick, write, ok, n, sample = await one_quiz()
                print(f"{label:24s} pick {pick:5.2f}s  write {write:5.2f}s  total {pick + write:5.2f}s  ok {ok}/{n} | {sample}")
            except Exception as error:
                print(f"{label:24s} FAILED {type(error).__name__}: {str(error)[:200]}")
    quiz_with_bank.ChatOpenAI = REAL
    quiztools.ChatOpenAI = REAL


asyncio.run(main())
