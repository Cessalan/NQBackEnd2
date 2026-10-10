"""Time and validate every quiz question type on the live quiz model.

Runs the real generators through core/quiz_model.py on the default tier (free
students) and the fast tier (Pro), so a model change shows up here first.
Read-only: makes model calls, writes nothing.
Usage: venv/Scripts/python tools/time_quiz_writer.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from core.quiz_model import use_quiz_tier, quiz_chat_model
from services import quiz_with_bank
from tools import quiztools
from tools.sata_prompts import generate_sata_question
from tools.casestudy_prompts import generate_casestudy_question
from tools.unfolding_casestudy_prompts import generate_unfolding_casestudy

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
TOPIC = 'Trauma primary survey'


async def timed(coro):
    start = time.perf_counter()
    try:
        result = await coro
        return time.perf_counter() - start, result, None
    except Exception as error:  # noqa: BLE001 - report every failure, keep going
        return time.perf_counter() - start, None, f'{type(error).__name__}: {str(error)[:160]}'


def valid(result, key='question'):
    return isinstance(result, dict) and bool(result.get(key) or result.get('scenario'))


async def run_tier(tier):
    use_quiz_tier(tier)
    model = quiz_chat_model()
    print(f'\n### tier={tier} model={model.model_name} service_tier={getattr(model, "service_tier", None)}')

    pick, picked, error = await timed(quiz_with_bank._extract_concepts_for_mode_plan(
        content_context=CONTENT, topic=TOPIC, mode_sequence=['knowledge'] * 5,
        language='english', learning_objective='exam_prep'))
    concepts = (picked or ([], []))[0]
    print(f'  {pick:5.2f}s concept pick: {len(concepts)} concepts {error or ""}')

    start = time.perf_counter()
    mcqs = await asyncio.gather(*(quiztools._generate_single_question(
        content=CONTENT, topic=c, difficulty='medium', question_num=i + 1, language='english',
        quiz_mode='knowledge', learning_objective='exam_prep') for i, c in enumerate(concepts)), return_exceptions=True)
    ok = sum(1 for q in mcqs if valid(q))
    print(f'  {time.perf_counter() - start:5.2f}s 5 MCQs in parallel: {ok}/{len(mcqs)} valid')

    for label, coro in (
        ('SATA', generate_sata_question(topic=TOPIC, difficulty='medium', question_num=1, language='english',
                                        content_context=CONTENT)),
        ('ordering case', generate_casestudy_question(topic=TOPIC, difficulty='medium', question_num=1,
                                                      language='english', content_context=CONTENT)),
        ('unfolding case', generate_unfolding_casestudy(topic=TOPIC, difficulty='medium', language='english')),
    ):
        seconds, result, error = await timed(coro)
        print(f'  {seconds:5.2f}s {label}: {"valid" if valid(result) else "INVALID"} {error or ""}')


async def main():
    for tier in (os.getenv('TIERS', 'default,fast').split(',')):
        await run_tier(tier)


asyncio.run(main())
