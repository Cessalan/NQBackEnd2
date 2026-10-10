"""The model that writes quiz questions, and the speed tier it runs on.

2026-10-07: quiz writing moved from gpt-4.1-mini / gpt-4.1 to gpt-6-luna for
every student, with the fast service tier for Pro members. The Pro tier is
what the "faster course quizzes" announcement promises
(src/config/productAnnouncements.js in the frontend); before this, nothing in
the quiz writer looked at membership at all.

Measured on 5 questions from an ABCDE passage (tools/time_quiz_writer.py):
gpt-4.1-mini 6.4-7.4s, Luna default tier 8.3-9.0s, Luna fast tier 5.5s, 5/5
valid questions on every setting.

HOW THE TIER REACHES THE GENERATORS
The generators (MCQ, SATA, ordering, unfolding case, concept picking) are
called from many places and none of them is given a chat id. So the tier is
set once per quiz, at the top of quiz_with_bank.stream_quiz_questions, in a
context variable. asyncio copies the current context into every task it
creates, so the parallel question tasks all see it. A call that never passes
through stream_quiz_questions gets the default tier, never fast: speed is
granted from the server's own membership check, not assumed.

QUIZ_GENERATION_MODEL overrides the model for a rollback without a code
change; a non-reasoning model gets the old temperature setting back.
"""
import contextvars
import os

QUIZ_MODEL = 'gpt-6-luna'
QUIZ_REASONING = 'low'   # measured: valid questions, fastest setting
TIERS = ('default', 'fast')

_tier = contextvars.ContextVar('quiz_service_tier', default='default')


def use_quiz_tier(tier):
    """Set the tier for the quiz being generated in the current task."""
    _tier.set(tier if tier in TIERS else 'default')


def quiz_tier():
    return _tier.get()


def _is_reasoning_model(model):
    return model.startswith(('gpt-6', 'gpt-5', 'o1', 'o3', 'o4'))


def quiz_chat_model(*, timeout=90):
    """A LangChain chat model for writing quiz questions on the current tier."""
    from langchain_openai import ChatOpenAI
    model = os.getenv('QUIZ_GENERATION_MODEL') or QUIZ_MODEL
    if not _is_reasoning_model(model):
        # Rollback path: the pre-Luna models took sampling parameters.
        return ChatOpenAI(model=model, temperature=0.7, request_timeout=timeout)
    # Reasoning models reject temperature; effort and tier replace it.
    return ChatOpenAI(model=model, reasoning_effort=QUIZ_REASONING, service_tier=quiz_tier(),
                      use_responses_api=False, request_timeout=timeout)
