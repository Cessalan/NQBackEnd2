import asyncio
import os
import unittest
from unittest.mock import patch

from core import quiz_model as qm


class QuizModelTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
        self.env.start()
        os.environ.pop('QUIZ_GENERATION_MODEL', None)
        qm.use_quiz_tier('default')

    def tearDown(self):
        self.env.stop()

    def test_everyone_writes_on_luna_without_sampling_parameters(self):
        llm = qm.quiz_chat_model()
        self.assertEqual(llm.model_name, 'gpt-6-luna')
        self.assertEqual(llm.reasoning_effort, 'low')
        self.assertEqual(llm.service_tier, 'default')
        self.assertIsNone(llm.temperature)

    def test_pro_tier_is_fast_and_anything_unknown_is_default(self):
        qm.use_quiz_tier('fast')
        self.assertEqual(qm.quiz_chat_model().service_tier, 'fast')
        qm.use_quiz_tier('priority')
        self.assertEqual(qm.quiz_chat_model().service_tier, 'default')
        qm.use_quiz_tier(None)
        self.assertEqual(qm.quiz_tier(), 'default')

    def test_parallel_question_tasks_inherit_the_quiz_tier(self):
        async def quiz():
            qm.use_quiz_tier('fast')
            async def question():
                return qm.quiz_chat_model().service_tier
            return await asyncio.gather(*(asyncio.create_task(question()) for _ in range(3)))
        self.assertEqual(asyncio.run(quiz()), ['fast', 'fast', 'fast'])

    def test_a_different_task_never_inherits_another_students_tier(self):
        async def pro_quiz():
            qm.use_quiz_tier('fast')
        async def other_student():
            return qm.quiz_tier()
        async def server():
            await asyncio.create_task(pro_quiz())
            return await asyncio.create_task(other_student())
        self.assertEqual(asyncio.run(server()), 'default')

    def test_env_override_rolls_back_to_a_sampling_model(self):
        with patch.dict(os.environ, {'QUIZ_GENERATION_MODEL': 'gpt-4.1-mini'}):
            llm = qm.quiz_chat_model()
        self.assertEqual(llm.model_name, 'gpt-4.1-mini')
        self.assertEqual(llm.temperature, 0.7)


if __name__ == '__main__':
    unittest.main()
