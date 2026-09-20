import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from services.quiz_tutor import respond


class StudyReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def call_tutor(self, selection, message='continue'):
        create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(
            type='text', text=json.dumps({'action': 'extend', 'settings': {'requested_total': 50},
                                         'reply': 'Which cue supports your reasoning?', 'sources_used': []}))]))
        request = SimpleNamespace(message=message, selection=selection,
            question={'question': 'Which cue?', 'options': ['Secret answer'], 'correctIndex': 0, 'rationale': 'Secret rationale'},
            history=[{'role': 'assistant', 'content': 'Earlier solution'}], settings={}, performance={}, language='en')
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch.dict('sys.modules', {'services.quiz_rationale': SimpleNamespace(_get_client=lambda: client)}):
            result = await respond(request, SimpleNamespace(vectorstore=None), reasoning=True)
        return result, create.call_args.kwargs

    async def test_reasoning_never_reconfigures_or_continues_quiz(self):
        result, args = await self.call_tutor({'isCorrect': False, 'selectedIndex': 1})
        self.assertEqual(result['action'], 'explain')
        self.assertEqual(result['settings'], {})
        self.assertIn('STUDY-PLAN REASONING', args['system'])
        payload = json.loads(args['messages'][0]['content'])
        self.assertEqual(payload['selection']['selectedIndex'], 1)
        self.assertEqual(payload['question']['rationale'], 'Secret rationale')

    async def test_pre_answer_excludes_key_options_and_previous_solutions(self):
        _, args = await self.call_tutor({})
        payload = json.loads(args['messages'][0]['content'])
        self.assertTrue(payload['hint_mode'])
        self.assertEqual(payload['question'], {'question': 'Which cue?'})
        self.assertEqual(payload['history'], [])
        self.assertEqual(payload['selection'], {})


if __name__ == '__main__':
    unittest.main()
