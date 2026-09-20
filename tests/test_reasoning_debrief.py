import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from services.reasoning_debrief import discussion_evidence, validated_summary, build_reasoning_debrief


def request():
    return SimpleNamespace(language='en', topic='Assessment', items=[SimpleNamespace(question_index=2,
        question='How would you assess relief?', correct=False, question_type='mcq', rationale='Use a direct measure.')],
        reasoning_discussions=[{'question_index': 2, 'history': [
            {'role': 'user', 'content': 'I thought vital signs measured relief.'},
            {'role': 'assistant', 'content': 'A direct measure is more useful here.'}]}])


class ReasoningDebriefTests(unittest.IsolatedAsyncioTestCase):
    def test_excludes_unanswered_and_other_question_discussions(self):
        body = request()
        body.reasoning_discussions.append({'question_index': 99, 'history': body.reasoning_discussions[0]['history']})
        self.assertEqual([e['question_index'] for e in discussion_evidence(body)], [2])

    def test_only_learner_evidence_can_support_a_focus(self):
        evidence = discussion_evidence(request())
        entry = {'question_index': 2, 'learner_quote': 'I thought vital signs measured relief.',
                 'summary': 'You linked vital signs with relief. Try distinguishing direct and indirect measures.', 'status': 'needs_check'}
        data = {'summaries': [entry], 'focus': {'question_index': 2, 'skill': 'Direct versus indirect assessment'}}
        result = validated_summary(data, evidence)
        self.assertFalse(result['hasPattern'])
        self.assertEqual(result['reasoningFocus']['question_type'], 'mcq')
        for quote in ('A direct measure is more useful here.', 'Invented learner reasoning'):
            entry['learner_quote'] = quote
            self.assertIsNone(validated_summary(data, evidence))

    def test_discussion_without_a_supported_focus_falls_back(self):
        self.assertIsNone(validated_summary({'summaries': [], 'focus': None}, discussion_evidence(request())))

    def test_discussed_point_can_inform_note_without_forcing_another_drill(self):
        result = validated_summary({'summaries': [{'question_index': 2,
            'learner_quote': 'I thought vital signs measured relief.',
            'summary': 'We discussed direct and indirect assessment.', 'status': 'discussed'}],
            'focus': None}, discussion_evidence(request()))
        self.assertEqual(result['noteMode'], 'reasoning')
        self.assertIsNone(result['reasoningFocus'])
        self.assertFalse(result['hasPattern'])

    async def test_model_failure_does_not_block_normal_debrief(self):
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError('offline'))))
        with patch.dict('sys.modules', {'services.quiz_rationale': SimpleNamespace(_get_client=lambda: client)}):
            self.assertIsNone(await build_reasoning_debrief(request()))

    async def test_no_discussions_skips_model(self):
        body = request()
        body.reasoning_discussions = []
        self.assertIsNone(await build_reasoning_debrief(body))


if __name__ == '__main__':
    unittest.main()
