import ast
import copy
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from tools.matrix_prompts import validate_matrix, generate_matrix_question
from services.quiz_tutor import tutor_question_context


def question():
    return {'questionType': 'matrix', 'question': 'Classify these findings.',
            'columns': [{'id': 'c1', 'label': 'Expected'}, {'id': 'c2', 'label': 'Unexpected'}],
            'rows': [{'id': f'r{i}', 'text': f'Finding {i}', 'correctColumnId': 'c1', 'explanation': 'Row rationale'} for i in range(3)]}


class MatrixTests(unittest.IsolatedAsyncioTestCase):
    def test_schema_rejects_ambiguous_or_broken_keys(self):
        self.assertTrue(validate_matrix(question()))
        for mutation in ('bad_key', 'duplicate_id', 'duplicate_label', 'missing_rationale'):
            q = question()
            if mutation == 'bad_key': q['rows'][0]['correctColumnId'] = 'missing'
            if mutation == 'duplicate_id': q['rows'][0]['id'] = q['rows'][1]['id']
            if mutation == 'duplicate_label': q['columns'][0]['label'] = q['columns'][1]['label']
            if mutation == 'missing_rationale': q['rows'][0]['explanation'] = ''
            self.assertFalse(validate_matrix(q), mutation)

    def test_hint_context_keeps_grid_but_strips_every_row_solution(self):
        context = tutor_question_context(SimpleNamespace(question=question(), selection={}))
        self.assertEqual(context['rows'][0], {'id': 'r0', 'text': 'Finding 0'})
        self.assertNotIn('correctColumnId', json.dumps(context))
        self.assertNotIn('explanation', json.dumps(context))

    def test_distribution_preserves_matrix_and_exact_requested_count(self):
        source = Path('tools/sata_prompts.py').read_text(encoding='utf-8-sig')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'distribute_question_types')
        scope = {'random': random, 'print': lambda *args: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<distribution>', 'exec'), scope)
        for count in (1, 2, 3, 5, 10):
            result = scope['distribute_question_types'](count, ['mcq', 'sata', 'matrix'])
            self.assertEqual(len(result), count)
            if count >= 3: self.assertIn('matrix', result)
        self.assertEqual(scope['distribute_question_types'](3, ['matrix']), ['matrix'] * 3)

    async def test_generation_retries_invalid_payload_without_changing_format(self):
        bad = copy.deepcopy(question()); bad['rows'][0]['correctColumnId'] = 'missing'
        create = AsyncMock(side_effect=[SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(q))]) for q in (bad, question())])
        with patch.dict('sys.modules', {'services.quiz_rationale': SimpleNamespace(_get_client=lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))}):
            result = await generate_matrix_question('Topic', 'medium', 1, 'en', content_context='Course excerpt')
        self.assertTrue(validate_matrix(result))
        self.assertEqual(create.await_count, 2)
        self.assertIn('Course excerpt', create.call_args.kwargs['messages'][0]['content'])


if __name__ == '__main__': unittest.main()
