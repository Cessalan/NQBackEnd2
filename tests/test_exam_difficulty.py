"""Exercise the exam endpoint with generation and persistence replaced by fakes."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from models.requests import StudyExamRequest
from pydantic import ValidationError


class ExamDifficultyTest(unittest.TestCase):
    def test_supported_levels_reach_generation_and_saved_config(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text(encoding='utf-8'))
        endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'generate_study_exam')
        endpoint.decorator_list = []
        module = ast.Module(body=[endpoint], type_ignores=[])
        received = []

        async def setup(*args):
            return SimpleNamespace(documents=[], vectorstore=None, chat_id='test')

        async def generate(**kwargs):
            received.append(kwargs)
            yield {'status': 'question_ready', 'question': {'questionType': 'matrix', 'question': 'Classify these findings'}}

        scope = dict(StudyExamRequest=StudyExamRequest, usage_guard=SimpleNamespace(check_quota=lambda _: {'allowed': True}),
                     _setup_study_session=setup, stream_quiz_with_bank=generate, hashlib=hashlib, json=json,
                     print=lambda *args: None)
        exec(compile(module, '<exam endpoint>', 'exec'), scope)
        for level in ['easy', 'medium', 'hard']:
            with self.subTest(level=level):
                request = StudyExamRequest(chat_id='test', topic='Assessment', question_difficulty=level, quiz_mode='applied')
                result = asyncio.run(scope['generate_study_exam'](request))
                self.assertEqual(received[-1]['difficulty'], level)
                self.assertEqual(received[-1]['node_difficulty'], 2)
                self.assertEqual(result['examConfig']['questionDifficulty'], level)

    def test_legacy_default_and_invalid_level(self):
        self.assertEqual(StudyExamRequest(chat_id='test', topic='Assessment').question_difficulty, 'medium')
        with self.assertRaises(ValidationError):
            StudyExamRequest(chat_id='test', topic='Assessment', question_difficulty='extreme')


if __name__ == '__main__':
    unittest.main()
