"""Exercise the real endpoint orchestration with isolated, local workers."""
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class PreviewStreamTests(unittest.TestCase):
    def test_materials_only_prepares_questions_without_starting_web_research(self):
        async def run():
            async def research(*args):
                self.fail('Research must not run for the quick-check upload flow')
                yield {}
            async def preview(*args):
                yield {'status': 'course_question_ready', 'questions': ['question']}
            module = types.SimpleNamespace(stream_question_preview=preview,
                material_report=lambda *args: {'research_ran': False, 'uploaded': True})
            with patch.dict(sys.modules, {'services.course_question_preview': module}):
                stream = await self.endpoint(research)(types.SimpleNamespace(
                    chat_id='test', courseContext={}, language='en', materials_only=True))
                events = [json.loads(chunk.removeprefix('data: ').strip()) async for chunk in stream]
                events = [event for event in events if event['status'] != 'heartbeat']
            self.assertEqual([e['status'] for e in events], ['course_question_ready', 'course_intelligence_ready', 'complete'])
            self.assertFalse(events[1]['report']['research_ran'])
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(asyncio.wait_for(run(), 2))

    def endpoint(self, research):
        tree = ast.parse(Path(__file__).resolve().parents[1].joinpath('main.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_course_intelligence_endpoint')
        fn.decorator_list = []
        async def load_store(_):
            return object()
        namespace = dict(asyncio=asyncio, json=json, CourseIntelligenceRequest=object,
            vectorstore_manager=types.SimpleNamespace(load_combined_vectorstore_from_firebase=load_store),
            StreamingResponse=lambda iterator, **kwargs: iterator,
            COURSE_INTELLIGENCE_HEARTBEAT_S=.01, _language_for_prompt=lambda _: 'English',
            course_intelligence_module_context=lambda _: 'test',
            course_intelligence=types.SimpleNamespace(stream_course_intelligence=research),
            ACTIVE_SESSIONS={'test': types.SimpleNamespace(session=types.SimpleNamespace(
                file_insights={'lecture.pdf': {'topics': ['Topic']}}, vectorstore=None))})
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'preview_endpoint', 'exec'), namespace)
        return namespace['run_course_intelligence_endpoint']

    def test_question_can_arrive_before_the_report(self):
        async def run():
            release = asyncio.Event()
            async def research(*args):
                await release.wait()
                yield {'status': 'course_intelligence_ready', 'report': {'final': True}}
            async def preview(*args):
                yield {'status': 'course_material_excerpt', 'source': {'filename': 'lecture.pdf'}}
                yield {'status': 'course_question_ready', 'questions': ['test']}
            with patch.dict(sys.modules, {'services.course_question_preview': types.SimpleNamespace(stream_question_preview=preview)}):
                stream = await self.endpoint(research)(types.SimpleNamespace(chat_id='test', courseContext={}, language='en'))
                received = []
                async for chunk in stream:
                    event = json.loads(chunk.removeprefix('data: ').strip())
                    received.append(event['status'])
                    if event['status'] == 'course_question_ready':
                        self.assertNotIn('course_intelligence_ready', received)
                        release.set()
                self.assertIn('course_intelligence_ready', received)
                self.assertEqual(received[-1], 'complete')
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(asyncio.wait_for(run(), 2))

    def test_disconnect_cancels_both_workers(self):
        async def run():
            cancelled = set()
            async def research(*args):
                try:
                    await asyncio.Event().wait()
                    yield {}
                finally:
                    cancelled.add('research')
            async def preview(*args):
                try:
                    yield {'status': 'course_material_excerpt'}
                    await asyncio.Event().wait()
                finally:
                    cancelled.add('preview')
            with patch.dict(sys.modules, {'services.course_question_preview': types.SimpleNamespace(stream_question_preview=preview)}):
                stream = await self.endpoint(research)(types.SimpleNamespace(chat_id='test', courseContext={}, language='en'))
                await anext(stream)
                await stream.aclose()
                self.assertEqual(cancelled, {'research', 'preview'})
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(asyncio.wait_for(run(), 2))


if __name__ == '__main__':
    unittest.main()
