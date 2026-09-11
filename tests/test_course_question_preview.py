import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.course_question_preview import source_cards, validate_questions, stream_question_preview, preview_passage

PASSAGE = 'A source passage long enough to be useful. It explains the tested distinction and supports the correct answer.'


class QuestionPreviewTests(unittest.TestCase):
    def test_preview_omits_credits_without_rewriting_the_source(self):
        text = 'H2021 Author Name, Inf. M.Sc. Adapté de Someone, B.Sc. ' + PASSAGE
        result = preview_passage(text)
        self.assertEqual(result, PASSAGE)
        self.assertIn(result, text)
        self.assertIsNone(preview_passage('Copyright Author Name, M.Sc. ' * 20))

    def question(self, **changes):
        return dict(question='A sample question?', options=['A', 'B', 'C', 'D'], correctIndex=0,
                    topic='Topic', rationale='Explanation', sourceId=0, sourceQuote=PASSAGE, **changes)

    def test_sources_must_belong_to_uploaded_files(self):
        docs = [types.SimpleNamespace(page_content=PASSAGE, metadata={'source': name})
                for name in ['unknown.pdf', '/tmp/lecture.pdf', 'lecture.pdf']]
        self.assertEqual(source_cards(docs, ['lecture.pdf']), [{'id': 0, 'filename': 'lecture.pdf', 'text': PASSAGE}])

    def test_invented_quotes_and_malformed_keys_are_rejected(self):
        valid = self.question()
        sources = [{'filename': 'lecture.pdf', 'text': PASSAGE}]
        for changes in [{'sourceQuote': 'An invented passage which is not in the document.'},
                        {'sourceId': True}, {'correctIndex': -1}, {'correctIndex': True},
                        {'topic': 'Other'}, {'options': ['A', 'A', 'C', 'D']}]:
            self.assertEqual(validate_questions([{**valid, **changes}], sources, ['Topic']), [])
        result = validate_questions([valid, valid], sources, ['Topic'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['source'], {'filename': 'lecture.pdf', 'excerpt': PASSAGE})

    def test_passage_precedes_questions_and_language_reaches_generator(self):
        events, prompts = [], []
        class LLM:
            def __init__(self, **kwargs): pass
            async def ainvoke(inner, prompt):
                prompts.append(prompt)
                self.assertEqual(events[0]['status'], 'course_material_excerpt')
                return types.SimpleNamespace(content=json.dumps([self.question(), {**self.question(), 'question': 'Another question?'}]))
        engine = types.SimpleNamespace(
            summarize_materials=lambda _: {'topics': ['Topic']}, normalize_context=lambda c: c,
            build_strategy=lambda *args: {'ordered_topics': ['Topic'], 'priority_topics': []})
        store = types.SimpleNamespace(similarity_search=lambda *args, **kwargs: [
            types.SimpleNamespace(page_content=PASSAGE, metadata={'source': 'lecture.pdf'})])
        async def run():
            async for event in stream_question_preview(store, {'lecture.pdf': {}}, {}, 'French', engine):
                events.append(event)
        with patch.dict(sys.modules, {'langchain_openai': types.SimpleNamespace(ChatOpenAI=LLM)}):
            asyncio.run(run())
        self.assertEqual(events[-1]['status'], 'course_question_ready')
        self.assertEqual(len(events[-1]['questions']), 2)
        self.assertIn('in French', prompts[0])


if __name__ == '__main__':
    unittest.main()
