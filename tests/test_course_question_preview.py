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
        self.assertIn('8-question readiness check', prompts[0])
        self.assertIn('Select all that apply.', prompts[0])
        self.assertIn('1. applied, 2. sata, 3. prioritization, 4. casestudy', prompts[0])

    def item(self, **changes):
        # question() already names every base field, so overrides are merged in.
        return {**self.question(), **changes}

    def sata(self, **changes):
        base = self.item(question='Which findings apply?', kind='sata',
                         options=['A', 'B', 'C', 'D', 'E'], correctIndices=[2, 0])
        base.pop('correctIndex')
        return {**base, **changes}

    def case(self, **changes):
        return {**self.item(question='What should the nurse do next?', kind='casestudy',
                            scenario='A patient on the unit reports new symptoms after a medication change this morning.'),
                **changes}

    def test_each_format_is_validated_on_its_own_terms(self):
        sources = [{'filename': 'lecture.pdf', 'text': PASSAGE}]
        result = validate_questions([self.question(), self.sata(), self.case()], sources, ['Topic'])
        self.assertEqual([q['format'] for q in result], ['mcq', 'sata', 'casestudy'])
        self.assertEqual([q['kind'] for q in result], ['applied', 'sata', 'casestudy'])
        self.assertEqual(result[1]['correctIndices'], [0, 2])
        self.assertNotIn('correctIndex', result[1])
        self.assertIn('new symptoms', result[2]['scenario'])
        self.assertTrue(all('sourceId' not in q for q in result))
        broken = [
            self.sata(question='one key', correctIndices=[1]),
            self.sata(question='every key', correctIndices=[0, 1, 2, 3, 4]),
            self.sata(question='bool key', correctIndices=[True, 2]),
            self.sata(question='repeated key', correctIndices=[2, 2]),
            self.sata(question='out of range', correctIndices=[0, 5]),
            self.sata(question='four options', options=['A', 'B', 'C', 'D']),
            self.sata(question='single key only', correctIndices=None, correctIndex=0),
            self.case(question='no scenario', scenario=''),
            self.case(question='short scenario', scenario='Too short.'),
            self.item(question='unknown kind', kind='essay'),
        ]
        self.assertEqual(len(validate_questions([self.question(), *broken], sources, ['Topic'])), 1)

    def test_the_opener_is_single_answer_and_quotes_the_passage_on_screen(self):
        sources = [{'filename': 'lecture.pdf', 'text': PASSAGE}, {'filename': 'other.pdf', 'text': PASSAGE}]
        later = self.item(question='Applied, other source', sourceId=1)
        opener = self.item(question='Opener')
        result = validate_questions([self.sata(), later, opener], sources, ['Topic'])
        self.assertEqual(result[0]['question'], 'Opener')
        self.assertEqual(len(result), 3)
        # Without a single-answer question on source 0 there is no opener, so no check.
        self.assertEqual(validate_questions([self.sata(), later], sources, ['Topic']), [])

    def test_citation_slips_are_repaired_only_from_the_source_text(self):
        other = 'A different passage about fall prevention. Keep the bed in the lowest position at all times.'
        sources = [{'filename': 'lecture.pdf', 'text': PASSAGE}, {'filename': 'falls.pdf', 'text': other}]
        opener = self.item()
        # Right words, wrong source number: re-attributed to where they are.
        moved = self.item(question='Moved', sourceId=0, sourceQuote='Keep the bed in the lowest position at all times.')
        # First word capitalised: the source's own casing is what gets shown.
        cased = self.item(question='Cased', sourceQuote='It Explains the tested distinction and supports the correct answer.')
        # Two sentences spliced with an ellipsis, plus an added word: the longest
        # real fragment is kept, the added word never is.
        spliced = self.item(question='Spliced', sourceQuote='A source passage long enough to be useful... supports the correct answer (always).')
        result = validate_questions([opener, moved, cased, spliced], sources, ['Topic'])
        by_title = {q['question']: q for q in result}
        self.assertEqual(by_title['Moved']['source']['filename'], 'falls.pdf')
        self.assertEqual(by_title['Cased']['source']['excerpt'],
                         'It explains the tested distinction and supports the correct answer.')
        self.assertIn(by_title['Spliced']['source']['excerpt'], PASSAGE)
        self.assertNotIn('always', by_title['Spliced']['source']['excerpt'])
        # Nothing real to cut out, or only a scrap: still refused.
        refused = [
            self.item(question='Invented', sourceQuote='Nothing like this sentence appears in either uploaded file.'),
            self.item(question='Scrap', sourceQuote='A source... answer.'),
        ]
        self.assertEqual(len(validate_questions([opener, *refused], sources, ['Topic'])), 1)

    def test_the_check_is_capped_at_eight(self):
        sources = [{'filename': 'lecture.pdf', 'text': PASSAGE}]
        many = [self.item(question=f'Question {i}?') for i in range(12)]
        self.assertEqual(len(validate_questions(many, sources, ['Topic'])), 8)


if __name__ == '__main__':
    unittest.main()
