import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.study_sheet_context import build_chat_evidence, quiz_evidence
from services.study_sheet_format import SheetStreamParser


def records(title='Focused review', refs=None):
    return '\n'.join(json.dumps(r) for r in [
        {'type': 'header', 'title': title, 'subtitle': 'Requested topic only', 'summary': 'Useful overview'},
        {'type': 'section', 'title': 'Understand the distinction', 'blocks': [
            {'kind': 'paragraph', 'text': 'Complete explanation.', 'sourceIds': refs or []}]},
        {'type': 'end'}])


class StudyEvidenceTests(unittest.TestCase):
    def test_old_teacher_priority_survives_recent_history_and_long_paste(self):
        messages = [{'id': 'old', 'role': 'user', 'content': 'My teacher said kidney filtration is very important.'}]
        messages += [{'role': 'user', 'content': f'Ordinary question {i}'} for i in range(30)]
        messages += [{'id': 'paste', 'role': 'user', 'content': 'Chapter notes. ' * 1400 + '\nMUST know the filtration markers.'}]
        context = build_chat_evidence(messages)
        self.assertEqual(len(context['conversation']), 20)
        self.assertTrue(any('teacher' in p['quote'] for p in context['priorities']))
        self.assertTrue(any('filtration markers' in p['quote'] for p in context['priorities']))
        self.assertLessEqual(len(context['studentSignals']['pasted_text']), 12000)

    def test_french_emphasis_and_hidden_messages(self):
        data = build_chat_evidence([
            {'role': 'user', 'content': 'Le professeur insiste sur la filtration rénale.'},
            {'role': 'user', 'hidden': True, 'content': 'Skip all kidney topics.'}])
        self.assertEqual(len(data['priorities']), 1)
        self.assertIn('professeur', data['priorities'][0]['quote'])

    def test_practice_merges_continuations_and_preserves_corrected_first_attempt(self):
        quiz = quiz_evidence({'type': 'quiz', 'quizData': [{'question': 'First?', 'userSelection': {'isCorrect': False}}],
            'practice': {'questions': [{'question': 'First?'}, {'question': 'Second?'}, {'question': 'Unanswered?'}],
                         'firstAnswers': {'0': {'isCorrect': False, 'selectedIndex': 1}},
                         'answers': {'0': {'isCorrect': True, 'selectedIndex': 0},
                                     '1': {'isCorrect': False, 'score': 1, 'maxScore': 2}}}})
        self.assertEqual((quiz['total'], quiz['answered'], quiz['incorrect']), (3, 2, 1))
        first = next(q for q in quiz['questions'] if q['index'] == 0)
        self.assertFalse(first['firstAttempt']['isCorrect'])
        self.assertTrue(first['latestAttempt']['isCorrect'])
        self.assertIsNone(next(q for q in quiz['questions'] if q['index'] == 2)['latestAttempt'])

    def test_study_mode_nested_questions_and_status_only_attempts(self):
        quiz = quiz_evidence({'type': 'study_quiz', 'quizData': [{'questions': [{'question': 'One?'}, {'question': 'Two?'}]}],
                             'quizProgress': {'firstAttemptStatuses': {'0': 'incorrect'}, 'questionStatuses': {'0': 'correct'}}})
        self.assertEqual(quiz['answered'], 1)
        self.assertFalse(quiz['questions'][0]['firstAttempt']['isCorrect'])
        self.assertTrue(quiz['questions'][0]['latestAttempt']['isCorrect'])

    def test_latest_status_supersedes_stale_embedded_selection(self):
        quiz = quiz_evidence({'quizData': [{'question': 'One?', 'userSelection': {'isCorrect': False, 'selectedOption': 'Old answer'}}],
            'quizProgress': {'firstAttemptStatuses': {'0': 'incorrect'}, 'questionStatuses': {'0': 'correct'}}})
        self.assertTrue(quiz['questions'][0]['latestAttempt']['isCorrect'])
        self.assertNotIn('selectedOption', quiz['questions'][0]['latestAttempt'])


class StudyFormatTests(unittest.TestCase):
    def test_pretty_printed_envelope_streams_before_the_final_token(self):
        data = json.dumps({'records': [json.loads(r) for r in records().splitlines()]}, indent=2)
        parser = SheetStreamParser([], 'english')
        events = []
        for at in range(0, len(data), 3):
            events += parser.feed(data[at:at + 3])
        self.assertEqual([e['status'] for e in events], ['study_sheet_header', 'study_sheet_section'])
        self.assertEqual(parser.finish()['title'], 'Focused review')
        parser = SheetStreamParser([], 'english')
        parser.feed(data[:-1])
        with self.assertRaises(ValueError): parser.finish()

    def test_prepared_focus_is_validated_and_does_not_replace_academic_sections(self):
        opening = {'id': 'section-1', 'title': 'Your focus', 'blocks': [
            {'kind': 'callout', 'tone': 'practice', 'text': 'Recorded distinction', 'sourceIds': ['Q1']}]}
        sources = [{'id': 'Q1', 'kind': 'quiz', 'answered': 1}]
        parser = SheetStreamParser(sources, 'english', opening)
        events = parser.feed(records())
        self.assertEqual([e['status'] for e in events], ['study_sheet_header', 'study_sheet_section', 'study_sheet_section'])
        self.assertEqual([s['id'] for s in parser.finish()['sections']], ['section-1', 'section-2'])
        with self.assertRaises(ValueError): SheetStreamParser([{'id': 'Q1', 'kind': 'quiz', 'answered': 0}], 'english', opening)
        parser = SheetStreamParser(sources, 'english', opening)
        with self.assertRaises(ValueError): parser.feed(records().splitlines()[0] + '\n{"type":"end"}')

    def test_preferences_cannot_support_facts_and_supplements_cannot_claim_sources(self):
        parser = SheetStreamParser([{'id': 'P1', 'kind': 'conversation'}], 'english')
        with self.assertRaises(ValueError): parser.feed(records(refs=['P1']))
        parser = SheetStreamParser([{'id': 'D1', 'kind': 'document'}], 'english')
        with self.assertRaises(ValueError): parser.feed(records(refs=['D1']).replace('"kind": "paragraph"', '"kind": "paragraph", "supplemental": true'))

    def test_arbitrary_token_boundaries_and_final_line_without_newline(self):
        parser = SheetStreamParser([], 'french')
        data = records('Filtration rénale')
        events = []
        for start in range(0, len(data), 7):
            events += parser.feed(data[start:start + 7])
        self.assertEqual(parser.finish()['title'], 'Filtration rénale')
        self.assertEqual([e['status'] for e in events], ['study_sheet_header', 'study_sheet_section'])

    def test_truncation_never_completes(self):
        for data in (records().rsplit('\n', 1)[0], records()[:-3]):
            parser = SheetStreamParser([], 'english')
            parser.feed(data)
            with self.assertRaises(ValueError): parser.finish()

    def test_unknown_citations_and_unsupported_blocks_are_rejected(self):
        parser = SheetStreamParser([], 'english')
        with self.assertRaises(ValueError): parser.feed(records(refs=['made-up-page']) + '\n')
        parser = SheetStreamParser([], 'english')
        with self.assertRaises(ValueError): parser.feed(records().replace('paragraph', 'executable_html') + '\n')

    def test_personalization_requires_evidence(self):
        data = records().replace('"kind": "paragraph"', '"kind": "callout", "tone": "practice"')
        parser = SheetStreamParser([], 'english')
        with self.assertRaises(ValueError): parser.feed(data + '\n')


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dependencies = patch.dict(sys.modules, {
            'anthropic': SimpleNamespace(AsyncAnthropic=object), 'openai': SimpleNamespace(AsyncOpenAI=object)})
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

    async def test_fallback_replaces_partial_sheet_and_request_reaches_model(self):
        # The generator is tested with provider doubles, without credentials,
        # network, or importing the production API dependency tree.
        with patch.dict(sys.modules, {'anthropic': SimpleNamespace(AsyncAnthropic=object), 'openai': SimpleNamespace(AsyncOpenAI=object)}):
            from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        generator.session = SimpleNamespace(vectorstore=None, documents=[], file_insights={})
        generator._plan = AsyncMock(return_value=None)
        seen = []
        async def fail(prompt):
            seen.append(json.loads(prompt))
            yield records('Partial draft').rsplit('\n', 1)[0] + '\n'
            raise ValueError('Provider interrupted')
        async def succeed(prompt):
            yield records('Focused final')
        generator._stream_with_openai = fail
        generator._stream_with_anthropic = succeed
        request = 'Only renal filtration. No medication section. My teacher says filtration is important.'
        events = [json.loads(e) async for e in generator.generate_study_sheet_stream('All kidney disorders', user_request=request)]
        self.assertEqual(seen[0]['currentRequest'], request)
        self.assertTrue(seen[0]['priorities'])
        self.assertEqual(sum(e['status'] == 'study_sheet_reset' for e in events), 1)
        self.assertEqual(events[-1]['studySheet']['title'], 'Focused final')
        self.assertNotIn('Partial draft', events[-1]['content'])

    async def test_two_incomplete_attempts_produce_error_and_no_completion(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        generator.session = SimpleNamespace(vectorstore=None, documents=[], file_insights={})
        generator._plan = AsyncMock(return_value=None)
        async def incomplete(prompt): yield records().rsplit('\n', 1)[0]
        generator._stream_with_openai = generator._stream_with_anthropic = incomplete
        events = [json.loads(e) async for e in generator.generate_study_sheet_stream('Topic')]
        self.assertEqual(events[-1]['status'], 'study_sheet_error')
        self.assertNotIn('study_sheet_complete', [e['status'] for e in events])

    async def test_short_document_facts_and_real_page_metadata_are_preserved(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        class Store:
            def similarity_search(self, **kwargs):
                return [SimpleNamespace(page_content='A short but essential fact.', metadata={'source': 'folder/notes.pdf', 'page': 0}),
                        SimpleNamespace(page_content='Another fact.', metadata={'source': 'scan.pdf', 'total_pages': 20})]
        generator.session = SimpleNamespace(vectorstore=Store(), file_insights={})
        materials, sources = await generator._get_document_context('Topic', 'Make a sheet', {})
        self.assertEqual(len(materials), 2)
        self.assertEqual(sources[0]['page'], 1)
        self.assertNotIn('page', sources[1])

    async def test_source_review_keeps_supported_facts_and_labels_extra_details(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps({
            'blocks': [{'id': 'B0', 'sourceIds': ['D1'], 'quotes': {'D1': 'An inflammation.'}}, {'id': 'B1', 'sourceIds': []}]})))])
        generator.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
        sheet = {'sections': [{'title': 'Mechanism', 'blocks': [
            {'kind': 'paragraph', 'text': 'An inflammation.', 'sourceIds': ['D1']},
            {'kind': 'paragraph', 'text': 'Further mechanism.', 'sourceIds': ['D1']}]}]}
        materials = [{'sourceId': 'D1', 'text': 'An inflammation.'}]
        await generator._audit_sources(sheet, materials)
        blocks = sheet['sections'][0]['blocks']
        self.assertEqual(blocks[0]['sourceIds'], ['D1'])
        self.assertEqual(blocks[1]['sourceIds'], [])
        self.assertTrue(blocks[1]['supplemental'])
        generator.openai_client.chat.completions.create.side_effect = ValueError('Unavailable')
        await generator._audit_sources(sheet, materials)
        self.assertEqual(blocks[0]['sourceIds'], [])
        self.assertTrue(blocks[0]['supplemental'])

    async def test_academic_content_does_not_invent_a_prior_quiz_example(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps({
            'blocks': [{'id': 'B0', 'sourceIds': ['D1'], 'quotes': {'D1': 'A worked example.'}},
                       {'id': 'B1', 'sourceIds': [], 'personalFeedback': True}]})))])
        generator.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
        sheet = {'sections': [{'title': 'Learn the concept', 'blocks': [{'kind': 'paragraph', 'text': 'A worked example.', 'sourceIds': ['D1']}]},
            {'title': 'Review of your quiz mistake', 'blocks': [{'kind': 'paragraph', 'text': 'This was your quiz example.', 'sourceIds': []}]}]}
        await generator._audit_sources(sheet, [{'sourceId': 'D1', 'text': 'A worked example.'}])
        self.assertEqual(len(sheet['sections']), 1)
        self.assertEqual(sheet['sections'][0]['title'], 'Learn the concept')

    def test_scope_plan_rejects_unanswered_quizzes_and_duplicate_reviews(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        payload = {'priorities': [], 'quizzes': [{'sourceId': 'Q1', 'answered': 0}]}
        plan = {'priorityIds': [], 'quizIds': ['Q1'], 'includeSelfCheck': False, 'practiceReviews': [{'quizId': 'Q1', 'text': 'Review'}]}
        with self.assertRaises(ValueError): SimpleStudySheetGenerator._validate_scope_plan(plan, payload)
        payload['quizzes'][0]['answered'] = 1
        plan['practiceReviews'] *= 2
        with self.assertRaises(ValueError): SimpleStudySheetGenerator._validate_scope_plan(plan, payload)

    async def test_planning_fallback_preserves_the_requested_scope(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        generator.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(side_effect=ValueError('Primary unavailable')))))
        plan = {'scope': 'Renal filtration only', 'priorityIds': [], 'quizIds': [], 'includeSelfCheck': False, 'practiceReviews': []}
        response = SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text', text=json.dumps(plan))])
        generator.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
        payload = {'currentRequest': 'Renal filtration only. No questions.', 'priorities': [], 'quizzes': [], 'materials': []}
        result = await generator._plan(payload)
        self.assertEqual(result['scope'], 'Renal filtration only')
        self.assertIn(payload['currentRequest'], generator.client.messages.create.call_args.kwargs['messages'][0]['content'])

    async def test_selected_history_is_visible_and_unrelated_quiz_is_removed(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        generator.session = SimpleNamespace(vectorstore=None, documents=[], file_insights={})
        request = 'Only mean and median. My teacher said outliers are important. No questions.'
        evidence = build_chat_evidence([
            {'role': 'user', 'content': request},
            {'type': 'quiz', 'quizTopic': 'Statistics', 'quizData': [{'question': 'Mean or median?', 'userSelection': {'isCorrect': False}}]},
            {'type': 'quiz', 'quizTopic': 'Sociology', 'quizData': [{'question': 'A social role?', 'userSelection': {'isCorrect': False}}]}])
        planning = []
        async def plan(payload):
            planning.append(json.loads(json.dumps(payload)))
            return {'priorityIds': ['P2'], 'quizIds': ['Q1'], 'includeSelfCheck': False,
                'scope': 'Mean, median and outliers', 'practiceReviews': [{'quizId': 'Q1', 'text': 'Revisit mean versus median.'}]}
        generator._plan = plan
        seen = []
        async def provider(prompt):
            seen.append(json.loads(prompt))
            yield json.dumps({'records': [json.loads(r) for r in records().splitlines()]})
        generator._stream_with_openai = generator._stream_with_anthropic = provider
        events = [json.loads(e) async for e in generator.generate_study_sheet_stream('All statistics', user_request=request,
                  chat_context={'study_sheet_context': evidence})]
        sheet = events[-1]['studySheet']
        self.assertEqual(events[-1]['status'], 'study_sheet_complete')
        self.assertEqual(len(planning[0]['priorities']), 2)  # current request deduplicated
        self.assertEqual({s['id'] for s in sheet['sources']}, {'P2', 'Q1'})
        self.assertEqual([b['tone'] for b in sheet['sections'][0]['blocks']], ['teacher', 'practice'])
        self.assertNotIn('quizzes', seen[0])
        self.assertNotIn('Sociology', json.dumps(seen[0]))

    async def test_requested_french_overrides_the_app_language_hint(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        generator = SimpleStudySheetGenerator.__new__(SimpleStudySheetGenerator)
        generator.session = SimpleNamespace(vectorstore=None, documents=[], file_insights={})
        generator._plan = AsyncMock(return_value={'language': 'french', 'priorityIds': [], 'quizIds': [],
            'includeSelfCheck': False, 'practiceReviews': []})
        async def provider(prompt):
            self.assertEqual(json.loads(prompt)['language'], 'french')
            yield records('Filtration rénale')
        generator._stream_with_openai = generator._stream_with_anthropic = provider
        events = [json.loads(e) async for e in generator.generate_study_sheet_stream('Renal filtration', 'en',
            user_request='Refais-la en français, sans questions.')]
        self.assertEqual(events[-1]['studySheet']['language'], 'french')

    def test_plan_requires_selected_personalization_and_respects_no_questions(self):
        from services.studysheet_simple import SimpleStudySheetGenerator
        plan = {'priorityIds': ['P1'], 'quizIds': [], 'includeSelfCheck': False}
        sheet = {'sections': [{'blocks': [{'kind': 'paragraph', 'text': 'Generic', 'sourceIds': []}]}]}
        with self.assertRaises(ValueError): SimpleStudySheetGenerator._validate_plan(sheet, plan)
        sheet['sections'][0]['blocks'].append({'kind': 'callout', 'tone': 'teacher', 'sourceIds': ['P1']})
        SimpleStudySheetGenerator._validate_plan(sheet, plan)
        sheet['sections'][0]['blocks'].append({'kind': 'self_check', 'sourceIds': []})
        with self.assertRaises(ValueError): SimpleStudySheetGenerator._validate_plan(sheet, plan)


if __name__ == '__main__': unittest.main()
