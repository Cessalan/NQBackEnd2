import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.document_understanding import (MaterialError, all_material_texts,
    analyse_document, evidence, sections, validate_section, understand_session, VERSION)
from services.material_practice import (allocate_slots, generate_planned_question,
    normalize_question, plan_key, prepare_plan, stream_material_practice)
from services import practice_profile as pp
from services.quiz_tutor import requested_question_total


def material(name='Module 1.docx', identifier='g1', quote='The clinical model defines health as the absence of signs and symptoms of disease.'):
    goal = {'id': identifier, 'topic': 'Health models', 'outcome': 'Recognize the clinical model',
            'reasoning': 'recognition', 'evidence': {'filename': name, 'start': 0, 'end': len(quote), 'quote': quote}}
    return {'filename': name, 'fingerprint': 'abc', 'goals': [goal], 'signals': [], 'examples': [], 'complete': True}


class UnderstandingTests(unittest.TestCase):
    def test_every_character_is_analysed_in_order_even_beyond_old_context_limit(self):
        text = 'First objective.\n' + 'Nursing content.\n' * 4000 + 'Final exam: family function is excluded.'
        visited = []
        async def fake(system, payload):
            visited.append(payload['material'])
            return {'purpose': 'Teaching', 'goals': [], 'examples': [], 'signals': [], 'warnings': []}
        result = asyncio.run(analyse_document(text, 'large.docx', call=fake))
        self.assertTrue(result['complete'])
        self.assertEqual(result['characters'], len(text))
        rebuilt = ''
        for part in sections(text):
            self.assertIn(part['text'], visited)
            rebuilt += part['text'][max(0, len(rebuilt) - part['start']):]
        self.assertEqual(rebuilt, text)
        self.assertIn('family function is excluded.', visited[-1])

    def test_evidence_is_exact_and_addressable(self):
        part = {'text': 'SBAR means Situation, Background, Assessment, Recommendation.', 'start': 100}
        ref = evidence('Situation, Background', part, 'notes.pdf')
        self.assertEqual(ref['start'], 111)
        self.assertEqual(ref['end'], 132)
        self.assertIsNone(evidence('An invented teaching statement', part, 'notes.pdf'))

    def test_pdf_spacing_and_decomposed_french_accents_keep_exact_source_offsets(self):
        raw = 'Pre\u0301fixe. E\u0301valuation\u00a0du patient :\n\n  ABCDE est la se\u0301quence initiale. Fin.'
        quoted = 'Évaluation du patient : ABCDE est la séquence initiale.'
        part = {'text': raw, 'start': 120}
        ref = evidence(quoted, part, 'evaluation.pdf')
        expected = 'E\u0301valuation\u00a0du patient :\n\n  ABCDE est la se\u0301quence initiale.'
        self.assertEqual(ref['quote'], expected)
        self.assertEqual(raw[ref['start'] - 120:ref['end'] - 120], expected)
        self.assertEqual(ref['start'], 120 + raw.index('E\u0301valuation'))

    def test_typography_matching_never_accepts_changed_facts_or_spliced_passages(self):
        part = {'text': 'Dose : 10 mg. Première étape. Ne pas ignorer la sécurité. Dernière étape.', 'start': 0}
        for quote in ['Dose : 100 mg.', 'Dose : 10 mcg.', 'dose : 10 mg.',
                      'Première étape. Dernière étape.', 'Première étape… Dernière étape.',
                      'Ignorer la sécurité.', 'Ｄｏｓｅ : 10 mg.']:
            self.assertIsNone(evidence(quote, part, 'notes.pdf'), quote)

    def test_layout_normalized_example_stems_are_replaced_by_the_original_text(self):
        raw = 'Quelle\n  est la première étape de la séquence ABCDE ?'
        part = {'text': raw, 'start': 50, 'end': 50 + len(raw)}
        quote = 'Quelle est la première étape de la séquence ABCDE ?'
        result = validate_section({'examples': [{'stem': quote, 'quote': quote,
            'format': 'mcq', 'options': ['A', 'B']}]}, part, 'evaluation.pdf')
        self.assertEqual(result['examples'][0]['stem'], raw)
        self.assertEqual(result['examples'][0]['evidence']['quote'], raw)

    def test_retry_identifies_rejected_record_then_accepts_only_corrected_source_evidence(self):
        quote = 'ABCDE est la séquence enseignée dans ce cours.'
        observed = []
        async def fake(system, payload):
            observed.append(payload)
            return {'goals': [{'topic': 'ABCDE', 'outcome': 'Reconnaître la séquence',
                'quote': quote if 'repair' in payload else 'Invented medical facts'}]}
        result = asyncio.run(analyse_document(quote, 'evaluation.pdf', call=fake))
        self.assertEqual(len(observed), 2)
        self.assertNotIn('repair', observed[0])
        self.assertIn('goals[0].quote', observed[1]['repair']['validationError'])
        self.assertEqual(observed[1]['repair']['invalidRecord']['quote'], 'Invented medical facts')
        self.assertTrue(result['complete'])
        self.assertEqual(result['goals'][0]['evidence']['quote'], quote)

    def test_terminal_error_has_section_and_record_path_but_no_rejected_content(self):
        invented = 'Private rejected phrase which is not in the document.'
        async def fake(*args):
            return {'signals': [{'kind': 'exclusion', 'subject': 'ABCDE', 'quote': invented}]}
        with self.assertRaises(MaterialError) as raised:
            asyncio.run(analyse_document('Actual teaching material.', 'evaluation.pdf', call=fake))
        self.assertIn('section 1/1', str(raised.exception))
        self.assertIn('signals[0].quote', str(raised.exception))
        self.assertNotIn(invented, str(raised.exception))

    def test_failed_section_cancels_other_analysis_calls_and_does_not_save_a_cache(self):
        cancelled = []
        async def fake(system, payload):
            if payload['section'] == 1:
                await asyncio.sleep(0)
                return {'goals': [{'topic': 'Invented', 'outcome': 'Invented', 'quote': 'Fabricated medical content'}]}
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(payload['section'])
                raise
        with patch('services.document_understanding.load_analysis', return_value=None), \
             patch('services.document_understanding.save_analysis') as save:
            with self.assertRaises(MaterialError):
                asyncio.run(analyse_document('Teaching material. ' * 1800, 'large.pdf', chat_id='test', call=fake))
        self.assertTrue(cancelled)
        save.assert_not_called()

    def test_unsupported_analysis_never_commits_partial_success(self):
        async def fake(*args):
            return {'goals': [{'topic': 'Sepsis', 'outcome': 'Treat sepsis', 'quote': 'Invented medical facts'}]}
        with self.assertRaises(MaterialError):
            asyncio.run(analyse_document('Actual notes about health models.', 'notes.docx', call=fake))

    def test_every_indexed_file_is_read_without_similarity_search(self):
        docs = {
            'a': SimpleNamespace(page_content='abcdefghij', metadata={'source': 'one', 'chunk_index': 0, 'start_index': 0}),
            'b': SimpleNamespace(page_content='hijklmnop', metadata={'source': 'one', 'chunk_index': 1, 'start_index': 7}),
            'c': SimpleNamespace(page_content='Last module', metadata={'source': 'two', 'chunk_index': 0, 'start_index': 0})}
        session = SimpleNamespace(vectorstore=SimpleNamespace(index_to_docstore_id={0:'a', 1:'b', 2:'c'},
            docstore=SimpleNamespace(search=lambda key: docs[key])), practice_profile={})
        self.assertEqual(all_material_texts(session), {'one': 'abcdefghijklmnop', 'two': 'Last module'})
        session.practice_profile = {'source': {'files': ['missing']}}
        with self.assertRaises(MaterialError):
            all_material_texts(session)

    def test_missing_upload_does_not_fall_back_to_general_generation(self):
        with self.assertRaises(MaterialError):
            all_material_texts(SimpleNamespace(vectorstore=None))

    def test_revised_upload_replaces_earlier_chunks_and_gaps_are_rejected(self):
        docs = [SimpleNamespace(page_content=text, metadata={'source':'notes', 'chunk_index':0,
                'start_index':start, 'material_revision':rev}) for text,start,rev in
                [('Obsolete teaching',0,'old'), ('Revised teaching',0,'new')]]
        session = SimpleNamespace(vectorstore=SimpleNamespace(index_to_docstore_id={i:i for i in range(2)},
                                  docstore=SimpleNamespace(search=lambda key:docs[key])), practice_profile={})
        self.assertEqual(all_material_texts(session), {'notes':'Revised teaching'})
        docs[1].metadata['start_index'] = 100
        with self.assertRaises(MaterialError): all_material_texts(session)

    def test_overlap_examples_are_deduplicated_by_source_location(self):
        from services.document_understanding import merge_sections
        text = 'A heart sound is objective data. True or false?'
        part = {'start': 0, 'end': len(text), 'text': text}
        payload = {'examples': [{'stem': text, 'quote': text, 'format': 'true_false', 'options': ['True', 'False']}]}
        validated = validate_section(payload, part, 'module.docx')
        self.assertEqual(len(merge_sections('module.docx', text, [validated, validated])['examples']), 1)

    def test_an_unreadable_figure_cannot_be_marked_as_complete(self):
        async def fake(*args): return {'goals':[],'signals':[],'examples':[],'warnings':[]}
        analysis = asyncio.run(analyse_document('An objective. [UNREADABLE FIGURE: Figure 1]', 'notes.docx',call=fake))
        self.assertFalse(analysis['complete'])
        self.assertIn('UNREADABLE FIGURE',analysis['warnings'][0])

    def test_legacy_upload_is_reanalysed_from_original_and_reused(self):
        from services.document_understanding import fingerprint
        original = 'Full original notes, figure labels and final exam annotation.'
        doc = SimpleNamespace(page_content='Old text-only index', metadata={'source':'notes.docx'})
        session = SimpleNamespace(chat_id='chat',user_language='en',material_analysis={},practice_profile={},
                  vectorstore=SimpleNamespace(index_to_docstore_id={0:'doc'},docstore=SimpleNamespace(search=lambda _:doc)))
        observed = []
        async def analyse(text, filename, **kwargs):
            observed.append(text)
            return {**material(filename), 'version':VERSION,'fingerprint':fingerprint(text),
                    'indexFingerprint':kwargs['index_fingerprint']}
        with patch('services.document_understanding.read_original_upload',return_value=original) as read, \
             patch('services.document_understanding.load_original_analysis',return_value=None), \
             patch('services.document_understanding.analyse_document',analyse):
            asyncio.run(understand_session(session))
            asyncio.run(understand_session(session))
        self.assertEqual(observed,[original]); read.assert_called_once_with('chat','notes.docx')

    def test_legacy_upload_without_original_does_not_claim_complete_analysis(self):
        doc = SimpleNamespace(page_content='Old text-only index',metadata={'source':'notes.docx'})
        session = SimpleNamespace(chat_id='chat',user_language='en',material_analysis={},practice_profile={},
                  vectorstore=SimpleNamespace(index_to_docstore_id={0:'doc'},docstore=SimpleNamespace(search=lambda _:doc)))
        with patch('services.document_understanding.read_original_upload',side_effect=MaterialError('Missing original')), \
             patch('services.document_understanding.load_original_analysis',return_value=None):
            with self.assertRaises(MaterialError): asyncio.run(understand_session(session))


class PlanningTests(unittest.TestCase):
    def test_fifty_slots_cover_all_modules_and_interleave_batches(self):
        goals = {f'g{i}': material(f'Module {i}.docx', f'g{i}')['goals'][0] for i in range(4)}
        allocations = [{'goalId': key, 'weight': 1, 'reasoning': ['recognition', 'application']} for key in goals]
        slots = allocate_slots(allocations, goals, {}, 50, ['mcq', 'sata'], 'medium')
        self.assertEqual(len(slots), 50)
        self.assertEqual({s['goalId'] for s in slots[:5]}, set(goals))
        self.assertEqual([s['index'] for s in slots], list(range(50)))
        self.assertEqual({s['reasoning'] for s in slots}, {'recognition', 'application'})

    def test_many_outcomes_in_first_module_do_not_monopolize_first_batch(self):
        goals = {f'g{i}':material(f'Module {i//6}.docx',f'g{i}')['goals'][0] for i in range(24)}
        slots = allocate_slots([{'goalId':g} for g in goals], goals, {}, 50, ['mcq'], 'medium')
        self.assertEqual(len({s['source']['filename'] for s in slots[:4]}), 4)

    def test_match_examples_can_be_switched_off(self):
        goal = material()['goals'][0]
        examples = {'a':{'id':'a', 'format':'mcq', 'reasoning':'recognition'}}
        slots = allocate_slots([{'goalId':'g1'}], {'g1':goal}, examples, 4, ['mcq','sata'], 'medium', False)
        self.assertEqual({s['format'] for s in slots}, {'mcq','sata'})
        self.assertTrue(all(s['exampleId'] is None for s in slots))

    def test_cancelled_generation_delivers_no_question(self):
        async def collect():
            return [e async for e in stream_material_practice(session=SimpleNamespace(practice_profile={}),
                topic='all',difficulty='medium',num_questions=5,question_types=['mcq'],cancelled=lambda:True)]
        self.assertEqual(asyncio.run(collect()), [])

    def test_examples_guide_format_but_explicit_choices_win(self):
        goal = material()['goals'][0]
        examples = {'a': {'id': 'a', 'format': 'true_false', 'reasoning': 'recognition'}}
        allocation = [{'goalId': 'g1'}]
        slots = allocate_slots(allocation, {'g1':goal}, examples, 5, ['mcq'], 'easy')
        self.assertTrue(all(s['format'] == 'mcq' for s in slots))
        slots = allocate_slots(allocation, {'g1':goal}, examples, 5, ['true_false'], 'easy')
        self.assertTrue(all(s['exampleId'] == 'a' for s in slots))
        slots = allocate_slots(allocation, {'g1':goal}, examples, 12, ['true_false','matrix'], 'easy')
        self.assertEqual({s['format'] for s in slots}, {'true_false','matrix'})

    def test_unknown_learning_goals_are_rejected(self):
        with self.assertRaises(MaterialError):
            allocate_slots([{'goalId':'invented'}], {}, {}, 50, ['mcq'], 'medium')

    def test_a_plan_containing_an_explicitly_excluded_goal_is_rejected(self):
        async def fake(*args): return {'allocations':[{'goalId':'g1'}],'excludedGoalIds':['g1']}
        settings={'total':50,'formats':['mcq'],'difficulty':'medium','matchExamples':True}
        with self.assertRaises(MaterialError): asyncio.run(prepare_plan([material()],settings,call=fake))

    def test_material_and_style_changes_invalidate_plan(self):
        a = material()
        self.assertNotEqual(plan_key([a], {'scope':'all'}), plan_key([{**a, 'fingerprint':'changed'}], {'scope':'all'}))
        self.assertNotEqual(plan_key([a], {'scope':'all'}), plan_key([a], {'scope':'all', 'formats':['sata']}))

    def test_exact_student_request_and_correction_are_remembered(self):
        request = 'I would like you to provide a 50 question practice test covering all modules.'
        self.assertEqual(requested_question_total(request), 50)
        self.assertEqual(pp.parse_changes(request)['requested_total'], 50)
        change = pp.parse_changes('Can you structure these questions more to align with how the example questions that are in my notes are?')
        self.assertTrue(change['match_examples'])
        profile, _ = pp.merge({}, {**change, 'generation_instructions':'Use only my notes.'})
        continued, _ = pp.merge(profile, pp.parse_changes('more'))
        self.assertEqual(continued['generationInstructions'], 'Use only my notes.')
        self.assertTrue(continued['matchExamples'])

    def test_continuation_uses_slots_after_loaded_questions_and_same_plan(self):
        analysis = material()
        goal = analysis['goals'][0]
        slots = allocate_slots([{'goalId':'g1'}], {'g1':goal}, {}, 50, ['mcq'], 'medium')
        settings = {'scope':'all', 'total':50, 'formats':['mcq'], 'difficulty':'medium', 'language':'en',
                    'matchExamples':True, 'instructions':'', 'emphasis':''}
        plan = {'id':plan_key([analysis],settings), 'slots':[{k:v for k,v in s.items() if k!='source'} for s in slots],
                'settings':settings, 'total':50, 'summary':'All modules', 'coverage':{'Health models':50}, 'formats':{'mcq':50}, 'uncertainties':[]}
        assigned = []
        async def understand(*args): return [analysis]
        async def generate(slot, *args, **kwargs):
            assigned.append(slot['index']); return {'question':f'Q{slot["index"]}'}
        async def collect():
            return [e async for e in stream_material_practice(session=SimpleNamespace(chat_id='chat',user_language='en',practice_profile={'requestedTotal':50}),
                topic='all',difficulty='medium',num_questions=5,question_types=['mcq'],index_offset=5,plan_id=plan['id'])]
        with patch('services.material_practice.understand_session', understand), patch('services.material_practice.load_plan', return_value=plan), patch('services.material_practice.generate_planned_question', generate):
            events = asyncio.run(collect())
        self.assertEqual(assigned, [5,6,7,8,9])
        self.assertEqual(next(e['plan']['total'] for e in events if e['status']=='practice_plan_ready'),50)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.analysis = material()
        self.slot = allocate_slots([{'goalId':'g1'}], {'g1':self.analysis['goals'][0]}, {}, 1, ['mcq'], 'medium')[0]
        self.payload = {'question':'Which model defines health as the absence of disease?',
            'options':['Clinical model','Other model','Another model','Fourth model'], 'correctIndices':[0],
            'rationale':'The notes define the clinical model this way.',
            'sourceQuotes':[self.slot['source']['quote']]}

    def test_fabricated_citations_and_invalid_keys_are_rejected(self):
        for patch_data in ({'sourceQuotes':['Invented quote about sepsis.']}, {'correctIndices':[True]}, {'correctIndices':[9]}):
            with self.assertRaises(MaterialError):
                normalize_question({**self.payload,**patch_data},self.slot,[self.slot['source']],'plan')

    def test_semantic_validator_blocks_off_source_questions_even_with_real_quote(self):
        async def fake(system,payload):
            if 'Verify a generated' in system:
                return {'supported':False,'reason':'Digoxin is not taught in these notes.'}
            return {**self.payload,'question':'What is the therapeutic digoxin level?'}
        with self.assertRaises(MaterialError):
            asyncio.run(generate_planned_question(self.slot,[self.analysis],'plan','en',[],call=fake))

    def test_rejected_question_is_repaired_before_delivery(self):
        attempts = []
        async def fake(system,payload):
            if 'Verify a generated' in system:
                return {'supported':len(attempts)==2,'reason':'Repair the wording.'}
            attempts.append(payload)
            return self.payload
        question = asyncio.run(generate_planned_question(self.slot,[self.analysis],'plan','en',[],call=fake))
        self.assertEqual(attempts[1]['repair'],'Repair the wording.')
        self.assertTrue(question['metadata']['sourceVerified'])
        self.assertEqual(question['metadata']['slotIndex'],0)

    def test_true_false_uses_existing_single_choice_renderer(self):
        question = normalize_question({**self.payload,'options':['True','False']},
            {**self.slot,'format':'true_false'},[self.slot['source']],'plan')
        self.assertEqual(question['questionType'],'mcq')
        self.assertEqual(question['metadata']['sourceFormat'],'true_false')
        self.assertEqual(len(question['options']),2)

    def test_matrix_retains_rows_and_correct_columns(self):
        payload = {**self.payload, 'questionType':'matrix',
                   'columns':[{'id':'c1','label':'Clinical'},{'id':'c2','label':'Other'}],
                   'rows':[{'id':f'r{i}','text':f'Situation {i}','correctColumnId':'c1',
                            'explanation':'The notes define this model.'} for i in range(3)]}
        result = normalize_question(payload,{**self.slot,'format':'matrix'},[self.slot['source']],'plan')
        self.assertEqual(result['questionType'],'matrix')
        self.assertEqual(result['rows'][0]['correctColumnId'],'c1')
        payload['rows'][0]['correctColumnId'] = 'missing'
        with self.assertRaises(MaterialError):
            normalize_question(payload,{**self.slot,'format':'matrix'},[self.slot['source']],'plan')

    def test_unfolding_validates_each_item_and_preserves_case_renderer(self):
        payload = {**self.payload, 'scenario':{'patientInfo':'A fictional assessment',
                    'items':[{**self.payload,'questionType':'mcq','clinicalData':{'assessment':'Source-based observation'}} for _ in range(6)]}}
        result = normalize_question(payload,{**self.slot,'format':'unfoldingcase'},[self.slot['source']],'plan')
        self.assertEqual(result['questionType'],'unfoldingCase')
        self.assertEqual(result['scenario']['items'][0]['answer'], 'A) Clinical model')
        payload['scenario']['items'][5]['sourceQuotes'] = ['Invented evidence']
        with self.assertRaises(MaterialError):
            normalize_question(payload,{**self.slot,'format':'unfoldingcase'},[self.slot['source']],'plan')

    def test_quote_with_collapsed_layout_resolves_to_exact_source_text(self):
        raw = 'The clinical model defines health' + chr(10) + chr(10) + '  as the absence of signs and symptoms of disease.'
        passage = {'filename': 'Module 1.docx', 'start': 0, 'end': len(raw), 'quote': raw}
        payload = {**self.payload, 'sourceQuotes': ['The clinical model defines health as the absence of signs and symptoms of disease.']}
        question = normalize_question(payload, self.slot, [passage], 'plan')
        self.assertEqual(question['metadata']['sourceQuotes'], [raw])
        for changed in ['The clinical model defines health as the presence of signs and symptoms of disease.',
                        'the clinical model defines health']:
            with self.assertRaises(MaterialError):
                normalize_question({**self.payload, 'sourceQuotes': [changed]}, self.slot, [passage], 'plan')


class BatchResilienceTests(unittest.TestCase):
    def setUp(self):
        self.analysis = material()
        goal = self.analysis['goals'][0]
        slots = allocate_slots([{'goalId': 'g1'}], {'g1': goal}, {}, 8, ['mcq'], 'medium')
        settings = {'scope': 'all', 'total': 8, 'formats': ['mcq'], 'difficulty': 'medium', 'language': 'en',
                    'matchExamples': True, 'instructions': '', 'emphasis': ''}
        self.plan = {'id': plan_key([self.analysis], settings), 'slots': [{k: v for k, v in s.items() if k != 'source'} for s in slots],
                     'settings': settings, 'total': 8, 'summary': 'All', 'coverage': {'Health models': 8}, 'formats': {'mcq': 8}, 'uncertainties': []}
        self.session = SimpleNamespace(chat_id='chat', user_language='en', practice_profile={'requestedTotal': 8})

    def collect(self, generate, num_questions=3, index_offset=0):
        async def understand(*args): return [self.analysis]
        async def run():
            return [e async for e in stream_material_practice(session=self.session, topic='all', difficulty='medium',
                num_questions=num_questions, question_types=['mcq'], index_offset=index_offset, plan_id=self.plan['id'])]
        with patch('services.material_practice.understand_session', understand),              patch('services.material_practice.load_plan', return_value=self.plan),              patch('services.material_practice.generate_planned_question', generate):
            return asyncio.run(run())

    def test_unsupported_slot_is_skipped_and_batch_still_fills(self):
        async def generate(slot, *args, **kwargs):
            if slot['index'] == 1:
                raise MaterialError('A question could not be supported by your material. Please retry.')
            return {'question': f'Q{slot["index"]}'}
        events = self.collect(generate)
        delivered = [e for e in events if e['status'] == 'question_ready']
        self.assertEqual([q['question']['question'] for q in delivered], ['Q0', 'Q2', 'Q3'])
        self.assertEqual([q['index'] for q in delivered], [0, 1, 2])
        self.assertEqual(events[-1]['status'], 'quiz_complete')
        self.assertEqual(events[-1]['total_generated'], 3)
        self.assertEqual(events[-1]['skipped'], 1)
        self.assertFalse(any(e['status'] == 'error' for e in events))

    def test_repeated_question_is_skipped_not_fatal(self):
        async def generate(slot, *args, **kwargs):
            return {'question': 'Same wording' if slot['index'] < 2 else f'Q{slot["index"]}'}
        events = self.collect(generate, num_questions=2)
        delivered = [e['question']['question'] for e in events if e['status'] == 'question_ready']
        self.assertEqual(delivered, ['Same wording', 'Q2'])
        self.assertEqual(events[-1]['skipped'], 1)

    def test_batch_with_nothing_supported_still_reports_an_error(self):
        async def generate(slot, *args, **kwargs):
            raise MaterialError('A question could not be supported by your material. Please retry.')
        with self.assertRaises(MaterialError):
            self.collect(generate)



class MainTopicTests(unittest.TestCase):
    def test_goals_carry_their_heading_and_fall_back_to_topic(self):
        text = 'Primary survey. Airway: look, listen, feel. Breathing: count the rate.'
        part = {'text': text, 'start': 0, 'end': len(text)}
        payload = {'purpose': 'p', 'signals': [], 'examples': [], 'warnings': [], 'goals': [
            {'mainTopic': ' Primary  survey ', 'topic': 'Airway assessment', 'outcome': 'Assess the airway', 'quote': 'Airway: look, listen, feel.'},
            {'topic': 'Breathing assessment', 'outcome': 'Count the rate', 'quote': 'Breathing: count the rate.'}]}
        result = validate_section(payload, part, 'abcde.pdf')
        self.assertEqual(result['goals'][0]['mainTopic'], 'Primary survey')
        self.assertEqual(result['goals'][1]['mainTopic'], 'Breathing assessment')

    def test_heading_spellings_from_concurrent_sections_are_unified(self):
        from services.document_understanding import unify_main_topics
        goals = [{'mainTopic': 'Primary Survey', 'topic': 'Airway'}, {'mainTopic': 'primary survey', 'topic': 'Breathing'},
                 {'mainTopic': 'Secondary survey', 'topic': 'MIST'}]
        unify_main_topics(goals)
        self.assertEqual([g['mainTopic'] for g in goals], ['Primary Survey', 'Primary Survey', 'Secondary survey'])

    def test_display_insights_speaks_in_main_topics_with_subtopics(self):
        from services.document_understanding import display_insights, main_topics
        analysis = {'complete': True, 'sections': 1, 'warnings': [], 'examples': [], 'goals': [
            {'mainTopic': 'Primary survey', 'topic': 'Airway assessment', 'outcome': 'Assess the airway', 'evidence': {'quote': 'a'}},
            {'mainTopic': 'Primary survey', 'topic': 'Breathing assessment', 'outcome': 'Count the rate', 'evidence': {'quote': 'b'}},
            {'mainTopic': 'Secondary survey', 'topic': 'MIST history', 'outcome': 'Take a MIST history', 'evidence': {'quote': 'c'}},
            {'topic': 'Legacy goal', 'outcome': 'Old analysis without heading', 'evidence': {'quote': 'd'}}]}
        groups = main_topics(analysis)
        self.assertEqual([g['title'] for g in groups], ['Primary survey', 'Secondary survey', 'Legacy goal'])
        self.assertEqual(groups[0]['subtopics'], ['Airway assessment', 'Breathing assessment'])
        self.assertEqual(groups[2]['subtopics'], [], 'a goal whose heading is itself has no subtopic')
        shown = display_insights(analysis)
        self.assertEqual(shown['topics'], ['Primary survey', 'Secondary survey', 'Legacy goal'])
        self.assertEqual(shown['insights'][0]['key_points'], ['Assess the airway', 'Count the rate'])
        self.assertEqual(shown['insights'][0]['subtopics'], ['Airway assessment', 'Breathing assessment'])
        self.assertEqual(len(shown['concepts']), 4)


if __name__ == '__main__':
    unittest.main()
