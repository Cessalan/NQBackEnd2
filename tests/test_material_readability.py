import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from services.document_understanding import (MaterialError, VERSION, apply_readability,
    fingerprint, load_analysis, merge_sections, understand_session)
from services.material_practice import plan_key


class ReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.native = 'The fictional Alpha card belongs to group One. The Beta card belongs to group Two.\n'
        self.text = ('[Page 1]\n' + self.native + '[PDF page 1: visual description; UNCERTAIN]\n'
                     'An uncertain invented value appears here.\n\n[Page 2]\nThe Gamma card belongs to group Three.\n')
        self.start = self.text.index('[PDF page')
        self.end = self.text.index('[Page 2]')
        def goal(name, quote):
            start = self.text.index(quote)
            return {'id':name, 'topic':'Cards', 'outcome':name,
                    'evidence':{'filename':'cards.pdf','start':start,'end':start+len(quote),'quote':quote},
                    'context':{'filename':'cards.pdf','start':0,'end':len(self.text),'quote':self.text}}
        self.goals = [goal('alpha',self.native),goal('uncertain','An uncertain invented value appears here.'),
                      goal('gamma','The Gamma card belongs to group Three.')]
        self.section = {'start':0,'end':len(self.text),'text':self.text,'goals':self.goals,
                        'signals':[],'examples':[],'purpose':'Cards','warnings':[]}

    def test_native_text_is_usable_but_uncertain_visual_facts_and_context_are_excluded(self):
        result = merge_sections('cards.pdf',self.text,[self.section])
        self.assertFalse(result['complete'])
        self.assertTrue(result['practiceReady'])
        self.assertEqual([g['id'] for g in result['goals']],['alpha','gamma'])
        for goal in result['goals']:
            ref = goal['context']
            self.assertEqual(ref['quote'],self.text[ref['start']:ref['end']])
            self.assertNotIn('invented value',ref['quote'])
        self.assertTrue(any('excluded' in warning for warning in result['warnings']))
        self.assertEqual(len(self.section['goals']),3)  # original analysis is retained

    def test_unreadable_or_uncertain_scanned_page_still_blocks_practice(self):
        for text in [self.text+'[UNREADABLE FIGURE: Figure 2]',
                     self.text.replace(self.native,''),self.text+'[UNCERTAIN CONTENT: table]']:
            with self.subTest(text=text[-80:]):
                self.assertFalse(merge_sections('cards.pdf',text,[self.section])['practiceReady'])

    def test_example_with_uncertain_answer_is_excluded_even_if_stem_is_readable(self):
        example = {'id':'example','stem':'Choose a card','format':'mcq','evidence':self.goals[0]['evidence'],
                   'answerEvidence':self.goals[1]['evidence']}
        result = merge_sections('cards.pdf',self.text,[{**self.section,'examples':[example]}])
        self.assertEqual(result['examples'],[])

    def test_delimited_visual_does_not_swallow_following_native_content(self):
        text = self.text.replace('\n\n[Page 2]', '\n[End visual description]')
        quote = 'The Gamma card belongs to group Three.'
        start = text.index(quote)
        goal = {**self.goals[2], 'evidence':{'filename':'cards.pdf','start':start,'end':start+len(quote),'quote':quote},
                'context':{'filename':'cards.pdf','start':0,'end':len(text),'quote':text}}
        result = merge_sections('cards.pdf',text,[{**self.section,'goals':[goal]}])
        self.assertTrue(result['practiceReady'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['goals'][0]['evidence']['quote'],quote)
        self.assertNotIn('invented value',result['goals'][0]['context']['quote'])

    def test_old_incomplete_saved_analysis_is_reused_without_a_model_call(self):
        digest = fingerprint(self.text)
        manifest = {'complete':False,'fingerprint':digest,'version':VERSION,'sections':1,
                    'characters':len(self.text)}
        ref = SimpleNamespace(get=lambda:SimpleNamespace(to_dict=lambda:manifest),
            collection=lambda _:SimpleNamespace(document=lambda _:SimpleNamespace(
                get=lambda:SimpleNamespace(to_dict=lambda:self.section))))
        with patch('services.document_understanding._ref',return_value=ref):
            result=load_analysis('chat','cards.pdf',digest)
            self.assertTrue(result['practiceReady'])
            self.assertFalse(result['complete'])
            manifest['characters']+=1
            self.assertIsNone(load_analysis('chat','cards.pdf',digest))

    def test_existing_in_memory_failure_recovers_without_reupload_or_model_call(self):
        original = {'version':VERSION,'filename':'cards.pdf','fingerprint':fingerprint(self.text),
                    'complete':False,'goals':self.goals,'signals':[],'examples':[],'warnings':[]}
        session=SimpleNamespace(chat_id='chat',user_language='en',material_analysis={'cards.pdf':original},vectorstore=None)
        with patch('services.document_understanding.all_material_texts',return_value={'cards.pdf':self.text}), \
             patch('services.document_understanding.analyse_document') as analyse:
            result=asyncio.run(understand_session(session))
        analyse.assert_not_called()
        self.assertTrue(result[0]['practiceReady'])
        self.assertNotEqual(plan_key([original],{}),plan_key(result,{}))

    def test_legacy_index_uses_saved_original_offsets_to_recover_uncertainty(self):
        result=merge_sections('cards.pdf',self.text,[self.section])
        index_text='Legacy index without the full page layout.'
        old={**result,'complete':False,'indexFingerprint':fingerprint(index_text)}
        old.pop('readabilityVersion'); old.pop('practiceReady')
        session=SimpleNamespace(chat_id='chat',user_language='en',material_analysis={'cards.pdf':old},vectorstore=None)
        with patch('services.document_understanding.all_material_texts',return_value={'cards.pdf':index_text}), \
             patch('services.document_understanding.load_analysis',return_value=result) as load, \
             patch('services.document_understanding.analyse_document') as analyse:
            recovered=asyncio.run(understand_session(session))
        load.assert_called_once_with('chat','cards.pdf',fingerprint(self.text))
        analyse.assert_not_called()
        self.assertTrue(recovered[0]['practiceReady'])


if __name__ == '__main__': unittest.main()
